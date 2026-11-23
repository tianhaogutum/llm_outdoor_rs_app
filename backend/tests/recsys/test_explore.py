"""Phase 5 exploration: tag clusters, Beta posteriors, seeded Thompson sampling, the reserved slot, filters."""

from __future__ import annotations

import numpy as np
import pytest

from app.llm import LLMClient, LLMConfig
from app.recsys import build_system
from app.recsys.constraints import HardConstraints
from app.recsys.explain import explain
from app.recsys.explore import EXPLORE_REASON, ExploreConfig, pick, posteriors, tag_clusters
from app.recsys.feedback import FeedbackConfig, WeightedEvent
from app.recsys.query import FeedbackEvent, UserQuery

from .factory import synthetic_catalog


@pytest.fixture(scope="module")
def cat():
    return synthetic_catalog(400, seed=5)


@pytest.fixture
def llm(tmp_path):
    return LLMClient(LLMConfig(cache_dir=tmp_path / "cache"))


def _we(cat, rid: int, reason: str, w: float = 1.0) -> WeightedEvent:
    return WeightedEvent(rid, cat.index_of(rid), reason, 0.0, w)


def test_clusters_are_deterministic_and_cover_the_catalogue(cat):
    cfg = ExploreConfig(k=4)
    c1 = tag_clusters(cat, cfg)
    assert len(c1.assign) == len(cat) and c1.n == cfg.k + 1 and sum(c1.sizes) == len(cat)
    assert c1.labels[-1] == "no theme tags"
    # same catalogue, fresh memo: identical assignment
    c2 = tag_clusters(synthetic_catalog(400, seed=5), cfg)
    assert np.array_equal(c1.assign, c2.assign) and c1.labels == c2.labels


def test_posteriors_count_completions_and_content_rejections_only(cat):
    cfg = ExploreConfig(k=4, prior_a=1.0, prior_b=1.0)
    cl = tag_clusters(cat, cfg)
    r1, r2 = int(cat.ids[0]), int(cat.ids[1])
    c1, c2 = int(cl.assign[0]), int(cl.assign[1])
    ab = posteriors(cl, [_we(cat, r1, "completed", 0.5), _we(cat, r2, "not-interested"), _we(cat, r2, "too-hard"),
                         _we(cat, r2, "too-far")], cfg)
    expect = np.ones((cl.n, 2))
    expect[c1, 0] += 0.5
    expect[c2, 1] += 1.0  # effort / distance reasons are not about the theme
    assert np.allclose(ab, expect)


def _ranked(cat, n=50):
    return list(range(n))


def test_pick_is_seeded_reserves_the_slot_and_respects_exclusions(cat):
    cfg = ExploreConfig(k=4)
    cl = tag_clusters(cat, cfg)
    elig = np.arange(len(cat))
    args = dict(events=[], ranked_idx=_ranked(cat), eligible_idx=elig, excluded_ids=set())
    a = pick(cat, cl, cfg, request_id="u1", **args)
    b = pick(cat, cl, cfg, request_id="u1", **args)
    assert a == b and a["reason"] == EXPLORE_REASON and a["position"] == 10
    assert a["index"] not in _ranked(cat)[:9]  # never one of the routes above the slot
    draws = {pick(cat, cl, cfg, request_id=f"u{i}", **args)["cluster"] for i in range(40)}
    assert len(draws) > 1  # the draw depends on the request
    # an excluded route is never picked, even if it is the cluster's best candidate
    ex = {int(cat.ids[a["index"]])}
    c = pick(cat, cl, cfg, request_id="u1", **{**args, "excluded_ids": ex})
    assert int(cat.ids[c["index"]]) not in ex
    # cold start only: enough feedback switches it off
    many = [_we(cat, int(cat.ids[i]), "completed") for i in range(cfg.max_events)]
    assert pick(cat, cl, cfg, request_id="u1", **{**args, "events": many}) is None


def test_strong_posterior_wins_the_slot(cat):
    cfg = ExploreConfig(k=4, max_events=100)
    cl = tag_clusters(cat, cfg)
    clusters = sorted({int(c) for c in cl.assign})
    assert len(clusters) >= 3
    good = clusters[0]

    def member(c):
        return int(cat.ids[int(np.flatnonzero(cl.assign == c)[0])])

    ev = [_we(cat, member(good), "completed")] * 3 + [_we(cat, member(c), "not-interested") for c in clusters[1:]] * 3
    wins = sum(
        pick(cat, cl, cfg, request_id=f"r{i}", events=ev, ranked_idx=[], eligible_idx=np.arange(len(cat)),
             excluded_ids=set())["cluster"] == good
        for i in range(30)
    )
    assert wins >= 20  # Beta(4, 1) against Beta(1, 4): Thompson almost always picks the good cluster


def _q(**kw) -> UserQuery:
    base = dict(
        request_id="p1",
        user_vector={"difficulty_range": [0, 2], "min_distance_km": 3, "max_distance_km": 15, "preferred_tags": []},
        category="hiking",
        home=(48.0, 11.0),
        max_travel_km=25.0,
        max_distance_km=10.0,
        intent="An easy lake loop with a castle and a beer garden",
    )
    base.update(kw)
    return UserQuery(**base)


@pytest.mark.parametrize("intent", ["An easy lake loop with a castle and a beer garden", None])
def test_pipeline_slot_passes_filters_and_is_marked(cat, llm, intent):
    sysm = build_system("recsys_v1_fb_explore", cat, llm=llm)
    for rid in ("a", "b", "c", "d"):
        q = _q(request_id=rid, intent=intent)
        rec = sysm.recommend(q, 10)
        hc = HardConstraints.from_query(q)
        assert all(hc.ok(cat.get(r)) for r in rec.route_ids)  # the explored route too
        ex = explain(rec)
        marked = [r for r in ex.routes if r.stages.explore is not None]
        assert len(marked) == 1 and marked[0].rank == 10 and marked[0].stages.explore.reason == EXPLORE_REASON
        assert sysm.recommend(q, 10).route_ids == rec.route_ids  # seeded: reproducible
        # the first nine are the feedback system's own top nine
        base = build_system("recsys_v1_fb", cat, llm=llm).recommend(q, 10)
        assert rec.route_ids[:9] == base.route_ids[:9]
    # feedback events update the posterior used for the draw and are excluded from the slot
    done = rec.route_ids[0]
    q2 = _q(request_id="d", intent=intent, feedback=(FeedbackEvent(done, "completed", 7.0),))
    rec2 = sysm.recommend(q2, 10)
    assert done not in rec2.route_ids
    info = next(t for t in rec2.trace if t.stage == "explore").info
    assert info["events"] == 1 and info["active"]
    # exploration with the feedback loop off still reads the events for its posterior
    off = build_system("recsys_v1", cat, llm=llm, explore=ExploreConfig(), feedback=None)
    assert next(t for t in off.recommend(q2, 10).trace if t.stage == "explore").info["events"] == 1
    assert FeedbackConfig().half_life_days > 0
