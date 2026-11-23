"""Phase 5 feedback: decay on both signals, slot adjustments with clamps, single application, Rocchio,
route effects, re-scoring, and the pipeline integration."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.llm import LLMClient, LLMConfig
from app.recsys import Catalog, build_system
from app.recsys.constraints import HardConstraints, haversine_km
from app.recsys.explain import explain
from app.recsys.feedback import (
    FeedbackConfig,
    adjust_query,
    feedback_direction,
    feedback_events_from_rows,
    near_duplicate,
    rescore,
    rocchio,
    tag_affinity,
    weigh,
)
from app.recsys.query import FeedbackEvent, UserQuery

from .factory import route_record, synthetic_catalog

H = 60.0


def _cat() -> Catalog:
    """Hand-made catalogue: positions and lengths chosen so every rule can be checked by hand."""
    recs = [
        route_record(1, length_m=8000, tags=("Lake", "Beer garden"), lat=48.00, lon=11.00),
        route_record(2, length_m=20000, tags=("Summit route", "Viewpoint"), lat=48.10, lon=11.00),
        route_record(3, length_m=8200, tags=("Lake", "Playground"), lat=48.001, lon=11.001),  # near-dup of 1
        route_record(4, length_m=3000, tags=("Castle", "Museum"), lat=48.30, lon=11.00),
        route_record(5, length_m=9000, tags=("Hut", "Summit route"), lat=48.05, lon=11.05),
    ]
    return Catalog.from_records(recs)


def _q(**kw) -> UserQuery:
    base = dict(
        request_id="u1",
        user_vector={"difficulty_range": [1, 2], "min_distance_km": 5.0, "max_distance_km": 20.0,
                     "preferred_tags": ["scenic"]},
        category="hiking",
        home=(48.0, 11.0),
        max_travel_km=40.0,
        max_distance_km=24.0,
        intent="a lake walk",
    )
    base.update(kw)
    return UserQuery(**base)


def _ev(*items) -> tuple[FeedbackEvent, ...]:
    return tuple(FeedbackEvent(r, reason, age) for r, reason, age in items)


# --------------------------------------------------------------------------- decay
def test_one_true_half_life_for_both_signals():
    cat = _cat()
    cfg = FeedbackConfig(half_life_days=H)
    w = {e.reason: e.weight for e in weigh(cat, _ev((1, "completed", H), (2, "too-hard", H)), cfg)}
    assert w == {"completed": pytest.approx(0.5), "too-hard": pytest.approx(0.5)}
    w2 = [e.weight for e in weigh(cat, _ev((1, "completed", 2 * H), (2, "not-interested", 0)), cfg)]
    assert w2 == [pytest.approx(0.25), pytest.approx(1.0)]  # oldest first
    # unknown routes and reasons are dropped
    assert weigh(cat, _ev((999, "completed", 0), (1, "liked", 0)), cfg) == []


def test_events_from_stored_rows():
    now = datetime(2026, 11, 23, tzinfo=UTC)
    fb = [{"route_id": 2, "reason": "too-far", "created_at": now - timedelta(days=3)},
          {"route_id": 4, "reason": "spam", "created_at": now}]
    souv = [{"route_id": 1, "completed_at": (now - timedelta(days=10)).replace(tzinfo=None)}]  # naive = UTC
    ev = feedback_events_from_rows(fb, souv, now=now)
    assert ev == (FeedbackEvent(1, "completed", 10.0), FeedbackEvent(2, "too-far", 3.0))


# --------------------------------------------------------------------------- slots
def test_difficulty_shift_is_weighted_and_clamped():
    cat = _cat()
    cfg = FeedbackConfig(difficulty_step=0.5)
    st = adjust_query(cat, _q(feedback=_ev((2, "too-hard", 0))), cfg)
    assert st.query.user_vector["difficulty_range"] == [0.5, 1.5]
    st = adjust_query(cat, _q(feedback=_ev((2, "too-hard", H))), cfg)  # half weight
    assert st.query.user_vector["difficulty_range"] == [0.75, 1.75]
    # clamps: never below 0, width stays >= 1; never above 3
    many_hard = _ev(*[(2, "too-hard", 0)] * 6)
    assert adjust_query(cat, _q(feedback=many_hard), cfg).query.user_vector["difficulty_range"] == [0.0, 1.0]
    many_easy = _ev(*[(4, "too-easy", 0)] * 6)
    assert adjust_query(cat, _q(feedback=many_easy), cfg).query.user_vector["difficulty_range"] == [2.0, 3.0]
    # too-hard and too-easy cancel
    both = _ev((2, "too-hard", 0), (4, "too-easy", 0))
    assert "difficulty_range" not in adjust_query(cat, _q(feedback=both), cfg).slot_changes


def test_cap_and_radius_pulls_with_floors():
    cat = _cat()
    cfg = FeedbackConfig(cap_step=0.5, radius_step=0.5, floor_share=0.5)
    # route 2 is 20 km (>= 0.6 x 24): the cap moves half way to 18 km
    st = adjust_query(cat, _q(feedback=_ev((2, "too-hard", 0))), cfg)
    assert st.query.max_distance_km == pytest.approx(21.0)
    # a short route rejected as too hard does not touch the cap (effort, not length)
    assert adjust_query(cat, _q(feedback=_ev((4, "too-hard", 0))), cfg).query.max_distance_km == 24.0
    # floor: never below half the user's cap however often
    st = adjust_query(cat, _q(max_distance_km=24.0, feedback=_ev(*[(2, "too-hard", 0)] * 20)), cfg)
    assert st.query.max_distance_km == pytest.approx(18.0)  # converges to 0.9 x 20 km, above the floor 12
    st = adjust_query(cat, _q(max_distance_km=30.0, feedback=_ev(*[(5, "too-hard", 0)] * 30)), cfg)
    assert st.query.max_distance_km == pytest.approx(30.0)  # 9 km route < 0.6 x cap: untouched
    # too-far pulls the radius towards 0.9 x the route's distance from home, floor half the radius
    d4 = haversine_km(48.0, 11.0, 48.30, 11.00)
    st = adjust_query(cat, _q(feedback=_ev((4, "too-far", 0))), cfg)
    assert st.query.max_travel_km == pytest.approx(40.0 - 0.5 * (40.0 - 0.9 * d4), abs=1e-3)
    st = adjust_query(cat, _q(feedback=_ev(*[(1, "too-far", 0)] * 30)), cfg)
    assert st.query.max_travel_km == pytest.approx(20.0)  # route 1 starts at home: floor 0.5 x 40


def test_tags_follow_net_decayed_preference():
    cat = _cat()
    st = adjust_query(cat, _q(feedback=_ev((1, "completed", 0), (2, "not-interested", 0))), FeedbackConfig())
    pref = st.tag_pref
    assert pref["Lake"] > 0 and pref["Beer garden"] > 0 and pref["Summit route"] < 0 and pref["Viewpoint"] < 0
    tags = st.query.user_vector["preferred_tags"]
    assert "lake" in tags and "beer garden" in tags
    assert set(st.query.user_vector["avoided_tags"]) == {"Summit route", "Viewpoint"}
    assert tag_affinity(cat.get(3).tags, pref) > 0 > tag_affinity(cat.get(5).tags, pref)
    # completion and rejection of the same tag cancel to zero
    st = adjust_query(cat, _q(feedback=_ev((5, "completed", 0), (2, "not-interested", 0))), FeedbackConfig())
    assert st.tag_pref["Summit route"] == pytest.approx(0.0)


def test_feedback_is_applied_once_from_the_original_query():
    cat = _cat()
    cfg = FeedbackConfig(difficulty_step=0.5)
    q = _q(feedback=_ev((2, "too-hard", 0), (2, "too-hard", 0)))
    st = adjust_query(cat, q, cfg)
    # two events of weight 1 at step 0.5 shift exactly 1.0 (no compounding of n(n+1)/2 steps)
    assert st.query.user_vector["difficulty_range"] == [0.0, 1.0]
    assert st.query.feedback == ()
    again = adjust_query(cat, st.query, cfg)  # nothing left to apply
    assert again.query == st.query and again.slot_changes == {}
    # the caller's original query is untouched
    assert q.user_vector["difficulty_range"] == [1, 2] and q.max_distance_km == 24.0


# --------------------------------------------------------------------------- Rocchio
def test_rocchio_moves_towards_completions_and_away_from_rejections():
    cat = _cat()
    rng = np.random.default_rng(0)
    E = rng.normal(size=(len(cat), 16))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    q = E[cat.index_of(4)] + 0.1 * rng.normal(size=16)
    q = q / np.linalg.norm(q)
    cfg = FeedbackConfig(alpha=1.0, beta=0.5, gamma=0.5)
    st = adjust_query(cat, _q(feedback=_ev((1, "completed", 0), (2, "not-interested", 0))), cfg)
    q2, info = rocchio(q, E, st)
    assert np.linalg.norm(q2) == pytest.approx(1.0, abs=1e-6)
    i1, i2 = cat.index_of(1), cat.index_of(2)
    assert q2 @ E[i1] > q @ E[i1] and q2 @ E[i2] < q @ E[i2]
    assert info["n_pos"] == 1 and info["n_neg"] == 1
    expected = q + 0.5 * E[i1] - 0.5 * E[i2]
    assert np.allclose(q2, expected / np.linalg.norm(expected), atol=1e-6)
    # decay: an old completion moves the query less
    old = adjust_query(cat, _q(feedback=_ev((1, "completed", 3 * H))), cfg)
    new = adjust_query(cat, _q(feedback=_ev((1, "completed", 0))), cfg)
    assert rocchio(q, E, old)[0] @ E[i1] == pytest.approx(rocchio(q, E, new)[0] @ E[i1])  # centroid is a mean
    # effort reasons do not move the embedding
    st = adjust_query(cat, _q(feedback=_ev((2, "too-hard", 0), (4, "too-far", 0))), cfg)
    assert np.allclose(rocchio(q, E, st)[0], q)
    # no intent: the feedback direction alone
    st = adjust_query(cat, _q(intent=None, feedback=_ev((1, "completed", 0))), cfg)
    d = feedback_direction(E, st)
    assert np.allclose(d, E[i1]) and rocchio(None, E, st)[0] is not None


# --------------------------------------------------------------------------- route effects
def test_recent_completions_and_rejections_are_excluded_then_come_back():
    cat = _cat()
    cfg = FeedbackConfig(half_life_days=H, exclude_weight=0.25)
    st = adjust_query(cat, _q(feedback=_ev((1, "completed", 10), (2, "too-hard", 2 * H - 1))), cfg)
    assert st.excluded == {1: "completed", 2: "rejected: too-hard"}
    st = adjust_query(cat, _q(feedback=_ev((1, "completed", 2 * H + 1), (2, "not-interested", 3 * H))), cfg)
    assert st.excluded == {}  # an old thumbs-down no longer hides the route forever


def test_rescore_drops_excluded_and_demotes_near_duplicates():
    cat = _cat()
    cfg = FeedbackConfig(eta_emb=0.0, eta_tag=0.0, nu_dup=0.6)
    assert near_duplicate(cat, cat.index_of(1), cat.index_of(3), cfg)
    assert not near_duplicate(cat, cat.index_of(1), cat.index_of(5), cfg)
    st = adjust_query(cat, _q(feedback=_ev((1, "not-interested", 0))), cfg)
    idx = [cat.index_of(r) for r in (1, 3, 5, 2)]
    kept, scores, contrib, info = rescore(cat, idx, np.array([4.0, 3.0, 2.0, 1.0]), st, None)
    ids = [int(cat.ids[i]) for i in kept]
    assert 1 not in ids and info["removed"] == [1]
    assert ids == [5, 3, 2]  # 3 nearly repeats the rejected route: relevance 1.0 - 0.6 < 0.5 of route 5
    c3 = contrib[cat.index_of(3)]
    assert c3["near_duplicate_of"] == 1 and c3["duplicate_penalty"] == pytest.approx(0.6)
    assert list(scores) == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- pipeline
@pytest.fixture(scope="module")
def scat():
    return synthetic_catalog(400, seed=5)


@pytest.fixture
def llm(tmp_path):
    return LLMClient(LLMConfig(cache_dir=tmp_path / "cache"))


def _pq(**kw) -> UserQuery:
    base = dict(
        request_id="p1",
        user_vector={"difficulty_range": [0, 2], "min_distance_km": 3, "max_distance_km": 15, "preferred_tags": []},
        category="hiking",
        home=(48.0, 11.0),
        max_travel_km=40.0,
        max_distance_km=14.0,
        intent="An easy lake loop with a castle and a beer garden, around 8 km, no busy roads",
    )
    base.update(kw)
    return UserQuery(**base)


@pytest.mark.parametrize("intent", ["An easy lake loop with a castle and a beer garden", None])
def test_pipeline_feedback_stage(scat, llm, intent):
    q = _pq(intent=intent)
    base = build_system("recsys_v1", scat, llm=llm).recommend(q, 10)
    fbs = build_system("recsys_v1_fb_light", scat, llm=llm)
    assert fbs.recommend(q, 10).route_ids == base.route_ids  # no events: identical to the default
    ev = _ev((base.route_ids[0], "completed", 7), (base.route_ids[1], "not-interested", 7),
             (base.route_ids[2], "too-hard", 14))
    rec = fbs.recommend(replace(q, feedback=ev), 10)
    assert not set(rec.route_ids) & set(base.route_ids[:3])
    stages = [t.stage for t in rec.trace]
    assert "feedback" in stages and stages.index("feedback") < stages.index("mmr")
    fb = next(t for t in rec.trace if t.stage == "feedback")
    assert fb.info["events"] == {"completed": 1, "not-interested": 1, "too-hard": 1}
    hc = HardConstraints.from_query(q)
    assert all(hc.ok(scat.get(r)) for r in rec.route_ids)
    ex = explain(rec)
    assert all(r.stages.feedback is not None for r in ex.routes)
    if intent:
        dense = next(t for t in rec.trace if t.stage == "dense")
        assert dense.info["rocchio"]["n_pos"] == 1
    # the legacy path (no feedback config) is the control: feedback ignored outside the v0 ranker
    ctrl = build_system("recsys_v1", scat, llm=llm).recommend(replace(q, feedback=ev), 10)
    if intent:
        assert ctrl.route_ids == base.route_ids
