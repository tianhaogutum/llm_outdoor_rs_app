"""The Phase 3 pipelines end to end on a synthetic catalogue with the mock LLM provider (no network):
every registered system runs, every stage reports its contribution, explanations validate."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from app.llm import LLMClient, LLMConfig
from app.recsys import HardConstraints, UserQuery, build_system, list_systems
from app.recsys.explain import RecommendationExplanation, explain, explanation_json_schema
from app.recsys.systems.pipelines import CONFIGS
from tests.recsys.factory import synthetic_catalog

PIPELINES = sorted(CONFIGS)
EXPECTED_STAGES = {
    "v1_filters": {"filter", "v0_rules"},
    "v2_bm25": {"filter", "bm25"},
    "v3_dense": {"filter", "dense"},
    "v4_rrf": {"filter", "bm25", "dense", "fusion"},
    "v5_rank_api": {"filter", "bm25", "dense", "fusion", "rank_api"},
    "v6_ltr": {"filter", "bm25", "dense", "fusion", "rank_api", "ltr"},
    "v6_ltr_noapi": {"filter", "bm25", "dense", "fusion", "ltr"},
    "v7_mmr": {"filter", "bm25", "dense", "fusion", "rank_api", "ltr", "mmr"},
    "recsys_v1": {"filter", "bm25", "dense", "fusion", "ltr", "mmr"},
    "recsys_v1_nocap": {"filter", "bm25", "dense", "fusion", "ltr", "mmr"},
    "recsys_v1_nosusp": {"filter", "bm25", "dense", "fusion", "ltr", "mmr"},
}


@pytest.fixture(scope="module")
def cat():
    return synthetic_catalog(400, seed=5)


@pytest.fixture
def llm(tmp_path):
    return LLMClient(LLMConfig(cache_dir=tmp_path / "cache"))


def _q(intent="An easy lake loop with a castle and a beer garden, around 8 km, no busy roads", **kw):
    base = dict(
        request_id="p1",
        user_vector={"difficulty_range": [0, 2], "min_distance_km": 3, "max_distance_km": 15, "preferred_tags": []},
        category="hiking",
        home=(48.0, 11.0),
        max_travel_km=40.0,
        max_distance_km=14.0,
        intent=intent,
    )
    base.update(kw)
    return UserQuery(**base)


def test_registry_has_the_ablation_chain_and_the_default():
    assert set(PIPELINES) <= set(list_systems()) and set(EXPECTED_STAGES) == set(PIPELINES)


@pytest.mark.parametrize("name", PIPELINES)
def test_pipeline_end_to_end_with_explanations(name, cat, llm):
    q = _q()
    sysm = build_system(name, cat, llm=llm)
    rec = sysm.recommend(q, 10)
    hc = HardConstraints.from_query(q)
    assert 0 < len(rec.route_ids) <= 10 and all(hc.ok(cat.get(r)) for r in rec.route_ids)
    assert {t.stage for t in rec.trace} == EXPECTED_STAGES[name]
    ex = explain(rec)
    assert isinstance(ex, RecommendationExplanation) and len(ex.routes) == len(rec.route_ids)
    doc = json.loads(ex.model_dump_json(by_alias=True))
    for r in doc["routes"]:
        st = r["stages"]
        assert st["filter"]["verdict"] == "pass"
        assert st["filter"]["radius"]["distance_km"] <= 40.0 and st["filter"]["radius"]["limit_km"] == 40.0
        for s in EXPECTED_STAGES[name] - {"filter"}:
            assert st[s] is not None, (name, s)
        if "bm25" in EXPECTED_STAGES[name]:
            assert all(t["contribution"] > 0 for t in st["bm25"]["matched_terms"])
        if "fusion" in EXPECTED_STAGES[name]:
            assert st["fusion"]["inputs"] and st["fusion"]["k"] == CONFIGS[name].rrf_k
        if "ltr" in EXPECTED_STAGES[name]:
            lt = st["ltr"]
            assert lt["bias"] + sum(f["contribution"] for f in lt["features"]) == pytest.approx(lt["score"], abs=1e-4)
            assert {f["name"] for f in lt["features"]} == set(sysm._ltr.feature_names)
        if "mmr" in EXPECTED_STAGES[name]:
            assert st["mmr"]["rank_after"] == r["rank"] and st["mmr"]["lambda"] == CONFIGS[name].mmr_lambda
    # cost and latency are accounted per stage
    if "rank_api" in EXPECTED_STAGES[name]:
        assert any(t.stage == "rank_api" for t in rec.trace)
    cfg = sysm.config()
    assert cfg["system"] == name and json.dumps(cfg)  # JSON-serialisable, stored in results files


def test_ranked_orders_follow_the_stage_scores(cat, llm):
    rec = build_system("v5_rank_api", cat, llm=llm).recommend(_q(), 10)
    api = [rec.contributions(r)["rank_api"]["score"] for r in rec.route_ids]
    assert api == sorted(api, reverse=True)
    rec = build_system("v4_rrf", cat, llm=llm).recommend(_q(), 10)
    fused = [rec.contributions(r)["fusion"]["rank"] for r in rec.route_ids]
    assert fused == sorted(fused)
    rec = build_system("v6_ltr_noapi", cat, llm=llm).recommend(_q(), 10)
    assert rec.scores == sorted(rec.scores, reverse=True)


def test_no_intent_falls_back_to_the_rule_scorer(cat, llm):
    q = _q(intent=None)
    for name in ("v4_rrf", "v6_ltr", "recsys_v1"):
        rec = build_system(name, cat, llm=llm).recommend(q, 10)
        stages = [t.stage for t in rec.trace]
        assert "bm25" not in stages and "dense" not in stages and "rank_api" not in stages and "ltr" not in stages
        assert "v0_rules" in stages and rec.trace[0].info["fallback"] == "no_intent"
        assert rec.route_ids and llm.usage_report()["totals"].get("rank", {}).get("calls", 0) == 0
    v1 = build_system("v1_filters", cat, llm=llm).recommend(q, 10)
    assert build_system("v4_rrf", cat, llm=llm).recommend(q, 10).route_ids == v1.route_ids


def test_async_and_sync_entry_points_agree(cat, llm):
    sysm = build_system("recsys_v1", cat, llm=llm)
    a = sysm.recommend(_q(), 10)
    b = asyncio.run(sysm.arecommend(_q(), 10))
    assert a.route_ids == b.route_ids and a.scores == b.scores


def test_stage_switches_by_config(cat, llm):
    s = build_system("v4_rrf", cat, llm=llm, generators=("bm25",))
    assert {t.stage for t in s.recommend(_q(), 10).trace} == {"filter", "bm25"}
    s = build_system("v6_ltr_noapi", cat, llm=llm, mmr_lambda=0.5)
    assert "mmr" in {t.stage for t in s.recommend(_q(), 10).trace}
    s = build_system("v1_filters", cat, llm=llm, filters=False)
    rec = s.recommend(_q(), 10)
    assert "filter" in {t.stage for t in rec.trace} and rec.contributions(rec.route_ids[0]).get("filter") is None
    with pytest.raises(ValueError):
        build_system("v4_rrf", cat, llm=llm, ranker="nonsense")


def test_tiny_eligible_set_returns_a_short_list(cat, llm):
    q = _q(home=(40.0, 0.0), max_travel_km=5.0)  # nowhere near any route: nothing, even after relaxing
    rec = build_system("recsys_v1", cat, llm=llm).recommend(q, 10)
    assert rec.route_ids == [] and rec.trace[0].info["n_eligible"] == 0


def test_explanation_json_schema():
    sch = explanation_json_schema()
    assert sch["title"] == "RecommendationExplanation"
    defs = sch["$defs"]
    for name in (
        "FilterContribution",
        "BM25Contribution",
        "DenseContribution",
        "FusionContribution",
        "RankApiContribution",
        "LTRContribution",
        "MMRContribution",
        "RuleContribution",
    ):
        assert name in defs
    assert "lambda" in defs["MMRContribution"]["properties"]


def test_mock_document_embeddings_are_built_in_memory(cat, llm):
    s = build_system("v3_dense", cat, llm=llm)
    d = s.doc_embeddings()
    assert d.matrix.shape == (len(cat), 768) and np.allclose(np.linalg.norm(d.matrix, axis=1), 1, atol=1e-5)
    assert s.config()["doc_embeddings"]["provider"] == "mock"
