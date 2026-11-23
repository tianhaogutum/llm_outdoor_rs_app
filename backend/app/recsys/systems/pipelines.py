"""Phase 3 systems: each ablation row is a complete :class:`~app.recsys.pipeline.Pipeline` with its config.

The ablation chain (docs/RESULTS.md) adds one component per row::

    v0_rules -> v1_filters -> v2_bm25 -> v3_dense -> v4_rrf -> v5_rank_api -> v6_ltr -> v7_mmr

``v2_bm25`` and ``v3_dense`` rank the eligible set by that generator's own score (standalone), ``v4_rrf``
by the fused score. ``recsys_v1`` is the Phase 3 default (the end of the ablation); the API serves
``SERVED_SYSTEM`` = ``recsys_serve`` (docs/DECISIONS.md D25). Every factory accepts config
overrides (``build_system("v4_rrf", catalog, rrf_k=30)``) plus ``llm=``, ``doc_embeddings=`` and
``ltr_model=`` (a loaded :class:`~app.recsys.ltr.TreeEnsemble`) to inject resources.
"""

from __future__ import annotations

from typing import Any

from ..bm25 import BM25Config
from ..catalog import Catalog
from ..explore import ExploreConfig
from ..feedback import FeedbackConfig
from ..pipeline import Pipeline, PipelineConfig
from .base import register_system

# Chosen on dev (eval/results/_cg_dev.json; docs/RECSYS.md section "Candidate generation").
BM25_DEFAULT = BM25Config(stem=True, glossary=False, negation=True, expand=True, w_title=0.5, w_tags=2.0)

CONFIGS: dict[str, PipelineConfig] = {
    "v1_filters": PipelineConfig(generators=(), ranker="v0", bm25=BM25_DEFAULT),
    "v2_bm25": PipelineConfig(generators=("bm25",), ranker="none", bm25=BM25_DEFAULT),
    "v3_dense": PipelineConfig(generators=("dense",), ranker="none", bm25=BM25_DEFAULT),
    "v4_rrf": PipelineConfig(generators=("bm25", "dense"), ranker="none", bm25=BM25_DEFAULT),
    "v5_rank_api": PipelineConfig(generators=("bm25", "dense"), ranker="api", bm25=BM25_DEFAULT),
    "v6_ltr": PipelineConfig(generators=("bm25", "dense"), ranker="ltr", ltr_model="v1", bm25=BM25_DEFAULT),
    "v6_ltr_noapi": PipelineConfig(generators=("bm25", "dense"), ranker="ltr", ltr_model="v1_noapi", bm25=BM25_DEFAULT),
}
# MMR lambda chosen on dev (eval/results/_mmr_sweep.json; docs/RECSYS.md section "Diversity")
MMR_LAMBDA = 0.8
CONFIGS["v7_mmr"] = CONFIGS["v6_ltr"].with_(mmr_lambda=MMR_LAMBDA)
# The Phase 3 default (docs/RECSYS.md section 7): the learned ranker without the ranker-API feature (the API's
# gain on dev CV does not pay for its latency and cost), then MMR. Not what the API serves: see SERVED_SYSTEM.
DEFAULT_SYSTEM = "recsys_v1"
CONFIGS[DEFAULT_SYSTEM] = CONFIGS["v6_ltr_noapi"].with_(mmr_lambda=MMR_LAMBDA)
# Robustness rows (docs/DECISIONS.md D19): the default with ranker models trained without the features that
# encode how the simulator draws distance caps (``length_cap_ratio``) or the whole suspicious group.
CONFIGS["recsys_v1_nocap"] = CONFIGS[DEFAULT_SYSTEM].with_(ltr_model="v1_noapi_nocap")
CONFIGS["recsys_v1_nosusp"] = CONFIGS[DEFAULT_SYSTEM].with_(ltr_model="v1_noapi_nosusp")
# Phase 5 (docs/RECSYS.md section "Feedback and exploration"). Without feedback events and without
# exploration each of these returns exactly what recsys_v1 returns.
FEEDBACK = FeedbackConfig()  # chosen on dev personas (D21): route effects + light slot steps
# the full loop at the dev grid's light setting (Rocchio, re-scoring, near-duplicates, slots): no better than
# FEEDBACK on dev; kept as a registered system so the simulation reports it on test too
LIGHT_LOOP = FeedbackConfig(beta=0.25, gamma=0.1, eta_emb=0.1, eta_tag=0.1, nu_dup=0.15)
ROUTE_EFFECTS_ONLY = FeedbackConfig(
    beta=0.0, gamma=0.0, eta_emb=0.0, eta_tag=0.0, nu_dup=0.0, difficulty_step=0.0, cap_step=0.0, radius_step=0.0
)
CONFIGS["recsys_v1_fb"] = CONFIGS[DEFAULT_SYSTEM].with_(feedback=FEEDBACK)
CONFIGS["recsys_v1_fb_explore"] = CONFIGS["recsys_v1_fb"].with_(explore=ExploreConfig())
CONFIGS["recsys_v1_fb_route"] = CONFIGS[DEFAULT_SYSTEM].with_(feedback=ROUTE_EFFECTS_ONLY)
CONFIGS["recsys_v1_fb_light"] = CONFIGS[DEFAULT_SYSTEM].with_(feedback=LIGHT_LOOP)
# The system the API serves (docs/DECISIONS.md D25): ranker without length_cap_ratio (D19), the feedback loop as
# chosen on dev (D21), exploration slot on (D22). A composition of registered parts; nothing new is trained.
SERVED_SYSTEM = "recsys_serve"
CONFIGS["recsys_serve_noexplore"] = CONFIGS["recsys_v1_nocap"].with_(feedback=FEEDBACK)
CONFIGS[SERVED_SYSTEM] = CONFIGS["recsys_serve_noexplore"].with_(explore=ExploreConfig())

DESCRIPTIONS: dict[str, str] = {
    "v1_filters": "hard filters (tab, radius, distance cap) in front of the v0_rules scorer",
    "v2_bm25": "filters + BM25F over title/description/tags, ranked by BM25",
    "v3_dense": "filters + dense retrieval (exact cosine), ranked by similarity",
    "v4_rrf": "filters + BM25 and dense fused by reciprocal rank, ranked by RRF",
    "v5_rank_api": "v4_rrf top-50 re-ranked by the semantic ranker API",
    "v6_ltr": "v4_rrf top-50 re-ranked by the learned ranker (LambdaMART) with the ranker-API score as a feature",
    "v6_ltr_noapi": "v4_rrf top-50 re-ranked by the learned ranker without the ranker-API feature (no API call)",
    "v7_mmr": "v6_ltr followed by MMR diversification",
    "recsys_v1": "DEFAULT: filters, BM25 + dense fused by RRF, learned ranker without the ranker API, MMR",
    "recsys_v1_nocap": "recsys_v1 with a ranker trained without length_cap_ratio (robustness row)",
    "recsys_v1_nosusp": "recsys_v1 with a ranker trained without slot ratios, home distance and intent km "
    "(robustness row)",
    "recsys_v1_fb": "recsys_v1 + feedback loop as chosen on dev (route effects, slot adjustments)",
    "recsys_v1_fb_light": "recsys_v1 + full light feedback loop (slots, Rocchio, re-scoring, near-duplicates)",
    "recsys_v1_fb_explore": "recsys_v1_fb + one exploration slot (Thompson sampling over tag clusters)",
    "recsys_serve": "SERVED: recsys_v1_nocap ranker + feedback loop (dev choice) + exploration slot (D25)",
    "recsys_serve_noexplore": "recsys_serve without the exploration slot (its feedback-on, no-exploration twin)",
    "recsys_v1_fb_route": "recsys_v1 + route effects only (recent completions and rejections not shown again)",
}


def _factory(name: str):
    def make(catalog: Catalog, *, llm=None, doc_embeddings=None, ltr_model=None, **overrides: Any) -> Pipeline:
        cfg = CONFIGS[name].with_(**overrides) if overrides else CONFIGS[name]
        return Pipeline(catalog, cfg, name=name, doc_embeddings=doc_embeddings, ltr_model=ltr_model, llm=llm)

    return make


for _name in CONFIGS:
    register_system(_name, description=DESCRIPTIONS[_name], uses_llm=True)(_factory(_name))
