"""Phase 3 systems: each ablation row is a complete :class:`~app.recsys.pipeline.Pipeline` with its config.

The ablation chain (docs/RESULTS.md) adds one component per row::

    v0_rules -> v1_filters -> v2_bm25 -> v3_dense -> v4_rrf -> v5_rank_api -> v6_ltr -> v7_mmr

``v2_bm25`` and ``v3_dense`` rank the eligible set by that generator's own score (standalone), ``v4_rrf``
by the fused score. ``recsys_v1`` is the default system the API serves. Every factory accepts config
overrides (``build_system("v4_rrf", catalog, rrf_k=30)``) plus ``llm=``, ``doc_embeddings=`` and
``ltr_model=`` (a loaded :class:`~app.recsys.ltr.TreeEnsemble`) to inject resources.
"""

from __future__ import annotations

from typing import Any

from ..bm25 import BM25Config
from ..catalog import Catalog
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
}

DESCRIPTIONS: dict[str, str] = {
    "v1_filters": "hard filters (tab, radius, distance cap) in front of the v0_rules scorer",
    "v2_bm25": "filters + BM25F over title/description/tags, ranked by BM25",
    "v3_dense": "filters + dense retrieval (exact cosine), ranked by similarity",
    "v4_rrf": "filters + BM25 and dense fused by reciprocal rank, ranked by RRF",
    "v5_rank_api": "v4_rrf top-50 re-ranked by the semantic ranker API",
}


def _factory(name: str):
    def make(catalog: Catalog, *, llm=None, doc_embeddings=None, ltr_model=None, **overrides: Any) -> Pipeline:
        cfg = CONFIGS[name].with_(**overrides) if overrides else CONFIGS[name]
        return Pipeline(catalog, cfg, name=name, doc_embeddings=doc_embeddings, ltr_model=ltr_model, llm=llm)

    return make


for _name in CONFIGS:
    register_system(_name, description=DESCRIPTIONS[_name], uses_llm=True)(_factory(_name))
