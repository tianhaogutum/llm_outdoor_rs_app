"""Explanation schema: what every stage contributed to each returned route.

The pipeline writes one entry per stage into ``StageTrace.contrib[route_id]``; :func:`explain` turns a
:class:`~app.recsys.result.Recommendation` into a validated :class:`RecommendationExplanation`, the
form the API returns and a UI renders (``model_dump()`` gives plain JSON; ``explanation_json_schema()``
the JSON schema). A stage that did not run for a request is absent (``None``), never faked.
Field-by-field documentation and a real example: docs/RECSYS.md section "Explanations".
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .result import Recommendation

SCHEMA_VERSION = "recsys.explanation/v1"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CategoryCheck(_M):
    applied: bool
    ok: bool
    value: str = Field(description="the route's category")
    allowed: Optional[list[str]] = Field(description="categories of the requested tab; null if no tab given")


class RadiusCheck(_M):
    applied: bool
    ok: bool = Field(description="start within the user's own radius")
    distance_km: Optional[float] = Field(description="great-circle distance home -> route start")
    limit_km: Optional[float] = Field(description="the user's travel radius")
    limit_used_km: Optional[float] = Field(description="radius actually applied (wider if the set was tiny)")
    relaxed: bool = Field(description="admitted only because the radius was widened")


class DistanceCapCheck(_M):
    applied: bool
    ok: bool
    length_km: float
    limit_km: Optional[float]


class FilterContribution(_M):
    verdict: Literal["pass", "pass_relaxed", "fail"]
    category: CategoryCheck
    radius: RadiusCheck
    distance_cap: DistanceCapCheck


class MatchedTerm(_M):
    term: str
    contribution: float


class BM25Contribution(_M):
    score: float
    rank: int = Field(description="rank by BM25 among the eligible routes (1 = best)")
    matched_terms: list[MatchedTerm]


class DenseContribution(_M):
    similarity: float = Field(description="cosine similarity of intent and route embeddings")
    rank: int


class FusionContribution(_M):
    rrf_score: float
    rank: int
    k: int
    inputs: dict[str, int] = Field(description="the route's rank in each fused list (absent = not retrieved)")


class RuleContribution(_M):
    difficulty: float
    distance: float
    tags: float
    total: float
    feedback_penalty: Optional[float] = None


class RankApiContribution(_M):
    score: float
    rank: int
    model: str


class FeatureContribution(_M):
    name: str
    value: Optional[float] = Field(description="feature value; null = missing")
    contribution: float = Field(description="Saabas path attribution to the ranker score")


class LTRContribution(_M):
    score: float
    rank: int
    bias: float = Field(description="model output before any split; score = bias + sum(contributions)")
    model: str
    features: list[FeatureContribution] = Field(description="all features, largest |contribution| first")


class MMRContribution(_M):
    lam: float = Field(alias="lambda")
    relevance: float = Field(description="ranker score min-max normalised over the candidates")
    max_similarity: float = Field(description="max cosine to the routes already placed above")
    penalty: float
    mmr_score: float
    rank_before: int
    rank_after: int

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StageContributions(_M):
    filter: Optional[FilterContribution] = None
    v0_rules: Optional[RuleContribution] = None
    bm25: Optional[BM25Contribution] = None
    dense: Optional[DenseContribution] = None
    fusion: Optional[FusionContribution] = None
    rank_api: Optional[RankApiContribution] = None
    ltr: Optional[LTRContribution] = None
    mmr: Optional[MMRContribution] = None


class RouteExplanation(_M):
    route_id: int
    rank: int
    score: float
    stages: StageContributions


class StageSummary(_M):
    stage: str
    n_in: int
    n_out: int
    local_ms: float
    upstream_latency_ms: float
    upstream_cost_usd: float
    info: dict[str, Any]


class RecommendationExplanation(_M):
    schema_version: Literal["recsys.explanation/v1"] = SCHEMA_VERSION
    request_id: str
    system: str
    routes: list[RouteExplanation]
    stages: list[StageSummary]


def explain(rec: Recommendation, top: Optional[int] = None) -> RecommendationExplanation:
    ids = rec.route_ids[:top] if top else rec.route_ids
    routes = []
    for rank, (rid, s) in enumerate(zip(ids, rec.scores, strict=False), start=1):
        routes.append(
            RouteExplanation(
                route_id=int(rid), rank=rank, score=float(s), stages=StageContributions(**rec.contributions(rid))
            )
        )
    stages = [
        StageSummary(
            stage=t.stage,
            n_in=t.n_in,
            n_out=t.n_out,
            local_ms=round(t.local_ms, 3),
            upstream_latency_ms=round(t.upstream_latency_ms, 3),
            upstream_cost_usd=round(t.upstream_cost_usd, 9),
            info=t.info,
        )
        for t in rec.trace
    ]
    return RecommendationExplanation(request_id=rec.request_id, system=rec.system, routes=routes, stages=stages)


def explanation_json_schema() -> dict[str, Any]:
    return RecommendationExplanation.model_json_schema(by_alias=True)
