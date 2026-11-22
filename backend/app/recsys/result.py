"""Ranked output of a system with a per-stage trace (explanations in Phase 3 are built from it)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTrace:
    """What one stage did for one request.

    ``local_ms`` is locally measured compute time (varies run to run). ``upstream_latency_ms`` and
    ``upstream_cost_usd`` are *nominal*: the recorded latency and cost of the LLM / embedding / rank
    calls the stage made, also when they were served from cache, so a replay still reports what a live
    request would cost. ``contrib`` holds per-route contributions (route id -> number or dict) for the
    routes the stage passed on; ``info`` holds stage-level facts (sizes, parameters).
    """

    stage: str
    n_in: int = 0
    n_out: int = 0
    local_ms: float = 0.0
    upstream_latency_ms: float = 0.0
    upstream_cost_usd: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)
    contrib: dict[int, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Deterministic summary (no local timing): stored in run files."""
        return {
            "stage": self.stage,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "upstream_latency_ms": round(self.upstream_latency_ms, 3),
            "upstream_cost_usd": round(self.upstream_cost_usd, 9),
            "info": self.info,
        }


@dataclass
class Recommendation:
    request_id: str
    system: str
    route_ids: list[int]
    scores: list[float]
    trace: list[StageTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.route_ids) != len(self.scores):
            raise ValueError("route_ids and scores differ in length")
        if len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("duplicate route ids in a ranking")

    @property
    def local_ms(self) -> float:
        return sum(t.local_ms for t in self.trace)

    @property
    def upstream_latency_ms(self) -> float:
        return sum(t.upstream_latency_ms for t in self.trace)

    @property
    def upstream_cost_usd(self) -> float:
        return sum(t.upstream_cost_usd for t in self.trace)

    def contributions(self, route_id: int) -> dict[str, Any]:
        """Per-stage contribution for one returned route (explanation input)."""
        return {t.stage: t.contrib[route_id] for t in self.trace if route_id in t.contrib}
