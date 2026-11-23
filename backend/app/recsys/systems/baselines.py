"""Baselines: ``random`` and ``v0_rules``."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from ..catalog import Catalog
from ..constraints import CATEGORY_TABS
from ..query import UserQuery
from ..result import Recommendation, StageTrace
from .base import register_system

# Fixed reference time for feedback ages, so offline runs are deterministic.
REFERENCE_NOW = datetime(2026, 11, 22, tzinfo=UTC)


def _tab_indices(catalog: Catalog, category: str) -> np.ndarray:
    def build():
        names = list(CATEGORY_TABS.get(category, ()))
        if not names:  # unknown tab: whole catalogue (same as the DB path)
            return np.arange(len(catalog))
        return np.flatnonzero(np.isin(catalog.category_name, names))

    return catalog.memo(f"tab:{category}", build)


def request_seed(seed: int, request_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{request_id}".encode()).digest()[:8], "big")


class RandomSystem:
    """Uniform random order over the request's category tab, seeded per request id."""

    name = "random"

    def __init__(self, catalog: Catalog, seed: int = 0) -> None:
        self.catalog = catalog
        self.seed = int(seed)

    def config(self) -> dict[str, Any]:
        return {"system": self.name, "seed": self.seed, "sees": ["category"]}

    def recommend(self, query: UserQuery, k: int) -> Recommendation:
        t0 = time.perf_counter()
        idx = _tab_indices(self.catalog, query.category)
        rng = np.random.default_rng(request_seed(self.seed, query.request_id))
        pick = idx[rng.permutation(len(idx))[:k]]
        ids = [int(self.catalog.ids[i]) for i in pick]
        scores = [round(1.0 - r / max(k, 1), 6) for r in range(len(ids))]
        tr = StageTrace("random", n_in=len(idx), n_out=len(ids), local_ms=(time.perf_counter() - t0) * 1000)
        return Recommendation(query.request_id, self.name, ids, scores, [tr])


class V0RulesSystem:
    """The product's current scorer via ``rank_routes_v0``: sees only the user vector and the category tab."""

    name = "v0_rules"

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def config(self) -> dict[str, Any]:
        from app.services.recommendation_service import SCORE_WEIGHTS

        return {
            "system": self.name,
            "ranker": "app.services.recommendation_service.rank_routes_v0",
            "weights": dict(SCORE_WEIGHTS),
            "sees": ["user_vector", "category", "feedback"],
            "feedback_reference_time": REFERENCE_NOW.isoformat(),
        }

    def _vectors(self, category: str) -> list[dict]:
        def build():
            out = []
            for i in _tab_indices(self.catalog, category):
                r = self.catalog.routes[int(i)]
                out.append({"id": r.id, **r.v0_vector()})
            return out  # id order, like the DB path (ties keep input order)

        return self.catalog.memo(f"v0vec:{category}", build)

    def recommend(self, query: UserQuery, k: int) -> Recommendation:
        from app.services.recommendation_service import rank_routes_v0

        t0 = time.perf_counter()
        routes = self._vectors(query.category)
        feedback = [
            {"route_id": f.route_id, "reason": f.reason, "created_at": REFERENCE_NOW - timedelta(days=f.age_days)}
            for f in query.feedback
            if f.reason in ("too-hard", "too-easy", "too-far", "not-interested")  # v0 has no completion signal
        ]
        ranked = rank_routes_v0(query.user_vector, routes, feedback, now=REFERENCE_NOW, limit=k)
        contrib = {
            e.route_id: {
                "difficulty": round(e.breakdown["difficulty"]["weighted_score"], 6),
                "distance": round(e.breakdown["distance"]["weighted_score"], 6),
                "tags": round(e.breakdown["tags"]["weighted_score"], 6),
                **({"feedback_penalty": e.breakdown["feedback_penalty"]} if "feedback_penalty" in e.breakdown else {}),
            }
            for e in ranked
        }
        tr = StageTrace(
            "v0_rules",
            n_in=len(routes),
            n_out=len(ranked),
            local_ms=(time.perf_counter() - t0) * 1000,
            contrib=contrib,
        )
        return Recommendation(
            query.request_id, self.name, [e.route_id for e in ranked], [round(e.score, 6) for e in ranked], [tr]
        )


@register_system("random", description="uniform random over the category tab, seeded per request")
def _make_random(catalog: Catalog, **cfg: Any) -> RandomSystem:
    return RandomSystem(catalog, **cfg)


@register_system("v0_rules", description="baseline rule scorer (rank_routes_v0): user vector + category tab")
def _make_v0(catalog: Catalog, **cfg: Any) -> V0RulesSystem:
    return V0RulesSystem(catalog, **cfg)
