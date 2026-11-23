"""What the served system does when a user skips the optional profile fields (docs/RECSYS.md section 12).

Each variant runs ``recsys_serve`` on the request with slots removed before the pipeline sees it:
``recsys_serve_nocap`` (no distance cap), ``recsys_serve_nohome`` (no home, so no radius filter either)
and ``recsys_serve_noslots`` (neither). Everything else (vector, tab, intent, feedback) is unchanged. These
are offline checks, not product variants: the evaluation still judges against the persona's real limits.
``recsys_v1_nocapslot`` is the same cap removal on ``recsys_v1``, whose ranker uses ``length_cap_ratio``
(it never saw that feature missing in training): the direct test of D19's robustness argument.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..catalog import Catalog
from ..query import UserQuery
from ..result import Recommendation
from .base import register_system

DROPS: dict[str, dict[str, Any]] = {
    "recsys_serve_nocap": {"max_distance_km": None},
    "recsys_serve_nohome": {"home": None, "max_travel_km": None},
    "recsys_serve_noslots": {"max_distance_km": None, "home": None, "max_travel_km": None},
    "recsys_v1_nocapslot": {"max_distance_km": None},
}
BASES = {"recsys_v1_nocapslot": "recsys_v1"}


class EmptySlots:
    def __init__(self, catalog: Catalog, name: str, **kw: Any) -> None:
        from . import build_system

        base = BASES.get(name, "recsys_serve")
        self.name, self.base, self.drop = name, base, DROPS[name]
        self.pipe = build_system(base, catalog, **kw)

    def config(self) -> dict[str, Any]:
        return {**self.pipe.config(), "system": self.name, "wraps": self.base, "slots_removed": sorted(self.drop)}

    async def arecommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        rec = await self.pipe.arecommend(replace(query, **self.drop), k)
        rec.system = self.name
        return rec

    def recommend(self, query: UserQuery, k: int = 10) -> Recommendation:
        from app.llm import run_sync

        return run_sync(self.arecommend(query, k))


def _factory(name: str):
    def make(catalog: Catalog, **cfg: Any) -> EmptySlots:
        return EmptySlots(catalog, name, **cfg)

    return make


for _n, _d in DROPS.items():
    register_system(
        _n,
        description=f"{BASES.get(_n, 'recsys_serve')} with {', '.join(sorted(_d))} removed (empty-slot check)",
        uses_llm=True,
    )(_factory(_n))
