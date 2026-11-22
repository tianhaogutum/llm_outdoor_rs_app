"""Deterministic hard-constraint predicates.

Used by the evaluation metric (violation rate, rule-graded judgements) now and by the Phase 3a hard
filter later, so both apply exactly the same rules:

- ``category``: the route's ``category_name`` belongs to the UI tab (:data:`CATEGORY_TABS`, identical to
  ``CATEGORY_MAPPING`` of the v0_rules service);
- ``radius``: great-circle distance from ``home`` to the route's start ``<= max_travel_km``;
- ``distance_cap``: route length ``<= max_distance_km``.

A constraint whose slot is unset (None) is not applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import Catalog, RouteRecord
    from .query import UserQuery

# Same mapping as app.services.recommendation_service.CATEGORY_MAPPING (a test pins the equality).
CATEGORY_TABS: dict[str, tuple[str, ...]] = {
    "running": ("Jogging", "Trail running"),
    "hiking": ("Theme trail", "Hiking trail"),
    "cycling": ("Cycling", "Mountainbiking", "Long distance cycling"),
}

EARTH_RADIUS_KM = 6371.0088
CONSTRAINT_NAMES = ("category", "radius", "distance_cap")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (mean Earth radius); works on scalars and numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(d) if np.ndim(d) == 0 else d


@dataclass(frozen=True)
class HardConstraints:
    category: Optional[str] = None
    home: Optional[tuple[float, float]] = None
    max_travel_km: Optional[float] = None
    max_distance_km: Optional[float] = None

    @classmethod
    def from_query(cls, q: UserQuery) -> HardConstraints:
        return cls(category=q.category, home=q.home, max_travel_km=q.max_travel_km, max_distance_km=q.max_distance_km)

    # ------------------------------------------------------- single route
    def violations(self, route: RouteRecord) -> list[str]:
        """Names of the constraints ``route`` violates (empty list = feasible), in CONSTRAINT_NAMES order."""
        out = []
        if self.category and route.category_name not in CATEGORY_TABS.get(self.category, ()):
            out.append("category")
        if self.home is not None and self.max_travel_km is not None:
            if haversine_km(self.home[0], self.home[1], route.start_lat, route.start_lon) > self.max_travel_km:
                out.append("radius")
        if self.max_distance_km is not None and route.length_km > self.max_distance_km:
            out.append("distance_cap")
        return out

    def ok(self, route: RouteRecord) -> bool:
        return not self.violations(route)

    # ----------------------------------------------------------- vectorised
    def masks(self, catalog: Catalog) -> dict[str, np.ndarray]:
        """Per-constraint boolean masks over the catalogue (True = satisfied)."""
        n = len(catalog)
        m = {name: np.ones(n, dtype=bool) for name in CONSTRAINT_NAMES}
        if self.category:
            m["category"] = np.isin(catalog.category_name, list(CATEGORY_TABS.get(self.category, ())))
        if self.home is not None and self.max_travel_km is not None:
            d = haversine_km(self.home[0], self.home[1], catalog.start_lat, catalog.start_lon)
            m["radius"] = d <= self.max_travel_km
        if self.max_distance_km is not None:
            m["distance_cap"] = catalog.length_km <= self.max_distance_km
        return m

    def mask(self, catalog: Catalog) -> np.ndarray:
        ms = self.masks(catalog)
        out = np.ones(len(catalog), dtype=bool)
        for v in ms.values():
            out &= v
        return out

    def home_distance_km(self, catalog: Catalog) -> Optional[np.ndarray]:
        if self.home is None:
            return None
        return haversine_km(self.home[0], self.home[1], catalog.start_lat, catalog.start_lon)
