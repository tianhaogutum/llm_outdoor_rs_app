"""Hard-filter stage (Phase 3a): category tab, radius from home, absolute distance cap.

Uses exactly the predicates of :class:`~app.recsys.constraints.HardConstraints`, so the filter and the
evaluation's violation metric / rule-graded judgements agree.

Policy (docs/RECSYS.md section "Hard filters"):

- **Missing slot**: a constraint whose slot is missing is not applied (no home or no radius -> no
  radius filter; no cap -> no length filter; no tab -> every category). The verdict says so
  (``"applied": false``) instead of pretending the route passed a check.
- **Unknown tab**: no route belongs to it, so the eligible set is empty (the API should validate the
  tab; the pipeline does not guess).
- **Tiny eligible set**: if fewer than ``min_eligible`` routes pass, *only the radius* is widened in
  steps of ``relax_factor`` up to ``relax_max_factor`` times the user's radius (the same idea the
  persona generator uses for sparse areas). The category is the user's tab and the distance cap is
  the user's absolute limit, so neither is ever relaxed. Routes admitted by a widened radius carry
  ``"relaxed": true`` and the limit actually used; the response can say "nothing closer than X km".
  If even the widest radius leaves fewer routes, the pipeline returns what there is (possibly fewer
  than k): an honest short list beats padding with infeasible routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .catalog import Catalog
from .constraints import CATEGORY_TABS, HardConstraints


@dataclass(frozen=True)
class FilterConfig:
    min_eligible: int = 10
    relax_factor: float = 1.25
    relax_max_factor: float = 2.0


@dataclass
class FilterResult:
    indices: np.ndarray  # catalogue indices of eligible routes, ascending
    home_km: Optional[np.ndarray]  # distance from home for every catalogue route (None without home)
    radius_used_km: Optional[float]
    info: dict[str, Any]

    def verdict(self, catalog: Catalog, hc: HardConstraints, i: int) -> dict[str, Any]:
        r = catalog.routes[i]
        allowed = list(CATEGORY_TABS.get(hc.category, ())) if hc.category else None
        out: dict[str, Any] = {
            "category": {
                "applied": hc.category is not None,
                "ok": allowed is None or r.category_name in allowed,
                "value": r.category_name,
                "allowed": allowed,
            },
            "radius": {
                "applied": hc.home is not None and hc.max_travel_km is not None,
                "ok": True,
                "distance_km": None if self.home_km is None else round(float(self.home_km[i]), 2),
                "limit_km": hc.max_travel_km,
                "limit_used_km": self.radius_used_km,
                "relaxed": False,
            },
            "distance_cap": {
                "applied": hc.max_distance_km is not None,
                "ok": hc.max_distance_km is None or r.length_km <= hc.max_distance_km,
                "length_km": round(r.length_km, 2),
                "limit_km": hc.max_distance_km,
            },
        }
        rad = out["radius"]
        if rad["applied"]:
            d = float(self.home_km[i])
            rad["ok"] = d <= float(hc.max_travel_km)
            rad["relaxed"] = not rad["ok"] and self.radius_used_km is not None and d <= self.radius_used_km
        passed = all(v["ok"] or (k == "radius" and v["relaxed"]) for k, v in out.items())
        out["verdict"] = "pass" if passed and not rad["relaxed"] else ("pass_relaxed" if passed else "fail")
        return out


def apply_filters(catalog: Catalog, hc: HardConstraints, cfg: FilterConfig = FilterConfig()) -> FilterResult:
    masks = hc.masks(catalog)
    base = masks["category"] & masks["distance_cap"]
    home_km = hc.home_distance_km(catalog)
    info: dict[str, Any] = {
        "applied": {
            "category": hc.category is not None,
            "radius": hc.home is not None and hc.max_travel_km is not None,
            "distance_cap": hc.max_distance_km is not None,
        },
        "failed_by": {k: int((~m).sum()) for k, m in masks.items()},
        "n_category": int(masks["category"].sum()),
    }
    radius_used = hc.max_travel_km
    if info["applied"]["radius"]:
        mask = base & (home_km <= hc.max_travel_km)
        steps = 0
        r = float(hc.max_travel_km)
        while (
            int(mask.sum()) < cfg.min_eligible
            and r * cfg.relax_factor <= hc.max_travel_km * cfg.relax_max_factor + 1e-9
        ):
            r *= cfg.relax_factor
            steps += 1
            mask = base & (home_km <= r)
        radius_used = round(r, 3)
        info["relax_steps"] = steps
        info["radius_used_km"] = radius_used
    else:
        mask = base
        info["relax_steps"] = 0
    idx = np.flatnonzero(mask)
    info["n_eligible"] = int(len(idx))
    info["short"] = bool(len(idx) < cfg.min_eligible)
    return FilterResult(idx, home_km, radius_used if info["applied"]["radius"] else None, info)
