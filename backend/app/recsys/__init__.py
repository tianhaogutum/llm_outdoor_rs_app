"""Shared recommender core, used by the API and by the offline evaluation harness (``evalkit``).

Layout (see docs/EVAL.md for how the harness uses it):

- :mod:`.catalog`      in-memory route catalogue (light records + numpy arrays), loaded once
- :mod:`.query`        :class:`UserQuery`: everything a system may see for one request
- :mod:`.result`       :class:`Recommendation` with per-stage :class:`StageTrace`
- :mod:`.constraints`  deterministic hard-constraint predicates (category tab, radius, distance cap)
- :mod:`.systems`      registry ``name -> factory``; ``random`` and ``v0_rules`` are registered

Design rule: similarity proposes, verifiable constraints decide eligibility, and every stage reports
its contribution in the trace.
"""

from .catalog import Catalog, RouteRecord, get_catalog
from .constraints import CATEGORY_TABS, HardConstraints, haversine_km
from .query import FeedbackEvent, UserQuery
from .result import Recommendation, StageTrace
from .systems import System, build_system, get_system_spec, list_systems, register_system

__all__ = [
    "CATEGORY_TABS",
    "Catalog",
    "FeedbackEvent",
    "HardConstraints",
    "Recommendation",
    "RouteRecord",
    "StageTrace",
    "System",
    "UserQuery",
    "build_system",
    "get_catalog",
    "get_system_spec",
    "haversine_km",
    "list_systems",
    "register_system",
]
