"""What a recommender system is allowed to see for one request."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class FeedbackEvent:
    """One piece of feedback (Phase 5): a negative reason or a completion."""

    route_id: int
    reason: str  # "too-hard" | "too-easy" | "too-far" | "not-interested" | "completed"
    age_days: float = 0.0


@dataclass(frozen=True)
class UserQuery:
    """Everything a system may see for one request. Never carries a simulated user's latent fields.

    - ``user_vector``: exactly what ``translate_questionnaire_to_vector`` returns for the 3-question form.
    - ``category``: the UI tab (``hiking`` / ``cycling`` / ``running``).
    - profile slots a product form could collect: ``home`` (lat, lon), ``max_travel_km`` (great-circle
      radius from home to the route start) and ``max_distance_km`` (absolute cap on route length).
    - ``intent``: optional free text; ``feedback``: optional history (Phase 5).
    - ``request_id``: stable id (seeds randomised systems).
    """

    request_id: str
    user_vector: dict[str, Any]
    category: str
    home: Optional[tuple[float, float]] = None
    max_travel_km: Optional[float] = None
    max_distance_km: Optional[float] = None
    intent: Optional[str] = None
    feedback: tuple[FeedbackEvent, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_vector": self.user_vector,
            "category": self.category,
            "home": list(self.home) if self.home else None,
            "max_travel_km": self.max_travel_km,
            "max_distance_km": self.max_distance_km,
            "intent": self.intent,
            "feedback": [vars(f) for f in self.feedback],
        }

    def fingerprint(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
