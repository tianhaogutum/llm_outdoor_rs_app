"""From chat slots to what recommenders read: the user vector and a :class:`app.recsys.query.UserQuery`.

The user vector (``translate_questionnaire_to_vector``) is what ``v0_rules`` scores against; the profile
slots (home, radius, cap, intent) and the category are what hard filters and retrieval read. The chat
updates the vector *incrementally*: only the keys behind slots that changed in a turn are rewritten, so
adjustments made elsewhere (feedback) survive untouched keys.

- difficulty band -> ``difficulty_range`` (questionnaire ranges); cleared -> the pre-chat range;
- distance band (+ activity) -> ``min_distance_km`` / ``max_distance_km``; a distance cap clips the
  maximum; band cleared -> the pre-chat range (still clipped by the cap);
- wanted tags are appended to ``preferred_tags`` (the catalogue spelling; v0_rules lowercases both
  sides), tags the chat removed or the user avoids are taken out; avoided tags are written to
  ``avoided_tags`` (a new key; v0_rules ignores it, later stages may penalise them).
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from .schema import ChatSlots, SlotChange
from .vocab import ACTIVITY_TAB, DIFFICULTY_RANGES, distance_range

FULL_DIFFICULTY = [0, 3]


def _fold(tags: list[str]) -> set[str]:
    return {t.casefold() for t in tags}


def project_vector(
    vector: dict[str, Any],
    base: Optional[dict[str, Any]],
    old: ChatSlots,
    new: ChatSlots,
    changes: list[SlotChange],
) -> dict[str, Any]:
    """New user vector after a turn (``vector`` is not modified). ``base`` is the pre-chat vector."""
    v = copy.deepcopy(vector)
    base = base or {}
    changed = {c.slot for c in changes}

    if "difficulty" in changed:
        if new.difficulty:
            v["difficulty_range"] = list(DIFFICULTY_RANGES[new.difficulty])
        else:
            v["difficulty_range"] = list(base.get("difficulty_range") or FULL_DIFFICULTY)

    if changed & {"distance_band", "activity", "distance_cap_km", "duration_hours"}:
        if new.distance_band:
            lo, hi = distance_range(new.distance_band, new.activity)
        elif "min_distance_km" in base or "max_distance_km" in base:
            lo, hi = float(base.get("min_distance_km", 0.0)), float(base.get("max_distance_km", 100.0))
        else:
            lo, hi = float(v.get("min_distance_km", 0.0)), float(v.get("max_distance_km", 100.0))
        if new.distance_cap_km is not None:
            hi = min(hi, float(new.distance_cap_km))
            lo = min(lo, round(hi / 2, 1))
        v["min_distance_km"], v["max_distance_km"] = lo, hi

    if changed & {"tags_preferred", "tags_avoided"}:
        dropped = (_fold(old.tags_preferred) - _fold(new.tags_preferred)) | _fold(new.tags_avoided)
        pref = [t for t in (v.get("preferred_tags") or []) if isinstance(t, str) and t.casefold() not in dropped]
        have = _fold(pref)
        pref += [t for t in new.tags_preferred if t.casefold() not in have]
        v["preferred_tags"] = pref
        v["avoided_tags"] = list(new.tags_avoided)
    return v


def category_for(slots: ChatSlots) -> Optional[str]:
    return ACTIVITY_TAB.get(slots.activity) if slots.activity else None


def user_query_from_slots(
    slots: ChatSlots,
    user_vector: dict[str, Any],
    *,
    request_id: str,
    category: Optional[str] = None,
    feedback: tuple = (),
):
    """The :class:`UserQuery` a recommender sees for a profile after chat. ``category`` (a UI tab) wins
    over the chat's activity; the radius applies only with a place (as in the hard filter)."""
    from app.recsys.query import UserQuery

    home = (slots.place.lat, slots.place.lon) if slots.place else None
    return UserQuery(
        request_id=request_id,
        user_vector=user_vector,
        category=category or category_for(slots) or "hiking",
        home=home,
        max_travel_km=slots.travel_radius_km if home else None,
        max_distance_km=slots.distance_cap_km,
        intent=slots.intent,
        feedback=tuple(feedback),
    )
