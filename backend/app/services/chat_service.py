"""Chat on a stored profile: load a :class:`ChatSession` from a ``DemoProfile``, write it back.

Where each slot lives on the profile (so every recommender behind the recommendations endpoint sees it):

- activity -> ``demo_profiles.activity``;
- place, travel radius, distance cap, intent -> ``home_name/home_lat/home_lon``, ``travel_radius_km``,
  ``distance_cap_km``, ``intent_text``;
- difficulty, distance band, tags -> the user vector (``user_vector_json``: ``difficulty_range``,
  ``min/max_distance_km``, ``preferred_tags``, ``avoided_tags``) and, by name, the chat state;
- the conversation -> ``chat_state_json`` (turns, clarifying-turn count, pending question, defaults,
  pre-chat vector).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.chat.engine import ChatSession
from app.chat.places import DbPlaceResolver
from app.chat.schema import ChatSlots, ChatState, PlaceRef
from app.chat.vocab import band_for_range, difficulty_for_range
from app.models.entities import DemoProfile


def _vector(profile: DemoProfile) -> dict[str, Any]:
    if not profile.user_vector_json:
        return {}
    try:
        v = json.loads(profile.user_vector_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return v if isinstance(v, dict) else {}


def _state(profile: DemoProfile) -> ChatState:
    if profile.chat_state_json:
        try:
            return ChatState.model_validate_json(profile.chat_state_json)
        except ValueError:
            pass  # unreadable state: start a new conversation rather than fail the request
    return ChatState()


def slots_from_profile(profile: DemoProfile, state: ChatState, vector: dict[str, Any]) -> ChatSlots:
    place = None
    if profile.home_name and profile.home_lat is not None and profile.home_lon is not None:
        place = PlaceRef(name=profile.home_name, lat=profile.home_lat, lon=profile.home_lon)
    activity = profile.activity if profile.activity in ("hiking", "cycling", "mtb", "running") else None
    return ChatSlots(
        activity=activity,
        difficulty=state.difficulty or difficulty_for_range(vector.get("difficulty_range")),
        distance_band=state.distance_band
        or band_for_range(vector.get("min_distance_km"), vector.get("max_distance_km"), activity),
        duration_hours=state.duration_hours,
        distance_cap_km=profile.distance_cap_km,
        place=place,
        travel_radius_km=profile.travel_radius_km,
        tags_preferred=list(state.tags_preferred),
        tags_avoided=list(state.tags_avoided),
        intent=profile.intent_text,
    )


def load_session(profile: DemoProfile, db: Any, *, extractor=None) -> ChatSession:
    state = _state(profile)
    vector = _vector(profile)
    slots = slots_from_profile(profile, state, vector)
    base = state.base_vector or vector
    questionnaire_tags = [t for t in (base.get("preferred_tags") or []) if isinstance(t, str)]
    return ChatSession(state, slots, vector, resolver=DbPlaceResolver(db), extractor=extractor,
                       questionnaire_tags=questionnaire_tags)


def store_session(profile: DemoProfile, session: ChatSession) -> None:
    """Write slots, vector and state back to the profile (caller commits)."""
    s = session.slots
    profile.activity = s.activity
    if s.place is not None:
        profile.home_name, profile.home_lat, profile.home_lon = s.place.name, s.place.lat, s.place.lon
    else:
        profile.home_name = profile.home_lat = profile.home_lon = None
    profile.travel_radius_km = s.travel_radius_km
    profile.distance_cap_km = s.distance_cap_km
    profile.intent_text = s.intent
    if session.vector or profile.user_vector_json:
        profile.user_vector_json = json.dumps(session.vector, ensure_ascii=False)
    profile.chat_state_json = session.state.model_dump_json()


def user_query_for_profile(profile: DemoProfile, *, category: Optional[str] = None, request_id: Optional[str] = None):
    """The ``UserQuery`` for a stored profile after chat (questionnaire vector + slots + intent)."""
    state = _state(profile)
    vector = _vector(profile)
    slots = slots_from_profile(profile, state, vector)
    from app.chat.projection import user_query_from_slots

    return user_query_from_slots(slots, vector, request_id=request_id or f"profile-{profile.id}", category=category)


def rules_extractor():
    """The chat's no-LLM extractor: the keyword rules (``app.chat.rules``) through the same validator.

    Used when the daily budget is spent (docs/SERVING.md, "Demo protections"); the turn policy, merge
    rules and replies are unchanged, only the extraction is the offline baseline's (lower recall)."""
    from app.chat.extract import Extraction, sanitize
    from app.chat.rules import rule_extract
    from app.chat.schema import ChatExtraction

    async def run(message: str, current: ChatSlots, pending: Optional[str]) -> Extraction:
        wire = ChatExtraction.model_validate(rule_extract(message))
        update, dropped = sanitize(wire)
        return Extraction(update=update, dropped=dropped, raw=wire.model_dump(), model="keyword-rules",
                          provider="none")

    return run
