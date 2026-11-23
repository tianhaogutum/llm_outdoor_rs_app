"""Conversational preference elicitation (BUILD_SPEC Phase 4). See docs/CHAT.md.

Entry points for the offline harness (no database):

- :func:`extract_slots` / :func:`extract_slots_sync`: run the extractor on a bare utterance;
- :func:`run_conversation`: drive a whole scripted conversation through the real turn logic;
- :func:`user_query_from_slots` (or ``ChatSession.user_query``): the ``UserQuery`` a recommender sees.

The API uses :class:`ChatSession` through ``app.services.chat_service``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from .engine import MAX_CLARIFY_TURNS, REQUIRED_SLOTS, ChatSession, TurnResult, default_extractor
from .extract import SYSTEM_PROMPT, Extraction, extract_slots, extract_slots_sync, sanitize
from .merge import DEFAULT_RADIUS_KM, apply_update
from .places import DbPlaceResolver, IndexPlaceResolver, PlaceResolver
from .projection import project_vector, user_query_from_slots
from .schema import ChatExtraction, ChatResponse, ChatSlots, ChatState, PlaceRef, SlotUpdate


async def run_conversation(
    messages: Sequence[str],
    *,
    resolver: PlaceResolver,
    start: Optional[ChatSlots] = None,
    vector: Optional[dict[str, Any]] = None,
    client: Any = None,
) -> tuple[ChatSession, list[TurnResult]]:
    """Run ``messages`` as one conversation from ``start`` slots and a user vector (default: empty)."""
    session = ChatSession(ChatState(), (start or ChatSlots()).model_copy(deep=True), dict(vector or {}),
                          resolver=resolver, extractor=default_extractor(client))
    results = [await session.send(m) for m in messages]
    return session, results


__all__ = [
    "DEFAULT_RADIUS_KM",
    "MAX_CLARIFY_TURNS",
    "REQUIRED_SLOTS",
    "SYSTEM_PROMPT",
    "ChatExtraction",
    "ChatResponse",
    "ChatSession",
    "ChatSlots",
    "ChatState",
    "DbPlaceResolver",
    "Extraction",
    "IndexPlaceResolver",
    "PlaceRef",
    "PlaceResolver",
    "SlotUpdate",
    "TurnResult",
    "apply_update",
    "extract_slots",
    "extract_slots_sync",
    "project_vector",
    "run_conversation",
    "sanitize",
    "user_query_from_slots",
]
