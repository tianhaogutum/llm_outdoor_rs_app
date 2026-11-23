"""One conversation turn, without a database: extract -> resolve place -> merge -> turn policy -> reply.

Turn policy (docs/CHAT.md section 3):

- Required slots: ``activity`` (the category hard filter) and ``place`` (the radius hard filter needs a
  centre). Everything else has a default: the questionnaire's difficulty and distance, no tags, no cap.
- After merging a message, if a required slot is missing and fewer than :data:`MAX_CLARIFY_TURNS`
  clarifying questions were asked in this conversation, ask exactly one question, about the first
  missing slot in :data:`REQUIRED_SLOTS` order.
- Otherwise the turn is ``ready``. If required slots are still missing at that point, the documented
  defaults are applied once and recorded: activity -> ``hiking`` (82% of the catalogue's routes),
  place -> none (no radius filter: the whole catalogue is eligible). A defaulted slot is never asked
  for again in the same conversation; the user can still set it at any time.
- The assistant's reply is written by code from the changes (never free text from the model).

:class:`ChatSession` holds (state, slots, user vector) and is what both the API service and the
offline harness drive.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from .extract import Extraction, extract_slots
from .merge import apply_update, diff
from .places import PlaceResolver
from .projection import category_for, project_vector, user_query_from_slots
from .schema import (
    ChatResponse,
    ChatSlots,
    ChatSlotsView,
    ChatState,
    ClarifyingQuestion,
    Dropped,
    SlotChange,
    SlotUpdate,
    Turn,
)
from .vocab import band_for_range, difficulty_for_range, distance_range

REQUIRED_SLOTS: tuple[str, ...] = ("activity", "place")
MAX_CLARIFY_TURNS = 2
DEFAULT_ACTIVITY = "hiking"
MAX_STORED_TURNS = 40

Extractor = Callable[[str, ChatSlots, Optional[str]], Awaitable[Extraction]]

QUESTIONS = {
    "en": {
        "activity": ("What would you like to do: hiking, cycling, mountain biking or running?",
                     ["hiking", "cycling", "mtb", "running"]),
        "place": ("Where should I look? Name a town or village in Bavaria to search around.", []),
    },
    "de": {
        "activity": ("Was möchtest du machen: Wandern, Radfahren, Mountainbiken oder Laufen?",
                     ["hiking", "cycling", "mtb", "running"]),
        "place": ("Wo soll ich suchen? Nenn mir einen Ort in Bayern, um den herum ich suchen soll.", []),
    },
}
_LABELS = {
    "en": {"activity": "activity", "difficulty": "difficulty", "distance_band": "length",
           "duration_hours": "duration", "distance_cap_km": "max length", "place": "place",
           "travel_radius_km": "radius", "tags_preferred": "wanted", "tags_avoided": "avoided",
           "intent": "looking for"},
    "de": {"activity": "Aktivität", "difficulty": "Schwierigkeit", "distance_band": "Länge",
           "duration_hours": "Dauer", "distance_cap_km": "max. Länge", "place": "Ort",
           "travel_radius_km": "Umkreis", "tags_preferred": "gewünscht", "tags_avoided": "vermeiden",
           "intent": "Suche"},
}
_TEXT = {
    "en": {"understood": "Got it: {items}.", "nothing": "I did not find a new preference in that.",
           "unknown_place": "I could not find “{name}” among the places of the catalogue (Bavaria).",
           "ready": "Updating your recommendations.",
           "defaults": "I will go with the defaults for now: {items}. You can change them any time.",
           "default_activity": "hiking", "default_place": "no location filter"},
    "de": {"understood": "Verstanden: {items}.", "nothing": "Darin habe ich keinen neuen Wunsch gefunden.",
           "unknown_place": "„{name}“ habe ich unter den Orten des Katalogs (Bayern) nicht gefunden.",
           "ready": "Ich passe deine Empfehlungen an.",
           "defaults": "Ich nehme vorerst die Standardwerte: {items}. Du kannst sie jederzeit ändern.",
           "default_activity": "Wandern", "default_place": "kein Ortsfilter"},
}


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, list):
        return ", ".join(value) if value else "–"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def describe_changes(changes: list[SlotChange], lang: str) -> str:
    labels = _LABELS[lang]
    parts = []
    for c in changes:
        if c.slot == "intent":
            continue
        if c.slot in ("tags_preferred", "tags_avoided"):
            added = [t for t in (c.new or []) if t not in (c.old or [])]
            removed = [t for t in (c.old or []) if t not in (c.new or [])]
            if added:
                parts.append(f"{labels[c.slot]} +{', +'.join(added)}")
            if removed:
                parts.append(f"{labels[c.slot]} −{', −'.join(removed)}")
            continue
        unit = " km" if c.slot in ("distance_cap_km", "travel_radius_km") else (" h" if c.slot == "duration_hours"
                                                                                  else "")
        new = _fmt(c.new) + (unit if c.new is not None else "")
        parts.append(f"{labels[c.slot]} {_fmt(c.old)} → {new}" if c.old is not None else f"{labels[c.slot]} {new}")
    return "; ".join(parts)


def missing_required(slots: ChatSlots, state: ChatState) -> list[str]:
    return [s for s in REQUIRED_SLOTS if getattr(slots, s) is None and s not in state.defaults_applied]


@dataclass
class TurnResult:
    status: str  # clarify | ready | idle
    reply: str
    question: Optional[ClarifyingQuestion] = None
    changes: list[SlotChange] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    extraction: Optional[Extraction] = None


class ChatSession:
    """Conversation state + slots + user vector, with the turn logic. No I/O except the two callables."""

    def __init__(
        self,
        state: ChatState,
        slots: ChatSlots,
        vector: dict[str, Any],
        *,
        resolver: PlaceResolver,
        extractor: Optional[Extractor] = None,
        questionnaire_tags: Optional[list[str]] = None,
    ) -> None:
        self.state = state
        self.slots = slots
        self.vector = vector
        self.resolver = resolver
        self.extractor = extractor or default_extractor()
        self.questionnaire_tags = questionnaire_tags if questionnaire_tags is not None else [
            t for t in (vector.get("preferred_tags") or []) if t not in slots.tags_preferred
        ]
        if self.state.base_vector is None:
            self.state.base_vector = dict(vector)

    # ------------------------------------------------------------------------------------ turns
    def _revert_cleared_bands(self, new: ChatSlots, changes: list[SlotChange]) -> tuple[ChatSlots, list[SlotChange]]:
        """A cleared difficulty or distance band falls back to the questionnaire's value (the pre-chat
        vector), so "clear" means "back to what the form said", in the response and after a reload."""
        base = self.state.base_vector or {}
        fill = {
            "difficulty": difficulty_for_range(base.get("difficulty_range")),
            "distance_band": band_for_range(base.get("min_distance_km"), base.get("max_distance_km"), new.activity),
        }
        out = []
        for c in changes:
            if c.slot in fill and c.new is None and fill[c.slot] is not None:
                new = new.model_copy(update={c.slot: fill[c.slot]})
                if c.old == fill[c.slot]:
                    continue
                c = c.model_copy(update={"new": fill[c.slot]})
            out.append(c)
        return new, out

    def _commit(self, new: ChatSlots, changes: list[SlotChange]) -> None:
        self.vector = project_vector(self.vector, self.state.base_vector, self.slots, new, changes)
        self.slots = new
        st = self.state
        st.difficulty, st.distance_band, st.duration_hours = new.difficulty, new.distance_band, new.duration_hours
        st.tags_preferred, st.tags_avoided = list(new.tags_preferred), list(new.tags_avoided)
        for c in changes:
            st.sources[c.slot] = c.source

    async def _resolve(self, update: SlotUpdate, dropped: list[Dropped]):
        if not update.place_name:
            return None
        ref = await self.resolver(update.place_name)
        if ref is None:
            dropped.append(Dropped(field="place_name", reason="no catalogue place matches", value=update.place_name))
        return ref

    async def send(self, message: str) -> TurnResult:
        """One user message -> one assistant turn. Raises the extractor's errors (state unchanged)."""
        st = self.state
        extraction = await self.extractor(message, self.slots, st.pending_slot)
        update, dropped = extraction.update, list(extraction.dropped)
        place = await self._resolve(update, dropped)
        new, changes = apply_update(self.slots, update, place=place, source="chat")
        new, changes = self._revert_cleared_bands(new, changes)
        if update.language:
            st.language = update.language
        lang = st.language

        missing = missing_required(new, st)
        question = None
        defaulted: list[str] = []
        if missing and st.clarify_turns < MAX_CLARIFY_TURNS:
            slot = missing[0]
            text, options = QUESTIONS[lang][slot]
            question = ClarifyingQuestion(slot=slot, text=text, options=options)
            st.clarify_turns += 1
            st.pending_slot = slot
            status = "clarify"
        else:
            st.pending_slot = None
            status = "ready"
            for slot in missing:
                st.defaults_applied.append(slot)
                defaulted.append(slot)
                if slot == "activity":
                    before = new
                    new = new.model_copy(update={"activity": DEFAULT_ACTIVITY})
                    changes += diff(before, new, "default")
        self._commit(new, changes)

        parts = []
        summary = describe_changes([c for c in changes if c.source != "default" or c.slot != "activity"], lang)
        if summary:
            parts.append(_TEXT[lang]["understood"].format(items=summary))
        elif not changes:
            parts.append(_TEXT[lang]["nothing"])
        for d in dropped:
            if d.field == "place_name" and d.reason == "no catalogue place matches":
                parts.append(_TEXT[lang]["unknown_place"].format(name=(d.value or "")[:60]))
        if defaulted:
            items = ", ".join(_TEXT[lang][f"default_{s}"] for s in defaulted)
            parts.append(_TEXT[lang]["defaults"].format(items=items))
        parts.append(question.text if question else _TEXT[lang]["ready"])
        reply = " ".join(parts)

        st.turns.append(Turn(role="user", text=message, at=_now()))
        st.turns.append(Turn(role="assistant", text=reply, at=_now(),
                             question_slot=question.slot if question else None))
        st.turns = st.turns[-MAX_STORED_TURNS:]
        return TurnResult(status, reply, question, changes, dropped, extraction)

    async def edit(self, update: SlotUpdate) -> TurnResult:
        """A structured edit from the UI chips: same merge rules, no LLM, no clarifying question."""
        dropped: list[Dropped] = []
        place = await self._resolve(update, dropped)
        new, changes = apply_update(self.slots, update, place=place, source="edit", default_radius_km=None)
        new, changes = self._revert_cleared_bands(new, changes)
        self._commit(new, changes)
        lang = self.state.language
        summary = describe_changes(changes, lang)
        reply = _TEXT[lang]["understood"].format(items=summary) if summary else _TEXT[lang]["nothing"]
        return TurnResult(self.status(), reply, None, changes, dropped)

    def status(self) -> str:
        if self.state.pending_slot:
            return "clarify"
        if not self.state.turns:
            return "idle"
        return "ready" if not missing_required(self.slots, self.state) else "idle"

    def reset(self) -> None:
        """Start a new conversation: history, clarifying-turn count and defaults are forgotten; slots stay."""
        self.state.turns, self.state.clarify_turns = [], 0
        self.state.pending_slot, self.state.defaults_applied = None, []

    # ---------------------------------------------------------------------------------- outputs
    def user_query(self, *, request_id: str, category: Optional[str] = None, feedback: tuple = ()):
        return user_query_from_slots(self.slots, self.vector, request_id=request_id, category=category,
                                     feedback=feedback)

    def view(self) -> ChatSlotsView:
        s = self.slots
        rng = self.vector.get("difficulty_range")
        lo, hi = self.vector.get("min_distance_km"), self.vector.get("max_distance_km")
        if lo is None and hi is None and s.distance_band:
            lo, hi = distance_range(s.distance_band, s.activity)
        return ChatSlotsView(
            **s.model_dump(),
            category=category_for(s),
            distance_range_km=[float(lo), float(hi)] if lo is not None and hi is not None else None,
            difficulty_range=[float(x) for x in rng] if isinstance(rng, list) and len(rng) == 2 else None,
            questionnaire_tags=list(self.questionnaire_tags),
            sources=dict(self.state.sources),
        )

    def response(self, profile_id: int, result: Optional[TurnResult] = None) -> ChatResponse:
        st = self.state
        question = result.question if result else None
        if question is None and st.pending_slot:
            text, options = QUESTIONS[st.language][st.pending_slot]
            question = ClarifyingQuestion(slot=st.pending_slot, text=text, options=options)
        ext = result.extraction if result else None
        return ChatResponse(
            profile_id=profile_id,
            status=result.status if result else self.status(),
            reply=result.reply if result else (st.turns[-1].text if st.turns else ""),
            question=question,
            changes=result.changes if result else [],
            dropped=result.dropped if result else [],
            slots=self.view(),
            intent=self.slots.intent,
            clarify_turns_used=st.clarify_turns,
            clarify_turns_max=MAX_CLARIFY_TURNS,
            defaults_applied=list(st.defaults_applied),
            turns=list(st.turns),
            model=ext.model if ext else None,
            cached=ext.cached if ext else None,
        )


def default_extractor(client: Any = None) -> Extractor:
    async def run(message: str, current: ChatSlots, pending: Optional[str]) -> Extraction:
        return await extract_slots(message, current=current, pending_slot=pending, client=client)

    return run
