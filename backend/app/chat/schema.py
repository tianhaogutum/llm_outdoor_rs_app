"""Schemas of the chat: what the model fills, what code accepts, what is stored and what the API returns.

Two layers on purpose (docs/CHAT.md section 5):

- :class:`ChatExtraction` is the *wire* schema the LLM fills (structured output). Its JSON schema
  advertises every enum, so providers with constrained decoding stay inside the vocabulary, but its
  Pydantic validation is lenient about *values* (plain strings and numbers). A structural failure (not
  JSON, wrong types) is a schema violation: counted by the LLM client's ledger and re-asked.
- :class:`SlotUpdate` is the *strict* schema. :func:`app.chat.extract.sanitize` builds it field by field
  from the wire object; a value that fails (unknown enum, out-of-range number, tag outside the
  vocabulary, over-long text) is dropped on its own and reported in ``dropped``, instead of throwing
  away the whole turn. Extra keys the model invents are ignored: only these fields can ever reach the
  profile.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator

from .vocab import (
    ACTIVITIES,
    CLEARABLE,
    DIFFICULTY_BANDS,
    DIFFICULTY_CHANGES,
    DISTANCE_BANDS,
    DISTANCE_CHANGES,
    LANGUAGES,
    TAGS,
)

Activity = Literal["hiking", "cycling", "mtb", "running"]
DifficultyBand = Literal["easy", "moderate", "hard"]
DistanceBand = Literal["short", "medium", "long"]
Language = Literal["en", "de"]
SlotName = Literal[
    "activity", "difficulty", "distance_band", "duration_hours", "distance_cap_km", "place",
    "travel_radius_km", "tags_preferred", "tags_avoided", "intent",
]
RequiredSlot = Literal["activity", "place"]
ClearableSlot = Literal[
    "activity", "difficulty", "distance_band", "duration_hours", "distance_cap_km", "place",
    "travel_radius_km", "tags", "intent",
]

# Limits enforced by the sanitizer (and by SlotUpdate itself).
MAX_DURATION_H = 48.0
MIN_DURATION_H = 0.25
MAX_DISTANCE_CAP_KM = 400.0
MIN_DISTANCE_CAP_KM = 1.0
MAX_RADIUS_KM = 300.0
MIN_RADIUS_KM = 1.0
MAX_TAGS_PER_LIST = 8
MAX_INTENT_CHARS = 300
MAX_PLACE_CHARS = 80
MAX_MESSAGE_CHARS = 1000


def _enum_or_null(values: tuple[str, ...], description: str) -> Any:
    return WithJsonSchema({"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}],
                           "description": description})


def _number_or_null(description: str) -> Any:
    return WithJsonSchema({"anyOf": [{"type": "number"}, {"type": "null"}], "description": description})


def _string_or_null(description: str, max_length: int) -> Any:
    return WithJsonSchema({"anyOf": [{"type": "string", "maxLength": max_length}, {"type": "null"}],
                           "description": description})


def _string_list(description: str, max_items: int) -> Any:
    return WithJsonSchema({"type": "array", "items": {"type": "string"}, "maxItems": max_items,
                           "description": description})


def _enum_list(values: tuple[str, ...], description: str, max_items: int) -> Any:
    return WithJsonSchema({"type": "array", "items": {"type": "string", "enum": list(values)},
                           "maxItems": max_items, "description": description})


def _all_required(schema: dict[str, Any]) -> None:
    # Every field is listed as required (nullable / possibly empty) in the JSON schema the model sees:
    # with optional fields, Gemini omits most of them. Pydantic validation keeps the defaults.
    schema["required"] = list(schema.get("properties", {}))


class ChatExtraction(BaseModel):
    """Wire schema: what the model returns for one user message. Lenient values, see module docstring."""

    model_config = ConfigDict(extra="ignore", json_schema_extra=_all_required)

    language: Annotated[Optional[str], _enum_or_null(LANGUAGES, "language of the user message")] = None
    activity: Annotated[Optional[str], _enum_or_null(ACTIVITIES, "activity, only if stated or implied")] = None
    difficulty: Annotated[Optional[str], _enum_or_null(
        DIFFICULTY_BANDS, "absolute difficulty / fitness band, only if stated")] = None
    difficulty_change: Annotated[Optional[str], _enum_or_null(
        DIFFICULTY_CHANGES, "relative change of the current difficulty ('a bit easier')")] = None
    distance_band: Annotated[Optional[str], _enum_or_null(
        DISTANCE_BANDS, "absolute length band, only if stated in words ('a short walk')")] = None
    distance_change: Annotated[Optional[str], _enum_or_null(
        DISTANCE_CHANGES, "relative change of the current length ('make it shorter')")] = None
    duration_hours: Annotated[Optional[float], _number_or_null(
        "stated time for the tour itself, in hours ('a couple of hours' = 2, 'half day' = 4)")] = None
    distance_target_km: Annotated[Optional[float], _number_or_null(
        "stated target length in km that is not a maximum ('about 10 km', 'a 40 km ride')")] = None
    distance_cap_km: Annotated[Optional[float], _number_or_null(
        "hard maximum route length in km, only if a limit is stated ('max 15 k', 'höchstens 10 km')")] = None
    place_name: Annotated[Optional[str], _string_or_null(
        "town or village to search around, as named locally in German (München, not Munich)",
        MAX_PLACE_CHARS)] = None
    travel_radius_km: Annotated[Optional[float], _number_or_null(
        "how far from the place the start may be, in km ('within 30 km'; '1 hour by car' = 60)")] = None
    tags_add: Annotated[list[Any], _enum_list(TAGS, "catalogue tags the user wants", MAX_TAGS_PER_LIST)] = []
    # Only tags_add carries the tag enum: Vertex rejects the request (400 INVALID_ARGUMENT) when two
    # 50-value enums are in one schema. The other lists are validated against the vocabulary by code.
    tags_avoid: Annotated[list[Any], _string_list(
        "catalogue tags (names from the list) the user explicitly does not want", MAX_TAGS_PER_LIST)] = []
    tags_remove: Annotated[list[Any], _string_list(
        "catalogue tags (names from the list) the user takes back: neither wanted nor avoided any more",
        MAX_TAGS_PER_LIST)] = []
    clear: Annotated[list[Any], _enum_list(
        CLEARABLE, "slots the user explicitly drops ('no distance limit after all')", len(CLEARABLE))] = []
    intent: Annotated[Optional[str], _string_or_null(
        "one-sentence restatement of everything the user wants so far, as a search query", MAX_INTENT_CHARS
    )] = None


class Dropped(BaseModel):
    """A proposed value the validator rejected (shown to the user, logged, counted in the evaluation)."""

    field: str
    reason: str
    value: Optional[str] = None  # repr of the rejected value, truncated


class SlotUpdate(BaseModel):
    """Strict, validated update for one turn. Built by the sanitizer or sent directly by the UI's chips."""

    model_config = ConfigDict(extra="forbid")

    language: Optional[Language] = None
    activity: Optional[Activity] = None
    difficulty: Optional[DifficultyBand] = None
    difficulty_change: Optional[Literal["easier", "harder"]] = None
    distance_band: Optional[DistanceBand] = None
    distance_change: Optional[Literal["shorter", "longer"]] = None
    duration_hours: Optional[float] = Field(default=None, ge=MIN_DURATION_H, le=MAX_DURATION_H)
    distance_target_km: Optional[float] = Field(default=None, ge=MIN_DISTANCE_CAP_KM, le=MAX_DISTANCE_CAP_KM)
    distance_cap_km: Optional[float] = Field(default=None, ge=MIN_DISTANCE_CAP_KM, le=MAX_DISTANCE_CAP_KM)
    place_name: Optional[str] = Field(default=None, min_length=2, max_length=MAX_PLACE_CHARS)
    travel_radius_km: Optional[float] = Field(default=None, ge=MIN_RADIUS_KM, le=MAX_RADIUS_KM)
    tags_add: list[str] = Field(default_factory=list, max_length=MAX_TAGS_PER_LIST)
    tags_avoid: list[str] = Field(default_factory=list, max_length=MAX_TAGS_PER_LIST)
    tags_remove: list[str] = Field(default_factory=list, max_length=MAX_TAGS_PER_LIST)
    clear: list[ClearableSlot] = Field(default_factory=list)
    intent: Optional[str] = Field(default=None, max_length=MAX_INTENT_CHARS)

    # The same checks the sanitizer applies, so a direct edit (the UI chips) cannot bypass them.
    @field_validator("tags_add", "tags_avoid", "tags_remove")
    @classmethod
    def _vocabulary(cls, values: list[str]) -> list[str]:
        from .vocab import canonical_tag

        out = []
        for v in values:
            t = canonical_tag(v)
            if t is None:
                raise ValueError(f"not a catalogue tag: {v!r}")
            if t not in out:
                out.append(t)
        return out

    @field_validator("place_name")
    @classmethod
    def _place(cls, value: Optional[str]) -> Optional[str]:
        from .extract import clean_place_name

        cleaned, reason = clean_place_name(value)
        if reason:
            raise ValueError(reason)
        return cleaned

    @field_validator("intent")
    @classmethod
    def _intent(cls, value: Optional[str]) -> Optional[str]:
        from .extract import clean_intent

        cleaned, reason = clean_intent(value)
        if reason:
            raise ValueError(reason)
        return cleaned


class PlaceRef(BaseModel):
    """A resolved place. Coordinates always come from the place lookup, never from the model."""

    name: str
    lat: float
    lon: float
    admin_district: Optional[str] = None


class ChatSlots(BaseModel):
    """The preference slots the chat manages (the merged state after every turn)."""

    activity: Optional[Activity] = None
    difficulty: Optional[DifficultyBand] = None
    distance_band: Optional[DistanceBand] = None
    duration_hours: Optional[float] = None
    distance_cap_km: Optional[float] = None
    place: Optional[PlaceRef] = None
    travel_radius_km: Optional[float] = None
    tags_preferred: list[str] = Field(default_factory=list)
    tags_avoided: list[str] = Field(default_factory=list)
    intent: Optional[str] = None


class SlotChange(BaseModel):
    """One slot that changed in a turn, old -> new. ``source``: ``chat`` (the message), ``edit`` (a chip
    in the UI) or ``default`` (a documented default the policy filled in)."""

    slot: SlotName
    old: Any = None
    new: Any = None
    source: Literal["chat", "edit", "default"] = "chat"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    at: Optional[str] = None  # ISO timestamp (UTC)
    question_slot: Optional[RequiredSlot] = None  # assistant turns that asked a clarifying question


class ChatState(BaseModel):
    """Conversation state stored on the profile (``demo_profiles.chat_state_json``)."""

    version: Literal[1] = 1
    turns: list[Turn] = Field(default_factory=list)
    clarify_turns: int = 0
    pending_slot: Optional[RequiredSlot] = None
    defaults_applied: list[RequiredSlot] = Field(default_factory=list)
    language: Language = "en"
    # Chat-only slots (the rest of ChatSlots lives in profile columns and the user vector).
    difficulty: Optional[DifficultyBand] = None
    distance_band: Optional[DistanceBand] = None
    duration_hours: Optional[float] = None
    tags_preferred: list[str] = Field(default_factory=list)
    tags_avoided: list[str] = Field(default_factory=list)
    sources: dict[str, Literal["chat", "edit", "default", "profile"]] = Field(default_factory=dict)
    # The user vector as it was before the first chat turn (clearing a band restores its range from here).
    base_vector: Optional[dict[str, Any]] = None


class ClarifyingQuestion(BaseModel):
    slot: RequiredSlot
    text: str
    options: list[str] = Field(default_factory=list)  # quick replies for the UI


# ------------------------------------------------------------------------------------------- API
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class SlotEditRequest(SlotUpdate):
    """``PATCH /api/profiles/{id}/chat/slots``: the chips. Same strict fields as a chat turn, no LLM."""


class ChatSlotsView(ChatSlots):
    """Slots plus what the client needs to render chips."""

    category: Optional[str] = None  # UI tab that follows from the activity
    distance_range_km: Optional[list[float]] = None  # [min, max] the vector uses
    difficulty_range: Optional[list[float]] = None
    questionnaire_tags: list[str] = Field(default_factory=list)  # tags from the 3-question form (read-only)
    sources: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    profile_id: int
    status: Literal["clarify", "ready", "idle"]
    reply: str
    question: Optional[ClarifyingQuestion] = None
    changes: list[SlotChange] = Field(default_factory=list)
    dropped: list[Dropped] = Field(default_factory=list)
    slots: ChatSlotsView
    intent: Optional[str] = None
    clarify_turns_used: int = 0
    clarify_turns_max: int = 2
    defaults_applied: list[str] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    model: Optional[str] = None
    cached: Optional[bool] = None
