"""
Pydantic schemas for API request/response models.
These schemas provide type validation and automatic API documentation.
"""
import json
from datetime import datetime
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.recsys.explain import StageContributions

# Attribution the ODbL requires wherever OSM-derived routes are shown.
OSM_ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0"
OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"

# ============================================================================
# Route Schemas
# ============================================================================

class BreakpointResponse(BaseModel):
    """Breakpoint response schema."""

    id: int
    route_id: int
    order_index: int
    poi_name: Optional[str] = None
    poi_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    main_quest_snippet: Optional[str] = None
    at_m: Optional[int] = None  # position along the route (derived OSM breakpoints)
    mini_quests: list["MiniQuestResponse"] = []

    model_config = ConfigDict(from_attributes=True)


class RecommendationScoreBreakdown(BaseModel):
    """Breakdown of recommendation score components."""

    difficulty: dict
    distance: dict
    tags: dict
    total: float

    # Keep the feedback keys the ranker adds (base_score, final_score, feedback_penalty, feedback_count);
    # the frontend types already expect them.
    model_config = ConfigDict(extra="allow")


def _parse_json_object(value: Any) -> Optional[dict]:
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class RouteCatalogFields(BaseModel):
    """
    Facts from the OSM catalogue (all None for legacy routes).

    ``ascent_m`` equals the legacy ``elevation`` column (total ascent); ``facts`` is the parsed
    ``facts_json`` (highlights, access, surface, elevation range, ...); ``attribution`` is set for
    OSM routes and must be shown with them.
    """

    source: Optional[str] = None
    source_url: Optional[str] = None
    activity: Optional[str] = None
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    is_loop: Optional[bool] = None
    ascent_m: Optional[int] = None
    descent_m: Optional[int] = None
    locality: Optional[str] = None
    admin_district: Optional[str] = None
    admin_region: Optional[str] = None
    facts: Optional[dict] = Field(default=None, validation_alias=AliasChoices("facts", "facts_json"))
    attribution: Optional[str] = None

    @field_validator("facts", mode="before")
    @classmethod
    def _facts_from_json(cls, value: Any) -> Optional[dict]:
        return _parse_json_object(value)

    @model_validator(mode="after")
    def _osm_attribution(self):
        if self.source == "osm" and not self.attribution:
            self.attribution = OSM_ATTRIBUTION
        return self


CATALOG_FIELD_NAMES = ("source", "source_url", "activity", "start_lat", "start_lon", "is_loop", "descent_m",
                       "locality", "admin_district", "admin_region", "facts_json")


def route_catalog_fields(route: Any) -> dict:
    """Catalogue fields of an ORM route as a dict for hand-built route payloads (no lazy loads)."""
    data = {name: getattr(route, name, None) for name in CATALOG_FIELD_NAMES}
    data["ascent_m"] = getattr(route, "elevation", None)
    return RouteCatalogFields.model_validate(data).model_dump()


class FeedbackEventView(BaseModel):
    reason: str  # too-hard | too-easy | too-far | not-interested | completed
    age_days: float


class FeedbackEffect(BaseModel):
    """What the user's feedback did to this route. Until the pipeline's feedback stage exists, the only
    effects are the rule scorer's penalty (no-intent path) and the interim serving rule that hides routes
    rejected or completed in the last ``window_days`` (those never appear, so they are counted here)."""

    events_considered: int = 0  # feedback events passed to the pipeline for this request
    on_this_route: list[FeedbackEventView] = []  # events about this very route (older than the window)
    rule_penalty: Optional[float] = None  # v0 rule scorer multiplier when it ranked this route
    hidden_routes: int = 0  # routes hidden by the interim rule in this list
    window_days: Optional[float] = None
    stage: Optional[dict] = None  # the pipeline's own feedback-stage contribution, when it has one


class ExplorationFlag(BaseModel):
    is_exploration: bool = False
    detail: Optional[dict] = None  # the exploration stage's contribution, when the pipeline has one


class RouteExplanationResponse(BaseModel):
    """Per-route explanation on pipeline paths (stages: ``app.recsys.explain.StageContributions``)."""

    schema_version: str = "recsys.explanation/v1"
    rank: int
    score: float  # final pipeline score (MMR score, else ranker score); not a percentage
    stages: StageContributions
    feedback: FeedbackEffect = FeedbackEffect()
    exploration: ExplorationFlag = ExplorationFlag()
    n_candidates: Optional[int] = None  # routes the ranker ordered (for "ranked 3rd of 50")
    n_eligible: Optional[int] = None  # routes that passed the hard filter
    other_stages: Optional[dict] = None  # trace stages this API version does not know yet (kept, untyped)


class RouteResponse(RouteCatalogFields):
    """Single route response schema with full details."""

    id: int
    title: str
    category_name: Optional[str] = None
    length_meters: Optional[float] = None
    duration_min: Optional[int] = None
    difficulty: Optional[int] = None
    short_description: Optional[str] = None
    location: Optional[str] = None
    elevation: Optional[int] = None
    tags_json: Optional[str] = None
    xp_required: int
    base_xp_reward: int
    story_prologue_title: Optional[str] = None
    story_prologue_body: Optional[str] = None
    story_epilogue_body: Optional[str] = None
    gpx_data_raw: Optional[str] = None  # GPX track data for map visualization
    breakpoints: list[BreakpointResponse] = []
    is_locked: bool = False  # Computed field based on user XP
    recommendation_score: Optional[float] = None  # CBF score (0.0-1.0); baseline path only
    recommendation_score_breakdown: Optional[RecommendationScoreBreakdown] = None  # Score components (baseline)
    # Pipeline paths: per-stage contributions from the trace, and a one-line reason built only from those
    # facts and the catalogue record (docs/SERVING.md "Explanations"). The baseline path has a reason too.
    explanation: Optional[RouteExplanationResponse] = None
    reason: Optional[str] = None
    is_exploration: bool = False

    model_config = ConfigDict(from_attributes=True)


class RouteSummaryResponse(RouteCatalogFields):
    """Light route payload for lists: no breakpoints, story texts or GPX."""

    id: int
    title: str
    category_name: Optional[str] = None
    length_meters: Optional[float] = None
    duration_min: Optional[int] = None
    difficulty: Optional[int] = None
    short_description: Optional[str] = None
    location: Optional[str] = None
    elevation: Optional[int] = None
    tags_json: Optional[str] = None
    xp_required: int
    base_xp_reward: int

    model_config = ConfigDict(from_attributes=True)


class RouteListResponse(BaseModel):
    """Paged route list."""

    routes: list[RouteSummaryResponse]
    total: int  # all routes matching the filter, not just this page
    limit: int
    offset: int


class RouteDetailResponse(RouteResponse):
    """Route detail response with all nested data."""
    pass


class RouteGeometryResponse(BaseModel):
    """Simplified route line for the map (``[[lon, lat], ...]``, WGS84) and elevation profile."""

    route_id: int
    coords: list[list[float]]
    bbox: list[float]  # [min_lon, min_lat, max_lon, max_lat]
    ele_profile_m: Optional[list[float]] = None
    ele_step_m: Optional[float] = None
    length_m: Optional[float] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    attribution: Optional[str] = None


class PlaceResponse(BaseModel):
    """A catalogue locality that a typed name resolved to."""

    id: int
    name: str
    admin_district: Optional[str] = None
    admin_region: Optional[str] = None
    lat: float
    lon: float
    accuracy_m: Optional[int] = None  # distance of the coordinates from the locality's OSM place node
    route_count: int  # routes starting near this locality
    match: str  # exact | prefix | word | fuzzy
    score: float


class PlaceSearchResponse(BaseModel):
    places: list[PlaceResponse]
    total: int


# ============================================================================
# Profile Schemas
# ============================================================================

class HomeLocation(BaseModel):
    """
    The user's home. Give ``lat``/``lon`` (e.g. from ``GET /api/places``) or only ``name``, which the
    API then resolves against the catalogue's localities (422 if nothing matches).
    """

    name: str = Field(min_length=1, max_length=255)
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def _both_or_neither(self):
        if (self.lat is None) != (self.lon is None):
            raise ValueError("give both lat and lon, or neither")
        return self


class ProfileSlots(BaseModel):
    """
    Optional preference slots for hard filters and retrieval. Absent means "no constraint"; the
    v0_rules ranking does not read them.
    """

    home: Optional[HomeLocation] = None
    travel_radius_km: Optional[float] = Field(default=None, gt=0, le=1000)  # max distance home -> route start
    distance_cap_km: Optional[float] = Field(default=None, gt=0, le=1000)  # absolute cap on route length
    intent: Optional[str] = Field(default=None, max_length=1000)  # "what are you looking for?"

    @field_validator("intent", mode="before")
    @classmethod
    def _blank_intent_is_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = " ".join(value.split())
            return value or None
        return value


class ProfileCreate(ProfileSlots):
    """Create profile request schema (from questionnaire)."""

    fitness: str  # "beginner" | "intermediate" | "advanced"
    type: list[str]  # ["history-culture", "natural-scenery", "family-fun"]
    narrative: str  # "adventure" | "mystery" | "playful"


class ProfileUpdate(ProfileSlots):
    """
    Update profile request schema (partial fields). Only fields present in the request change; an
    explicit null clears a slot (``"home": null`` clears name and coordinates).
    """

    total_xp: Optional[int] = None
    level: Optional[int] = None
    user_vector_json: Optional[str] = None
    genai_welcome_summary: Optional[str] = None
    unlocked_routes_json: Optional[str] = None


class HomeLocationResponse(BaseModel):
    name: str
    lat: float
    lon: float


class ProfileResponse(BaseModel):
    """Profile response schema."""

    id: int
    total_xp: int
    level: int
    user_vector_json: Optional[str] = None
    genai_welcome_summary: Optional[str] = None
    unlocked_routes_json: Optional[str] = None
    home: Optional[HomeLocationResponse] = None
    travel_radius_km: Optional[float] = None
    distance_cap_km: Optional[float] = None
    intent: Optional[str] = None
    activity: Optional[str] = None  # set by the chat (hiking | cycling | mtb | running)

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def _slots_from_orm(cls, data: Any) -> Any:
        if isinstance(data, dict) or not hasattr(data, "home_name"):
            return data
        out = {name: getattr(data, name) for name in (
            "id", "total_xp", "level", "user_vector_json", "genai_welcome_summary", "unlocked_routes_json",
            "travel_radius_km", "distance_cap_km",
        )}
        out["activity"] = getattr(data, "activity", None)
        out["intent"] = data.intent_text
        if data.home_name is not None and data.home_lat is not None and data.home_lon is not None:
            out["home"] = {"name": data.home_name, "lat": data.home_lat, "lon": data.home_lon}
        return out


class ProfileCreateResponse(BaseModel):
    """Profile creation response schema (after questionnaire submission)."""

    id: int
    welcome_summary: str
    user_vector: dict  # For debugging, shows translated user preferences
    profile: Optional[ProfileResponse] = None  # stored profile incl. the optional slots
    # Where welcome_summary came from: "llm" (live), "pregenerated" (LLM-generated offline,
    # backend/data/narratives) or "template" (docs/DEPLOY.md)
    welcome_summary_source: Optional[str] = None


class WelcomeSummaryRequest(BaseModel):
    """Request to generate welcome summary."""

    profile_id: int


class WelcomeSummaryResponse(BaseModel):
    """Welcome summary response."""

    summary: str


# ============================================================================
# MiniQuest Schemas
# ============================================================================

class MiniQuestResponse(BaseModel):
    """MiniQuest response schema."""

    id: int
    breakpoint_id: int
    task_description: str
    xp_reward: int

    model_config = ConfigDict(from_attributes=True)


class QuestCompleteRequest(BaseModel):
    """Complete quest request schema."""

    quest_id: int


class QuestCompleteResponse(BaseModel):
    """Complete quest response schema."""

    xp_gained: int
    new_total_xp: int
    new_level: int


# ============================================================================
# Souvenir Schemas
# ============================================================================

class SouvenirResponse(BaseModel):
    """Souvenir response schema."""

    id: int
    demo_profile_id: int
    route_id: int
    completed_at: datetime
    total_xp_gained: int
    genai_summary: Optional[str] = None
    xp_breakdown_json: Optional[str] = None
    pixel_image_svg: Optional[str] = None  # LLM-generated pixel art SVG
    route: Optional[RouteResponse] = None  # Nested route info

    model_config = ConfigDict(from_attributes=True)


class SouvenirListResponse(BaseModel):
    """Souvenir list response schema."""

    souvenirs: list[SouvenirResponse]
    total: int


# ============================================================================
# Feedback Schemas
# ============================================================================

class FeedbackCreate(BaseModel):
    """Create feedback request schema."""

    route_id: int
    reason: str  # too_difficult, too_far, not_interested, too_easy, etc.


class FeedbackResponse(BaseModel):
    """Feedback response schema."""

    id: int
    demo_profile_id: int
    route_id: int
    reason: str

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Recommendation Schemas
# ============================================================================

class RecommendationRequest(BaseModel):
    """Recommendation request schema (query parameters)."""

    profile_id: Optional[int] = None  # User profile ID for personalized recommendations
    category: Optional[str] = None  # Activity category: running/hiking/cycling or None for all


class ServedQuery(BaseModel):
    """What the pipeline saw for this request (the profile state turned into a ``UserQuery``)."""

    category: Optional[str] = None  # the tab applied ("" / None = every category)
    home_name: Optional[str] = None
    home: Optional[list[float]] = None  # [lat, lon]
    travel_radius_km: Optional[float] = None
    distance_cap_km: Optional[float] = None
    intent: Optional[str] = None
    vector_source: Optional[str] = None  # questionnaire | questionnaire+chat | reconstructed | user_vector | none
    feedback_events: int = 0  # negative feedback rows + completions passed to the pipeline
    suppressed_route_ids: list[int] = []  # hidden by the interim serving rule (docs/SERVING.md)
    feedback_handled_by: Optional[str] = None  # pipeline (its feedback stage) | serving_interim_rule


class ServedBy(BaseModel):
    """Which system and path produced the list, and why (docs/SERVING.md "Paths and fallbacks")."""

    configured_system: str  # RECSYS_SYSTEM
    system: str  # what actually ranked: recsys_serve | v2_bm25 | v0_rules | random | ...
    path: str  # pipeline | no_llm | baseline | random
    reason: Optional[str] = None  # why this path (no_intent, embedding_unavailable, timeout, budget, ...)
    degraded: bool = False  # True if a fallback replaced the configured system
    request_id: Optional[str] = None
    explanation_schema: Optional[str] = None  # recsys.explanation/v1 on pipeline paths
    query: Optional[ServedQuery] = None
    stages: list[dict] = []  # StageSummary per stage (n_in, n_out, local_ms, upstream ms / USD, info)
    latency_ms: Optional[float] = None
    upstream_cost_usd: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Recommendation response schema."""

    routes: list[RouteResponse]
    total: int
    is_personalized: bool = False  # True if based on user profile, False if random
    served_by: Optional[ServedBy] = None


# ============================================================================
# Route Completion Schemas
# ============================================================================

class RouteCompleteRequest(BaseModel):
    """Complete route request schema."""

    route_id: int  # Route ID to complete
    completed_quest_ids: list[int] = []  # List of completed quest IDs


class RouteCompleteResponse(BaseModel):
    """Complete route response schema."""

    souvenir: SouvenirResponse
    xp_breakdown: dict  # {"base": 100, "distance": 30, "quests": 20, "total": 150}
    total_xp_gained: int  # XP earned from this route completion
    new_total_xp: int  # New total XP after this completion
    new_level: int
    newly_unlocked_achievements: Optional[list[dict]] = []  # Newly unlocked achievements
    # Where souvenir.genai_summary came from: "llm" (live), "pregenerated" (LLM-generated offline,
    # backend/data/narratives) or "template" (docs/DEPLOY.md)
    summary_source: Optional[str] = None


# ============================================================================
# XP Schemas
# ============================================================================

class XPBreakdownResponse(BaseModel):
    """XP breakdown response schema."""

    base_xp: int
    distance_xp: int
    difficulty_xp: int
    quest_xp: int
    total: int
    xp_to_next_level: int


# ============================================================================
# Route Simulation Schemas
# ============================================================================

class RouteStartRequest(BaseModel):
    """Start route simulation request."""

    route_id: int


class RouteStartResponse(BaseModel):
    """Start route simulation response."""

    route: RouteResponse
    current_breakpoint_index: int = 0


class RouteProgressUpdate(BaseModel):
    """Update route progress request."""

    current_breakpoint_index: int


class RouteProgressResponse(BaseModel):
    """Route progress response."""

    current_breakpoint: BreakpointResponse
    current_breakpoint_index: int
    total_breakpoints: int
    progress_percentage: float


# ============================================================================
# Story Generation Schemas
# ============================================================================

class StoryGenerateRequest(BaseModel):
    """Story generation request schema."""
    narrative_style: str = "adventure"  # adventure, mystery, playful
    force_regenerate: bool = False  # Force regenerate even if story exists


class StoryMiniQuest(BaseModel):
    """Mini quest data for story generation (without database IDs)."""
    task_description: str
    xp_reward: int


class StoryBreakpointContent(BaseModel):
    """Story content for a single breakpoint."""
    index: int
    main_quest: str
    mini_quests: list[StoryMiniQuest] = []


class StoryGenerateResponse(BaseModel):
    """Story generation response schema."""
    title: str
    outline: str
    prologue: str
    epilogue: str
    breakpoints: list[StoryBreakpointContent]


# ============================================================================
# Statistics Schemas
# ============================================================================

class ProfileStatisticsResponse(BaseModel):
    """Profile statistics response schema."""

    total_distance_km: float  # Sum of all completed route distances
    total_elevation_m: int  # Sum of all completed route elevations
    routes_completed: int  # Total number of completed routes (souvenirs)
    achievements_unlocked: int  # Number of unlocked achievements
    activity_breakdown: dict[str, int]  # {"running": 2, "hiking": 5, "cycling": 1}

    model_config = ConfigDict(from_attributes=True)


# Update forward references
BreakpointResponse.model_rebuild()

