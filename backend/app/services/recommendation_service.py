"""
Recommendation service implementing Content-Based Filtering (CBF).

This service provides route recommendations by matching user preferences
with route attributes using similarity scoring.

Enhanced with feedback-aware recommendations that learn from user feedback.

This scorer is the ``v0_rules`` baseline. The ranking itself lives in the pure function
:func:`rank_routes_v0`, which works on plain route records and plain feedback records and needs no
database; :func:`get_recommended_routes` only loads data and delegates to it.
"""
import json
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.logger import get_logger, log_business_logic
from app.models.entities import Breakpoint, DemoProfile, ProfileFeedback, Route

logger = get_logger(__name__)


# Difficulty scale. Route difficulty in the catalogue is an integer from DIFFICULTY_MIN to DIFFICULTY_MAX
# (legacy Outdooractive data: 20 / 35 / 46 / 10 routes at 0 / 1 / 2 / 3). The frontend labels the levels
# 0 = easy, 1 = moderate, 2 = difficult, 3 = expert (frontend/lib/api-types.ts, mapApiDifficulty).
# User preference ranges and feedback-adjusted ranges are floats on the same scale.
DIFFICULTY_MIN = 0
DIFFICULTY_MAX = 3
# Feedback shifts keep the preferred difficulty range at least this wide at either end of the scale.
DIFFICULTY_MIN_RANGE_WIDTH = 1.0
# Step (in difficulty levels, before time decay) applied per "too-hard" / "too-easy" feedback.
DIFFICULTY_FEEDBACK_STEP = 0.5

# Distance defaults used when a user vector carries no distance preference.
DEFAULT_MIN_DISTANCE_KM = 0.0
DEFAULT_MAX_DISTANCE_KM = 100.0
# "too-far" feedback shrinks max distance by this fraction (before time decay)...
DISTANCE_FEEDBACK_SHRINK = 0.1
# ...but never below min_distance_km + this gap.
MIN_DISTANCE_GAP_KM = 2.0


def default_difficulty_range() -> list[int]:
    """Full difficulty scale, used when a user vector has no difficulty preference (fresh list)."""
    return [DIFFICULTY_MIN, DIFFICULTY_MAX]


# Category mapping: custom category → route.category_name values
CATEGORY_MAPPING = {
    "running": ["Jogging", "Trail running"],
    "hiking": ["Theme trail", "Hiking trail"],
    "cycling": ["Cycling", "Mountainbiking", "Long distance cycling"],
}

# Weights for CBF scoring components
SCORE_WEIGHTS = {
    "difficulty": 0.4,
    "distance": 0.3,
    "tags": 0.3,
}

# Feedback-aware recommendation parameters
# Penalty multipliers based on feedback count:
# - 1 feedback: 50% (0.5)
# - 2 feedbacks: 10% (0.1)
# - 3+ feedbacks: 1% (0.01)
FEEDBACK_PENALTY_MULTIPLIERS = {
    1: 0.5,   # 50%
    2: 0.1,   # 10%
    3: 0.01,  # 1%
}
FEEDBACK_FILTER_THRESHOLD = 4  # Filter routes with 4+ feedback entries (after 3rd feedback shows at 1%)
TIME_DECAY_HALF_LIFE_DAYS = 30.0  # 30 days half-life for feedback weight


def _field(record: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object (ORM instance, dataclass, ...)."""
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _route_vector_from_fields(difficulty: Optional[int], length_meters: Optional[float], tags_json: Any) -> dict:
    """Shared body of :func:`extract_route_vector` and :func:`route_vector_from_record`."""
    # Parse tags from JSON
    tags = []
    if tags_json:
        try:
            tags = json.loads(tags_json)
            if isinstance(tags, list):
                # Flatten if nested or extract tag names
                flat_tags = []
                for tag in tags:
                    if isinstance(tag, str):
                        flat_tags.append(tag.lower())
                    elif isinstance(tag, dict) and "name" in tag:
                        flat_tags.append(tag["name"].lower())
                tags = flat_tags
        except (json.JSONDecodeError, TypeError):
            tags = []

    return {
        "difficulty": difficulty if difficulty is not None else DIFFICULTY_MIN,
        "length_km": (length_meters / 1000.0) if length_meters else 0.0,
        "tags": tags,
    }


def extract_route_vector(route: Route) -> dict:
    """
    Extract route features into a comparable vector.

    Parameters
    ----------
    route : Route
        Route entity from database (any object with ``difficulty``, ``length_meters`` and ``tags_json``)

    Returns
    -------
    dict
        Route vector with difficulty (DIFFICULTY_MIN..DIFFICULTY_MAX; missing -> DIFFICULTY_MIN),
        length_km and tags
    """
    return _route_vector_from_fields(route.difficulty, route.length_meters, route.tags_json)


def route_vector_from_record(record: Mapping[str, Any]) -> dict:
    """
    Route vector from a plain record shaped like a ``routes`` row (e.g. a line of ``routes.jsonl``).

    Uses ``difficulty``, ``length_meters`` and ``tags_json`` (a JSON string). If the record has no
    ``tags_json`` but a ``tags`` list, that list is used instead (same normalisation).
    Gives the same result as :func:`extract_route_vector` on the equivalent ORM row.
    """
    tags_json = record.get("tags_json")
    if isinstance(tags_json, list):
        tags_json = json.dumps(tags_json)
    elif tags_json is None and isinstance(record.get("tags"), list):
        tags_json = json.dumps(record["tags"])
    return _route_vector_from_fields(record.get("difficulty"), record.get("length_meters"), tags_json)


def calculate_difficulty_score(user_difficulty_range: list[int], route_difficulty: int) -> float:
    """
    Calculate difficulty match score.

    Returns 1.0 if route difficulty is within user's acceptable range, else 0.0.

    Parameters
    ----------
    user_difficulty_range : list[int]
        [min_difficulty, max_difficulty] from user profile, on the 0-3 scale (may be fractional
        after feedback adjustment)
    route_difficulty : int
        Route difficulty level (DIFFICULTY_MIN..DIFFICULTY_MAX, i.e. 0-3)

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    if not user_difficulty_range or len(user_difficulty_range) < 2:
        return 0.5  # Neutral score if no preference

    min_diff, max_diff = user_difficulty_range[0], user_difficulty_range[1]

    if min_diff <= route_difficulty <= max_diff:
        return 1.0

    # Calculate distance from range with decay
    if route_difficulty < min_diff:
        distance = min_diff - route_difficulty
    else:
        distance = route_difficulty - max_diff

    # Exponential decay: 0.5^distance
    return max(0.0, 0.5 ** distance)


def calculate_distance_score(
    user_min_km: float,
    user_max_km: float,
    route_length_km: float
) -> float:
    """
    Calculate distance match score.

    Returns 1.0 if route length is within user's preferred range,
    with exponential decay for distances outside the range.

    Parameters
    ----------
    user_min_km : float
        Minimum preferred distance
    user_max_km : float
        Maximum preferred distance
    route_length_km : float
        Route length in kilometers

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    if route_length_km == 0.0:
        return 0.3  # Low score for missing data

    if user_min_km <= route_length_km <= user_max_km:
        return 1.0

    # Calculate distance from acceptable range
    if route_length_km < user_min_km:
        distance_ratio = (user_min_km - route_length_km) / user_max_km
    else:
        distance_ratio = (route_length_km - user_max_km) / user_max_km

    # Exponential decay based on how far outside the range
    return max(0.0, 0.7 ** distance_ratio)


def calculate_tag_score(user_tags: list[str], route_tags: list[str]) -> float:
    """
    Calculate tag overlap score using Jaccard similarity.

    Jaccard similarity = |intersection| / |union|

    Parameters
    ----------
    user_tags : list[str]
        User's preferred tags
    route_tags : list[str]
        Route's tags

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    if not user_tags and not route_tags:
        return 0.5  # Neutral if both empty

    if not user_tags or not route_tags:
        return 0.2  # Low score if one is empty

    # Convert to sets for set operations
    user_set = set(tag.lower() for tag in user_tags)
    route_set = set(tag.lower() for tag in route_tags)

    intersection = user_set & route_set
    union = user_set | route_set

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def calculate_cbf_score(user_vector: dict, route_vector: dict) -> tuple[float, dict]:
    """
    Calculate overall CBF similarity score between user and route.

    Parameters
    ----------
    user_vector : dict
        User preference vector with difficulty_range, min/max_distance_km, preferred_tags
    route_vector : dict
        Route feature vector with difficulty, length_km, tags

    Returns
    -------
    tuple[float, dict]
        Overall similarity score between 0.0 and 1.0, and score breakdown
    """
    # Calculate component scores
    difficulty_score = calculate_difficulty_score(
        user_vector.get("difficulty_range", default_difficulty_range()),
        route_vector["difficulty"]
    )

    distance_score = calculate_distance_score(
        user_vector.get("min_distance_km", DEFAULT_MIN_DISTANCE_KM),
        user_vector.get("max_distance_km", DEFAULT_MAX_DISTANCE_KM),
        route_vector["length_km"]
    )

    tag_score = calculate_tag_score(
        user_vector.get("preferred_tags", []),
        route_vector["tags"]
    )

    # Weighted average
    final_score = (
        SCORE_WEIGHTS["difficulty"] * difficulty_score +
        SCORE_WEIGHTS["distance"] * distance_score +
        SCORE_WEIGHTS["tags"] * tag_score
    )

    # Return score and breakdown
    score_breakdown = {
        "difficulty": {
            "score": difficulty_score,
            "weight": SCORE_WEIGHTS["difficulty"],
            "weighted_score": SCORE_WEIGHTS["difficulty"] * difficulty_score,
            "user_range": user_vector.get("difficulty_range", default_difficulty_range()),
            "route_value": route_vector["difficulty"],
        },
        "distance": {
            "score": distance_score,
            "weight": SCORE_WEIGHTS["distance"],
            "weighted_score": SCORE_WEIGHTS["distance"] * distance_score,
            "user_range": [
                user_vector.get("min_distance_km", DEFAULT_MIN_DISTANCE_KM),
                user_vector.get("max_distance_km", DEFAULT_MAX_DISTANCE_KM),
            ],
            "route_value": route_vector["length_km"],
        },
        "tags": {
            "score": tag_score,
            "weight": SCORE_WEIGHTS["tags"],
            "weighted_score": SCORE_WEIGHTS["tags"] * tag_score,
            "user_tags": user_vector.get("preferred_tags", []),
            "route_tags": route_vector["tags"],
        },
        "total": final_score,
    }

    return final_score, score_breakdown


def calculate_time_decay_weight(days_ago: float, half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS) -> float:
    """
    Calculate time decay weight for feedback.

    More recent feedback has higher weight. Exponential decay with a true half-life:
    ``weight = 0.5 ** (days_ago / half_life_days)``, so the weight is 0.5 after one half-life and
    0.25 after two. Feedback with age <= 0 (just given, or a clock in the future) gets weight 1.0.

    Parameters
    ----------
    days_ago : float
        Number of days since feedback was given
    half_life_days : float
        Half-life in days (default: 30 days)

    Returns
    -------
    float
        Weight between 0.0 and 1.0
    """
    if days_ago <= 0:
        return 1.0
    return 0.5 ** (days_ago / half_life_days)


def _as_utc(moment: datetime) -> datetime:
    """Naive datetimes (SQLite returns these) are taken to be UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def feedback_age_days(created_at: Optional[datetime], now: Optional[datetime] = None) -> float:
    """
    Age of a feedback entry in days.

    ``created_at`` and ``now`` may be naive (treated as UTC) or timezone-aware. A missing ``created_at``
    gives 0.0 (treated as fresh). ``now`` defaults to the current UTC time; pass it for deterministic tests.
    The result is negative if ``created_at`` lies after ``now``.
    """
    if created_at is None:
        return 0.0
    if now is None:
        now = datetime.now(UTC)
    return (_as_utc(now) - _as_utc(created_at)).total_seconds() / 86400.0


def adjust_user_vector_with_feedback(
    user_vector: dict,
    feedback_entries: Iterable[Any],
    route_vectors: dict[int, dict],
    now: Optional[datetime] = None,
) -> dict:
    """
    Adjust user preference vector based on feedback history.

    This function learns from user feedback to adapt preferences:
    - too-hard: Lower maximum difficulty preference
    - too-easy: Raise minimum difficulty preference
    - too-far: Reduce maximum distance preference
    - not-interested: Remove route tags from preferred tags

    Parameters
    ----------
    user_vector : dict
        Original user preference vector
    feedback_entries : Iterable
        User feedback entries: ``ProfileFeedback`` rows or any objects/mappings with ``route_id``,
        ``reason`` and optionally ``created_at``
    route_vectors : dict[int, dict]
        Dictionary mapping route_id to route_vector. Feedback on routes missing here is ignored.
    now : datetime, optional
        Reference time for the time decay (defaults to the current UTC time)

    Returns
    -------
    dict
        Adjusted user preference vector
    """
    # Create a copy to avoid modifying original
    adjusted_vector = user_vector.copy()

    # Initialize adjusted fields
    if "difficulty_range" not in adjusted_vector:
        adjusted_vector["difficulty_range"] = default_difficulty_range()
    else:
        adjusted_vector["difficulty_range"] = list(adjusted_vector["difficulty_range"])

    if "max_distance_km" not in adjusted_vector:
        adjusted_vector["max_distance_km"] = DEFAULT_MAX_DISTANCE_KM
    else:
        adjusted_vector["max_distance_km"] = float(adjusted_vector["max_distance_km"])

    if "preferred_tags" not in adjusted_vector:
        adjusted_vector["preferred_tags"] = []
    else:
        adjusted_vector["preferred_tags"] = list(adjusted_vector["preferred_tags"])

    # Process each feedback entry
    if now is None:
        now = datetime.now(UTC)

    for feedback in feedback_entries:
        route_vector = route_vectors.get(_field(feedback, "route_id"))
        if not route_vector:
            continue

        # Time decay weight from the real age of the feedback (recent feedback matters more)
        days_ago = feedback_age_days(_field(feedback, "created_at"), now)
        weight = calculate_time_decay_weight(days_ago)

        reason = _field(feedback, "reason")

        # Adjust preferences based on feedback reason
        if reason == "too-hard":
            # Lower difficulty preference (shift entire range downward)
            # Both min and max decrease to maintain range width of 1.0
            # This gradually transitions user to easier difficulty levels
            step = DIFFICULTY_FEEDBACK_STEP * weight
            new_min = max(DIFFICULTY_MIN, adjusted_vector["difficulty_range"][0] - step)
            new_max = max(DIFFICULTY_MIN, adjusted_vector["difficulty_range"][1] - step)

            # Edge case protection: Maintain range width of 1.0
            # If min reaches the bottom of the scale, ensure max is at least DIFFICULTY_MIN + 1.0
            if new_min == DIFFICULTY_MIN and new_max < DIFFICULTY_MIN + DIFFICULTY_MIN_RANGE_WIDTH:
                new_max = DIFFICULTY_MIN + DIFFICULTY_MIN_RANGE_WIDTH  # Keep range width = 1.0

            adjusted_vector["difficulty_range"][0] = new_min
            adjusted_vector["difficulty_range"][1] = new_max
        elif reason == "too-easy":
            # Raise difficulty preference (shift entire range upward)
            # Both min and max increase to maintain range width of 1.0
            # This gradually transitions user from beginner → intermediate → advanced
            step = DIFFICULTY_FEEDBACK_STEP * weight
            new_min = min(DIFFICULTY_MAX, adjusted_vector["difficulty_range"][0] + step)
            new_max = min(DIFFICULTY_MAX, adjusted_vector["difficulty_range"][1] + step)

            # Edge case protection: Maintain range width of 1.0
            # If max reaches the top of the scale, ensure min is at most DIFFICULTY_MAX - 1.0
            if new_max == DIFFICULTY_MAX and new_min > DIFFICULTY_MAX - DIFFICULTY_MIN_RANGE_WIDTH:
                new_min = DIFFICULTY_MAX - DIFFICULTY_MIN_RANGE_WIDTH  # Keep range width = 1.0

            adjusted_vector["difficulty_range"][0] = new_min
            adjusted_vector["difficulty_range"][1] = new_max
        elif reason == "too-far":
            # Reduce maximum distance preference
            # Ensure it doesn't go below minimum distance
            new_max = adjusted_vector["max_distance_km"] * (1 - DISTANCE_FEEDBACK_SHRINK * weight)
            # Keep a reasonable gap between min and max (at least 2km)
            min_allowed_max = adjusted_vector.get("min_distance_km", DEFAULT_MIN_DISTANCE_KM) + MIN_DISTANCE_GAP_KM
            adjusted_vector["max_distance_km"] = max(new_max, min_allowed_max)
        elif reason == "not-interested":
            # Remove route tags from preferred tags
            route_tags = route_vector.get("tags", [])
            route_tag_set = set(tag.lower() for tag in route_tags)
            adjusted_vector["preferred_tags"] = [
                tag for tag in adjusted_vector["preferred_tags"]
                if tag.lower() not in route_tag_set
            ]

    # Ensure difficulty range is valid
    if adjusted_vector["difficulty_range"][0] > adjusted_vector["difficulty_range"][1]:
        # If range is invalid, reset to original or default
        adjusted_vector["difficulty_range"] = user_vector.get("difficulty_range", default_difficulty_range())

    # Ensure max_distance is positive
    if adjusted_vector["max_distance_km"] <= 0:
        adjusted_vector["max_distance_km"] = user_vector.get("max_distance_km", DEFAULT_MAX_DISTANCE_KM)

    return adjusted_vector


def calculate_feedback_penalty(
    route_id: int,
    feedback_entries: Iterable[Any]
) -> float:
    """
    Calculate feedback penalty multiplier for a route.

    Penalty schedule:
    - 1 feedback: 50% (0.5)
    - 2 feedbacks: 10% (0.1)
    - 3+ feedbacks: 1% (0.01)

    Parameters
    ----------
    route_id : int
        Route ID to check
    feedback_entries : Iterable
        All user feedback entries (objects or mappings with ``route_id``)

    Returns
    -------
    float
        Penalty multiplier (0.0 to 1.0)
        - 1.0 if no feedback (no penalty)
        - 0.5 if 1 feedback (50%)
        - 0.1 if 2 feedbacks (10%)
        - 0.01 if 3+ feedbacks (1%)
    """
    route_feedback = [f for f in feedback_entries if _field(f, "route_id") == route_id]

    if not route_feedback:
        return 1.0  # No penalty

    feedback_count = len(route_feedback)

    # Apply penalty based on feedback count
    if feedback_count >= 3:
        return FEEDBACK_PENALTY_MULTIPLIERS[3]  # 1%
    elif feedback_count == 2:
        return FEEDBACK_PENALTY_MULTIPLIERS[2]  # 10%
    else:  # feedback_count == 1
        return FEEDBACK_PENALTY_MULTIPLIERS[1]  # 50%


@dataclass
class RankedRoute:
    """One entry of a v0_rules ranking."""

    route_id: int
    score: float
    breakdown: dict


def routes_in_category(routes: Iterable[Any], category: Optional[str]) -> list[Any]:
    """
    Same filter as the DB query: keep routes whose ``category_name`` belongs to ``category``.

    ``category`` None or not in CATEGORY_MAPPING keeps every route. Works on mappings and objects.
    """
    routes = list(routes)
    if not category or category not in CATEGORY_MAPPING:
        return routes
    allowed = set(CATEGORY_MAPPING[category])
    return [r for r in routes if _field(r, "category_name") in allowed]


def _as_route_vector(route: Any) -> dict:
    """Vector for a route given as a ready vector, a raw ``routes``-row mapping, or an ORM Route."""
    if isinstance(route, Mapping):
        if "length_km" in route and "tags" in route and "difficulty" in route:
            return {"difficulty": route["difficulty"], "length_km": route["length_km"], "tags": route["tags"]}
        return route_vector_from_record(route)
    return extract_route_vector(route)


def rank_routes_v0(
    user_vector: dict,
    routes: Sequence[Any],
    feedback_entries: Iterable[Any] = (),
    now: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> list[RankedRoute]:
    """
    The v0_rules ranking, as a pure function (no database, no ORM required).

    Parameters
    ----------
    user_vector : dict
        User preference vector (difficulty_range, min/max_distance_km, preferred_tags)
    routes : Sequence
        Candidate routes, each with an ``id`` and either vector keys (``difficulty``, ``length_km``,
        ``tags``) or ``routes``-row keys (``difficulty``, ``length_meters``, ``tags_json``); ORM ``Route``
        instances also work. Category filtering is the caller's job (see :func:`routes_in_category`).
        The order matters for ties: equal scores keep the input order (stable sort). The DB path passes
        routes ordered by id.
    feedback_entries : Iterable
        The user's feedback: objects or mappings with ``route_id``, ``reason`` and optionally
        ``created_at``. Feedback on routes not in ``routes`` does not adjust the user vector.
    now : datetime, optional
        Reference time for the feedback time decay (default: current UTC time).
    limit : int, optional
        Return only the first ``limit`` entries (Python slice semantics); None returns all.

    Returns
    -------
    list[RankedRoute]
        Routes sorted by score, descending.
    """
    feedback_entries = list(feedback_entries)

    # Build route vectors dictionary for feedback processing
    ordered_ids = []
    route_vectors = {}
    for route in routes:
        route_id = _field(route, "id")
        ordered_ids.append(route_id)
        route_vectors[route_id] = _as_route_vector(route)

    # Adjust user vector based on feedback (learn from user preferences)
    if feedback_entries:
        adjusted_user_vector = adjust_user_vector_with_feedback(user_vector, feedback_entries, route_vectors, now=now)
    else:
        adjusted_user_vector = user_vector

    feedback_counts: dict[Any, int] = {}
    for f in feedback_entries:
        rid = _field(f, "route_id")
        feedback_counts[rid] = feedback_counts.get(rid, 0) + 1

    ranked = []
    for route_id in ordered_ids:
        route_vector = route_vectors[route_id]

        # Check if route should be filtered (too many feedback entries)
        route_feedback_count = feedback_counts.get(route_id, 0)
        if route_feedback_count >= FEEDBACK_FILTER_THRESHOLD:
            # Skip routes with 4+ negative feedback entries (after showing at 1% for 3 feedbacks)
            continue

        # Calculate base CBF score using adjusted user vector
        base_score, score_breakdown = calculate_cbf_score(adjusted_user_vector, route_vector)

        # Apply feedback penalty
        penalty_multiplier = calculate_feedback_penalty(route_id, feedback_entries)
        final_score = base_score * penalty_multiplier

        # Update score breakdown with feedback information
        score_breakdown["feedback_adjusted"] = True
        score_breakdown["base_score"] = base_score
        score_breakdown["final_score"] = final_score
        if penalty_multiplier < 1.0:
            score_breakdown["feedback_penalty"] = penalty_multiplier
            score_breakdown["feedback_count"] = route_feedback_count

        ranked.append(RankedRoute(route_id=route_id, score=final_score, breakdown=score_breakdown))

    # Sort by score (descending); stable, so ties keep the input order
    ranked.sort(key=lambda x: x.score, reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    return ranked


async def get_recommended_routes(
    db: AsyncSession,
    profile_id: Optional[int] = None,
    category: Optional[str] = None,
    limit: int = 20,
    now: Optional[datetime] = None,
) -> list[Route]:
    """
    Get recommended routes using CBF or random selection.

    Parameters
    ----------
    db : AsyncSession
        Database session
    profile_id : Optional[int]
        User profile ID for personalized recommendations. If None, returns random routes.
    category : Optional[str]
        Activity type filter: "running", "hiking", "cycling", or None for all
    limit : int
        Maximum number of routes to return
    now : Optional[datetime]
        Reference time for the feedback time decay (default: current UTC time)

    Returns
    -------
    list[Route]
        List of recommended routes, sorted by relevance (if personalized) or random
    """
    start_time = time.time()

    logger.debug(
        f"🔄 Starting route recommendation calculation: profile_id={profile_id}, category={category}, limit={limit}"
    )

    # Build base query - limit initial load to avoid loading too much data
    # We'll load relationships only for the final selected routes.
    # Ordered by id: the ranking's tie order depends on the input order, so make it explicit.
    query = select(Route).order_by(Route.id)

    # Apply category filter if specified
    if category and category in CATEGORY_MAPPING:
        category_names = CATEGORY_MAPPING[category]
        query = query.where(Route.category_name.in_(category_names))

    # For personalized recommendations, we need to score all routes first
    # For random recommendations, we can limit early
    if profile_id is None:
        # For random, limit early and load relationships only for selected routes
        query = query.limit(limit * 3)  # Get 3x limit for better randomization

    # Execute query (without relationships for now - faster)
    result = await db.execute(query)
    routes = list(result.scalars().all())

    # If no profile_id, return random routes
    if profile_id is None:
        logger.debug(f"🎲 Random recommendation mode: selecting {limit} routes from {len(routes)} candidates")
        random.shuffle(routes)
        selected_routes = routes[:limit]
        # Now load relationships only for selected routes
        route_ids = [r.id for r in selected_routes]
        query_with_relations = select(Route).where(Route.id.in_(route_ids)).options(
            selectinload(Route.breakpoints).selectinload(Breakpoint.mini_quests)
        )
        result_with_relations = await db.execute(query_with_relations)
        routes_with_relations = {r.id: r for r in result_with_relations.scalars().all()}
        final_routes = [routes_with_relations[r.id] for r in selected_routes if r.id in routes_with_relations]
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"✅ Random recommendation completed: returned {len(final_routes)} routes, duration={duration_ms:.2f}ms"
        )
        return final_routes

    # Get user profile and vector
    logger.debug(f"🔍 Fetching user profile and preference vector: profile_id={profile_id}")
    profile = await db.get(DemoProfile, profile_id)
    if not profile or not profile.user_vector_json:
        logger.warning(
            "⚠️ User profile or preference vector not found, falling back to random recommendations: "
            f"profile_id={profile_id}"
        )
        random.shuffle(routes)
        return routes[:limit]

    try:
        user_vector = json.loads(profile.user_vector_json)
        logger.debug(f"✅ User preference vector parsed successfully: {user_vector}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"⚠️ Failed to parse user preference vector, falling back to random recommendations: {e}")
        random.shuffle(routes)
        return routes[:limit]

    # Fetch user feedback entries for feedback-aware recommendations
    logger.debug(f"🔍 Fetching user feedback entries: profile_id={profile_id}")
    feedback_query = select(ProfileFeedback).where(
        ProfileFeedback.demo_profile_id == profile_id
    )
    feedback_result = await db.execute(feedback_query)
    feedback_entries = list(feedback_result.scalars().all())
    logger.debug(f"📊 Number of user feedback entries: {len(feedback_entries)}")

    # Score with the pure v0_rules ranker
    logger.debug(f"📊 Starting CBF score calculation: total_routes={len(routes)}")
    ranked = rank_routes_v0(user_vector, routes, feedback_entries, now=now)
    logger.debug(f"📊 CBF score calculation completed: valid_routes={len(ranked)}")

    # Log top scores for debugging
    for idx, entry in enumerate(ranked[:3], 1):
        logger.debug(f"  {idx}. Route {entry.route_id}: score={entry.score:.4f}")

    selected = ranked[:limit]

    # Now load relationships only for the final selected routes (much faster)
    route_ids = [entry.route_id for entry in selected]
    query_with_relations = select(Route).where(Route.id.in_(route_ids)).options(
        selectinload(Route.breakpoints).selectinload(Breakpoint.mini_quests)
    )
    result_with_relations = await db.execute(query_with_relations)
    routes_with_relations = {r.id: r for r in result_with_relations.scalars().all()}

    # Attach scores as route attributes for the API response, maintaining order
    final_routes = []
    for entry in selected:
        if entry.route_id in routes_with_relations:
            final_route = routes_with_relations[entry.route_id]
            final_route.recommendation_score = entry.score
            final_route.recommendation_score_breakdown = entry.breakdown
            final_routes.append(final_route)

    duration_ms = (time.time() - start_time) * 1000
    log_business_logic(
        logger,
        "calculate",
        "recommended routes",
        entity_id=profile_id,
        routes_count=len(final_routes),
        total_candidates=len(routes),
        feedback_count=len(feedback_entries)
    )

    logger.info(
        f"✅ Personalized recommendation completed: returned {len(final_routes)} routes, duration={duration_ms:.2f}ms"
    )

    return final_routes
