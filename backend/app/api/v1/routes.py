"""
Routes API endpoints.

This module provides endpoints for:
- Route recommendations (US-01, US-02)
- Route list (paged, light payload), route detail and route geometry for the map
- Generating complete stories for routes (US-06, US-07) and retrieving existing story content
"""
import json
import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.schemas import (
    OSM_ATTRIBUTION,
    RecommendationResponse,
    RouteDetailResponse,
    RouteGeometryResponse,
    RouteListResponse,
    RouteResponse,
    RouteSummaryResponse,
    StoryGenerateRequest,
    StoryGenerateResponse,
)
from app.database import get_db
from app.logger import get_logger, log_business_logic, log_request
from app.models.entities import Breakpoint, DemoProfile, Route, RouteGeometry
from app.services import recsys_service
from app.services.recommendation_service import CATEGORY_MAPPING
from app.services.story_generator import generate_story_for_route

logger = get_logger(__name__)

router = APIRouter(prefix="/routes", tags=["routes"])

MAX_PAGE = 1000  # upper bound for `limit` on list endpoints (the catalogue has ~9k routes)


async def _ensure_breakpoints_loaded(db: AsyncSession, routes: list[Route]) -> None:
    """
    Load breakpoints (and their quests) for routes that came back without them, in one query.

    The recommender's fallback path (profile without a user vector) returns routes whose breakpoints
    were never loaded; serialising them would trigger a lazy load, which fails under asyncio (it was
    an HTTP 500, DECISIONS.md D7.2). The ranking itself is untouched.
    """
    missing = [r.id for r in routes if "breakpoints" in sa_inspect(r).unloaded]
    if not missing:
        return
    await db.execute(
        select(Route)
        .where(Route.id.in_(missing))
        .options(selectinload(Route.breakpoints).selectinload(Breakpoint.mini_quests))
        .execution_options(populate_existing=True)
    )


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_route_recommendations(
    request: Request,
    profile_id: Optional[int] = None,
    category: Optional[str] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 20,
    db: Annotated[AsyncSession, Depends(get_db)] = ...,
):
    """
    Route recommendations from the configured system (``RECSYS_SYSTEM``, default ``recsys_serve_v2``).

    Query Parameters:
    - profile_id: optional profile for personalised recommendations (404 if unknown); without it the
      list is uniform random over the tab
    - category: optional tab (running / hiking / cycling); without it the profile's activity, else all
    - limit: number of routes (1-1000). On the pipeline's intent path at most its candidate depth (50)
      come back; the no-intent and baseline paths can return more.

    ``served_by`` says which system and path produced the list and why (docs/SERVING.md). Pipeline paths
    carry a per-route ``explanation`` (per-stage contributions, ``recsys.explanation/v1.1``) and a one-line
    ``reason``; the baseline path keeps ``recommendation_score`` / ``recommendation_score_breakdown``.
    """
    start_time = time.time()
    logger.info(f"🗺️ Fetching route recommendations: profile_id={profile_id}, category={category}, limit={limit}")

    profile = None
    if profile_id is not None:
        profile = await db.get(DemoProfile, profile_id)
        if not profile:
            logger.warning(f"❌ User profile not found: profile_id={profile_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile with id {profile_id} not found"
            )

    res = getattr(request.app.state, "recsys", None)
    served = await recsys_service.recommend(
        db, res=res, profile=profile, category=category, limit=limit, request=request
    )
    await _ensure_breakpoints_loaded(db, served.routes)

    route_responses = []
    for route in served.routes:
        route_dict = RouteResponse.model_validate(route).model_dump()
        route_dict["is_locked"] = bool(profile and profile.total_xp < route.xp_required)
        if hasattr(route, "recommendation_score"):  # baseline path (set by recommendation_service)
            route_dict["recommendation_score"] = route.recommendation_score
        if hasattr(route, "recommendation_score_breakdown"):
            route_dict["recommendation_score_breakdown"] = route.recommendation_score_breakdown
        route_dict.update(served.extras.get(route.id, {}))
        route_responses.append(RouteResponse(**route_dict))

    duration_ms = (time.time() - start_time) * 1000
    sb = served.served_by
    log_request(
        logger,
        "GET",
        "/api/routes/recommendations",
        status_code=200,
        duration_ms=duration_ms,
        user_id=profile_id,
        category=category,
        routes_count=len(route_responses),
        is_personalized=served.is_personalized,
        system=sb.get("system"),
        served_path=sb.get("path"),
        reason=sb.get("reason"),
    )
    return RecommendationResponse(
        routes=route_responses,
        total=len(route_responses),
        is_personalized=served.is_personalized,
        served_by=sb,
    )


@router.get("", response_model=RouteListResponse)
async def list_routes(
    db: Annotated[AsyncSession, Depends(get_db)],
    category: Optional[str] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """
    Paged list of routes ordered by id, with a light payload (no breakpoints, story texts or GPX).

    ``category`` is one of the tabs (``running``, ``hiking``, ``cycling``); anything else lists all.
    ``total`` counts every route matching the filter.
    """
    query = select(Route)
    count_query = select(func.count(Route.id))
    if category and category in CATEGORY_MAPPING:
        query = query.where(Route.category_name.in_(CATEGORY_MAPPING[category]))
        count_query = count_query.where(Route.category_name.in_(CATEGORY_MAPPING[category]))
    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(query.order_by(Route.id).limit(limit).offset(offset))
    routes = [RouteSummaryResponse.model_validate(r) for r in result.scalars().all()]
    return RouteListResponse(routes=routes, total=total, limit=limit, offset=offset)


@router.get("/{route_id}", response_model=RouteDetailResponse)
async def get_route(route_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    """One route with breakpoints and mini quests. The line itself is at ``/routes/{id}/geometry``."""
    result = await db.execute(
        select(Route)
        .where(Route.id == route_id)
        .options(selectinload(Route.breakpoints).selectinload(Breakpoint.mini_quests))
    )
    route = result.scalar_one_or_none()
    if route is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Route with id {route_id} not found")
    return RouteDetailResponse.model_validate(route)


@router.get("/{route_id}/geometry", response_model=RouteGeometryResponse)
async def get_route_geometry(route_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    """
    The route's simplified line (``[[lon, lat], ...]``) and elevation profile, for the map.

    404 when the route does not exist or has no stored geometry (legacy routes).
    """
    row = (
        await db.execute(
            select(RouteGeometry, Route.length_meters, Route.source, Route.source_url)
            .join(Route, Route.id == RouteGeometry.route_id)
            .where(RouteGeometry.route_id == route_id)
        )
    ).one_or_none()
    if row is None:
        exists = await db.get(Route, route_id)
        detail = (f"Route {route_id} has no stored geometry" if exists
                  else f"Route with id {route_id} not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    geom, length_m, source, source_url = row
    coords = json.loads(geom.coords_json)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return RouteGeometryResponse(
        route_id=route_id,
        coords=coords,
        bbox=[min(lons), min(lats), max(lons), max(lats)],
        ele_profile_m=json.loads(geom.ele_profile_json) if geom.ele_profile_json else None,
        ele_step_m=geom.ele_step_m,
        length_m=length_m,
        source=source,
        source_url=source_url,
        attribution=OSM_ATTRIBUTION if source == "osm" else None,
    )


@router.post("/{route_id}/generate-story", response_model=StoryGenerateResponse)
async def generate_route_story(
    route_id: int,
    request: StoryGenerateRequest,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Generate complete story for specified route and save to database.

    This endpoint implements the Two-Step Global Generation approach:
    1. If story exists and force_regenerate=False, return existing (0 delay)
    2. Otherwise, generate new story using Llama3.1:8b via Ollama (10-15s)
    3. Save to database for future use

    Args:
        route_id: ID of the route to generate story for
        request: Story generation parameters
        db: Database session

    Returns:
        Complete story with title, outline, prologue, epilogue, and breakpoint content

    Raises:
        HTTPException: 404 if route not found, 400 if no breakpoints
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info(f"📖 Generating route story: route_id={route_id}, narrative_style={request.narrative_style}, force_regenerate={request.force_regenerate}")  # noqa: E501

    # 1. Fetch route with breakpoints and mini_quests
    logger.debug(f"🔍 Fetching route information: route_id={route_id}")
    result = await db.execute(
        select(Route)
        .where(Route.id == route_id)
        .options(
            selectinload(Route.breakpoints).selectinload(Breakpoint.mini_quests)
        )
    )
    route = result.scalar_one_or_none()

    if not route:
        logger.error(f"❌ Route not found: route_id={route_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Route with id {route_id} not found"
        )

    logger.debug(f"✅ Route fetched successfully: route_id={route_id}, breakpoints_count={len(route.breakpoints) if route.breakpoints else 0}")  # noqa: E501

    if not route.breakpoints:
        logger.error(f"❌ Route has no breakpoints: route_id={route_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Route has no breakpoints. Cannot generate story."
        )

    # 2. Check if story already exists
    if route.story_prologue_body and not request.force_regenerate:
        logger.info("📚 Using existing story (skipping generation)")
        duration_ms = (time.time() - start_time) * 1000
        log_request(
            logger,
            "POST",
            f"/api/routes/{route_id}/generate-story",
            status_code=200,
            duration_ms=duration_ms,
            cached=True
        )
        logger.info(f"✅ Story returned from cache: duration={duration_ms:.2f}ms")
        logger.info("=" * 80)
        return _assemble_existing_story(route)

    # 3. Generate new story
    logger.info("🤖 Starting new story generation...")
    story_data = await generate_story_for_route(
        route=route,
        breakpoints=route.breakpoints,
        narrative_style=request.narrative_style
    )
    logger.info("✅ Story generation completed")

    # 4. Save to database
    logger.debug("💾 Saving story to database...")
    await _save_story_to_db(route, story_data, db)
    logger.info("✅ Story saved to database")

    duration_ms = (time.time() - start_time) * 1000
    log_business_logic(
        logger,
        "generate",
        "route story",
        entity_id=route_id,
        narrative_style=request.narrative_style,
        breakpoints_count=len(route.breakpoints)
    )

    log_request(
        logger,
        "POST",
        f"/api/routes/{route_id}/generate-story",
        status_code=200,
        duration_ms=duration_ms,
        cached=False
    )

    logger.info(f"✅ Story generated and returned successfully: duration={duration_ms:.2f}ms")
    logger.info("=" * 80)

    # 5. Return result
    return StoryGenerateResponse(**story_data)


@router.get("/{route_id}/story", response_model=StoryGenerateResponse)
async def get_route_story(
    route_id: int,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Retrieve existing story for a route.

    Args:
        route_id: ID of the route
        db: Database session

    Returns:
        Existing story content

    Raises:
        HTTPException: 404 if route not found or no story exists
    """
    result = await db.execute(
        select(Route)
        .where(Route.id == route_id)
        .options(
            selectinload(Route.breakpoints).selectinload(Breakpoint.mini_quests)
        )
    )
    route = result.scalar_one_or_none()

    if not route:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Route with id {route_id} not found"
        )

    if not route.story_prologue_body:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No story generated for this route yet"
        )

    return _assemble_existing_story(route)


async def _save_story_to_db(
    route: Route,
    story_data: dict,
    db: AsyncSession
):
    """
    Save generated story to database.

    Updates Route table with prologue and epilogue,
    Breakpoint table with main_quest for each breakpoint,
    and MiniQuest table with generated mini quests.

    Args:
        route: Route entity to update
        story_data: Generated story data
        db: Database session
    """
    from app.models.entities import MiniQuest

    # Update Route table
    route.story_prologue_title = story_data["title"]
    route.story_prologue_body = story_data["prologue"]
    route.story_epilogue_body = story_data["epilogue"]

    # Update Breakpoint table and create MiniQuests
    for bp_data in story_data["breakpoints"]:
        idx = bp_data["index"]
        if idx < len(route.breakpoints):
            bp = route.breakpoints[idx]
            bp.main_quest_snippet = bp_data["main_quest"]

            # Clear existing mini quests for this breakpoint
            for existing_quest in bp.mini_quests:
                await db.delete(existing_quest)

            # Create new mini quests
            mini_quests_data = bp_data.get("mini_quests", [])
            for quest_data in mini_quests_data:
                mini_quest = MiniQuest(
                    breakpoint_id=bp.id,
                    task_description=quest_data["task_description"],
                    xp_reward=quest_data["xp_reward"]
                )
                db.add(mini_quest)

    await db.commit()
    await db.refresh(route)


def _assemble_existing_story(route: Route) -> dict:
    """
    Assemble existing story data from database.

    Args:
        route: Route entity with story data

    Returns:
        dict matching StoryGenerateResponse format
    """
    return {
        "title": route.story_prologue_title or "Untitled Adventure",
        "outline": "Existing story outline",  # Could be stored separately if needed
        "prologue": route.story_prologue_body or "",
        "epilogue": route.story_epilogue_body or "",
        "breakpoints": [
            {
                "index": bp.order_index,
                "main_quest": bp.main_quest_snippet or "",
                "mini_quests": [
                    {
                        "task_description": quest.task_description,
                        "xp_reward": quest.xp_reward
                    }
                    for quest in bp.mini_quests
                ]
            }
            for bp in sorted(route.breakpoints, key=lambda x: x.order_index)
        ]
    }

