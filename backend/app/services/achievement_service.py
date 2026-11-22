"""
Achievement service for checking and unlocking user achievements.
"""
from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.entities import (
    Achievement,
    DemoProfile,
    ProfileAchievement,
    Route,
    Souvenir,
)


async def get_all_achievements(db: AsyncSession) -> list[Achievement]:
    """
    Get all achievement definitions.
    """
    result = await db.execute(select(Achievement).order_by(Achievement.id))
    return list(result.scalars().all())


async def get_user_achievements(
    profile_id: int, db: AsyncSession
) -> list[ProfileAchievement]:
    """
    Get all achievements unlocked by a user.
    """
    result = await db.execute(
        select(ProfileAchievement)
        .where(ProfileAchievement.demo_profile_id == profile_id)
        .options(selectinload(ProfileAchievement.achievement))
    )
    return list(result.scalars().all())


async def check_achievement_condition(
    achievement: Achievement,
    profile: DemoProfile,
    completed_routes: list[Route],
    db: AsyncSession,
) -> bool:
    """
    Check if a user meets the condition for an achievement.

    Args:
        achievement: The achievement to check
        profile: User profile
        completed_routes: List of completed routes (with route data)
        db: Database session

    Returns:
        True if condition is met, False otherwise
    """
    condition_type = achievement.condition_type
    condition_value = json.loads(achievement.condition_value)

    if condition_type == "route_count":
        required_count = condition_value.get("count", 1)
        return len(completed_routes) >= required_count

    elif condition_type == "route_type":
        required_type = condition_value.get("type")  # hiking, running, cycling
        return any(
            _map_category_to_type(route.category_name) == required_type
            for route in completed_routes
        )

    elif condition_type == "level":
        required_level = condition_value.get("level", 5)
        return profile.level >= required_level

    elif condition_type == "xp":
        required_xp = condition_value.get("xp", 1000)
        return profile.total_xp >= required_xp

    elif condition_type == "distance":
        required_distance_km = condition_value.get("distance_km", 50)
        total_distance_m = sum(
            route.length_meters or 0 for route in completed_routes
        )
        total_distance_km = total_distance_m / 1000.0
        return total_distance_km >= required_distance_km

    elif condition_type == "all_route_types":
        # Check if user has completed all route types (hiking, running, cycling)
        required_types = set(condition_value.get("types", ["hiking", "running", "cycling"]))
        completed_types = set(
            _map_category_to_type(route.category_name)
            for route in completed_routes
            if _map_category_to_type(route.category_name) is not None
        )
        return required_types.issubset(completed_types)

    elif condition_type == "single_route_distance":
        # Check if any single route exceeds the required distance
        required_distance_km = condition_value.get("distance_km", 30)
        return any(
            (route.length_meters or 0) / 1000.0 >= required_distance_km
            for route in completed_routes
        )

    elif condition_type == "route_difficulty":
        # Check if user has completed a route of specified difficulty
        # difficulty: 0=easy, 1=medium, 2=hard, 3+=expert
        required_difficulty = condition_value.get("difficulty", 2)
        return any(
            (route.difficulty or 0) >= required_difficulty
            for route in completed_routes
        )

    return False


def _map_category_to_type(category_name: Optional[str]) -> Optional[str]:
    """
    Map backend category_name to frontend route type.
    """
    if not category_name:
        return None

    lower = category_name.lower()
    if "run" in lower or "jogging" in lower:
        return "running"
    if "cycling" in lower or "mountain" in lower or "bike" in lower:
        return "cycling"
    return "hiking"  # Default to hiking


async def check_and_unlock_achievements(
    profile_id: int, db: AsyncSession
) -> list[Achievement]:
    """
    Check all achievements and unlock any that the user qualifies for.

    Returns:
        List of newly unlocked achievements
    """
    # Get user profile
    profile = await db.get(DemoProfile, profile_id)
    if not profile:
        return []

    # Get all achievements
    all_achievements = await get_all_achievements(db)

    # Get user's already unlocked achievements
    unlocked_result = await db.execute(
        select(ProfileAchievement.achievement_id).where(
            ProfileAchievement.demo_profile_id == profile_id
        )
    )
    unlocked_achievement_ids = set(unlocked_result.scalars().all())

    # Get user's completed routes
    souvenirs_result = await db.execute(
        select(Souvenir)
        .where(Souvenir.demo_profile_id == profile_id)
        .options(selectinload(Souvenir.route))
    )
    souvenirs = list(souvenirs_result.scalars().all())
    completed_routes = [souvenir.route for souvenir in souvenirs if souvenir.route]

    # Check each achievement
    newly_unlocked: list[Achievement] = []

    for achievement in all_achievements:
        # Skip if already unlocked
        if achievement.id in unlocked_achievement_ids:
            continue

        # Check condition
        if await check_achievement_condition(
            achievement, profile, completed_routes, db
        ):
            # Unlock achievement
            profile_achievement = ProfileAchievement(
                demo_profile_id=profile_id,
                achievement_id=achievement.id,
            )
            db.add(profile_achievement)
            newly_unlocked.append(achievement)

    if newly_unlocked:
        await db.commit()

    return newly_unlocked


async def seed_achievements(db: AsyncSession) -> None:
    """
    Seed the database with default achievements.
    """
    achievements_data = [
        {
            "achievement_key": "first-steps",
            "name": "First Steps",
            "description": "Complete your first route",
            "icon": "🥾",
            "condition_type": "route_count",
            "condition_value": json.dumps({"count": 1}),
        },
        {
            "achievement_key": "explorer",
            "name": "Explorer",
            "description": "Complete 3 different routes",
            "icon": "🗺️",
            "condition_type": "route_count",
            "condition_value": json.dumps({"count": 3}),
        },
        {
            "achievement_key": "hiker",
            "name": "Trail Hiker",
            "description": "Complete a hiking route",
            "icon": "⛰️",
            "condition_type": "route_type",
            "condition_value": json.dumps({"type": "hiking"}),
        },
        {
            "achievement_key": "runner",
            "name": "Trail Runner",
            "description": "Complete a running route",
            "icon": "🏃",
            "condition_type": "route_type",
            "condition_value": json.dumps({"type": "running"}),
        },
        {
            "achievement_key": "cyclist",
            "name": "Cyclist",
            "description": "Complete a cycling route",
            "icon": "🚴",
            "condition_type": "route_type",
            "condition_value": json.dumps({"type": "cycling"}),
        },
        {
            "achievement_key": "level-5",
            "name": "Rising Star",
            "description": "Reach Level 5",
            "icon": "⭐",
            "condition_type": "level",
            "condition_value": json.dumps({"level": 5}),
        },
        {
            "achievement_key": "xp-1000",
            "name": "XP Collector",
            "description": "Earn 1000 total XP",
            "icon": "💎",
            "condition_type": "xp",
            "condition_value": json.dumps({"xp": 1000}),
        },
        {
            "achievement_key": "distance-50",
            "name": "Long Distance",
            "description": "Travel 50km total",
            "icon": "🎯",
            "condition_type": "distance",
            "condition_value": json.dumps({"distance_km": 50}),
        },
        # New easier achievements for better progression
        {
            "achievement_key": "early-bird",
            "name": "Early Bird",
            "description": "Complete 2 different routes",
            "icon": "🐦",
            "condition_type": "route_count",
            "condition_value": json.dumps({"count": 2}),
        },
        {
            "achievement_key": "adventurer",
            "name": "Adventurer",
            "description": "Complete 5 different routes",
            "icon": "🗺️",
            "condition_type": "route_count",
            "condition_value": json.dumps({"count": 5}),
        },
        {
            "achievement_key": "weekend-warrior",
            "name": "Weekend Warrior",
            "description": "Reach Level 3",
            "icon": "💪",
            "condition_type": "level",
            "condition_value": json.dumps({"level": 3}),
        },
        # Top 5 New Achievements
        {
            "achievement_key": "all-rounder",
            "name": "All-Rounder",
            "description": "Complete at least one hiking, running, and cycling route",
            "icon": "🌈",
            "condition_type": "all_route_types",
            "condition_value": json.dumps({"types": ["hiking", "running", "cycling"]}),
        },
        {
            "achievement_key": "souvenir-collector",
            "name": "Souvenir Collector",
            "description": "Collect 10 digital souvenirs",
            "icon": "🎁",
            "condition_type": "route_count",
            "condition_value": json.dumps({"count": 10}),
        },
        {
            "achievement_key": "marathoner",
            "name": "Marathoner",
            "description": "Complete a route longer than 30km",
            "icon": "🏅",
            "condition_type": "single_route_distance",
            "condition_value": json.dumps({"distance_km": 30}),
        },
        {
            "achievement_key": "century-club",
            "name": "Century Club",
            "description": "Travel 100km total distance",
            "icon": "💯",
            "condition_type": "distance",
            "condition_value": json.dumps({"distance_km": 100}),
        },
        {
            "achievement_key": "mountain-goat",
            "name": "Mountain Goat",
            "description": "Complete a difficult route",
            "icon": "🐐",
            "condition_type": "route_difficulty",
            "condition_value": json.dumps({"difficulty": 2}),
        },
    ]

    for data in achievements_data:
        # Check if achievement already exists
        result = await db.execute(
            select(Achievement).where(
                Achievement.achievement_key == data["achievement_key"]
            )
        )
        existing = result.scalar_one_or_none()

        if not existing:
            achievement = Achievement(**data)
            db.add(achievement)

    await db.commit()

