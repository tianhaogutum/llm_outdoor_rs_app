"""
Domain database models for the hiking simulator backend.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class DemoProfile(Base):
    """
    Temporary player profile captured after the US-03 questionnaire.
    """

    __tablename__ = "demo_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_xp: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    level: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    user_vector_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    genai_welcome_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    unlocked_routes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Profile slots for hard filters and retrieval (all optional; absent = no constraint).
    # Home location: a name plus WGS84 coordinates, usually resolved from the catalogue's localities
    # (GET /api/places). Travel radius limits how far a route's start may be from home; the distance
    # cap is an absolute upper bound on route length. The intent is the user's own words.
    home_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    home_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    travel_radius_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_cap_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intent_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Phase 4 chat: activity slot (hiking|cycling|mtb|running) and the server-side conversation state
    # (app/chat/schema.py ChatState as JSON). Both null until the first chat turn.
    activity: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    chat_state_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The questionnaire's user vector as created, never rewritten by feedback. The baseline's feedback
    # endpoint rewrites ``user_vector_json`` on every feedback (DECISIONS.md D7.1); the pipeline starts from
    # this vector instead (plus the chat's slots) and applies feedback once (docs/SERVING.md).
    questionnaire_vector_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    souvenirs: Mapped[list[Souvenir]] = relationship(
        "Souvenir",
        back_populates="demo_profile",
        cascade="all, delete-orphan",
    )
    feedback_entries: Mapped[list[ProfileFeedback]] = relationship(
        "ProfileFeedback",
        back_populates="demo_profile",
        cascade="all, delete-orphan",
    )
    achievements: Mapped[list[ProfileAchievement]] = relationship(
        "ProfileAchievement",
        back_populates="demo_profile",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<DemoProfile id={self.id} level={self.level} total_xp={self.total_xp}>"


class Route(Base):
    """
    Core route metadata plus custom storytelling fields.

    Two catalogues can fill this table (one at a time): the legacy Outdooractive seed and the OSM
    catalogue (docs/DATA.md). The catalogue columns after ``story_epilogue_body`` are NULL for legacy
    routes.

    ``elevation`` is the total **ascent** in metres for both catalogues (the legacy column name is
    kept because XP, statistics and the v0_rules code read it); ``ascent_m`` is a read-only alias.
    ``descent_m`` is new and only known for OSM routes.
    """

    __tablename__ = "routes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    category_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    length_meters: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 0-3 (easy, moderate, difficult, expert); see DIFFICULTY_MIN / DIFFICULTY_MAX in recommendation_service
    difficulty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    short_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    elevation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gpx_data_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    xp_required: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    base_xp_reward: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    story_prologue_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    story_prologue_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    story_epilogue_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --- catalogue facts (OSM catalogue; NULL for legacy routes) ---
    source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "osm"
    source_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # OSM relation page
    activity: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # hiking|cycling|mtb|running
    start_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_loop: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    descent_m: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    locality: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    admin_district: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    admin_region: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Compact display facts (highlights, access, surface, ...); see scripts/catalog_seed.py:display_facts
    facts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    breakpoints: Mapped[list[Breakpoint]] = relationship(
        "Breakpoint",
        back_populates="route",
        cascade="all, delete-orphan",
        order_by="Breakpoint.order_index",
    )
    souvenirs: Mapped[list[Souvenir]] = relationship(
        "Souvenir",
        back_populates="route",
        cascade="all, delete-orphan",
    )
    feedback_entries: Mapped[list[ProfileFeedback]] = relationship(
        "ProfileFeedback",
        back_populates="route",
        cascade="all, delete-orphan",
    )

    @property
    def ascent_m(self) -> Optional[int]:
        """Total ascent in metres (stored in the legacy ``elevation`` column)."""
        return self.elevation

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Route id={self.id} title={self.title!r}>"


class RouteGeometry(Base):
    """
    Simplified line and elevation profile of a route, kept out of ``routes`` so that loading all
    routes (the recommender does) stays light. Served by ``GET /api/routes/{id}/geometry``.
    """

    __tablename__ = "route_geometries"

    route_id: Mapped[int] = mapped_column(ForeignKey("routes.id", ondelete="CASCADE"), primary_key=True)
    coords_json: Mapped[str] = mapped_column(Text, nullable=False)  # [[lon, lat], ...]
    ele_profile_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # [m, ...] every ele_step_m
    ele_step_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<RouteGeometry route_id={self.route_id}>"


class Place(Base):
    """
    Localities named by the catalogue (the ``locality`` of route starts), for resolving a typed home
    location without an external geocoder. Coordinates are those of the route start closest to the
    locality's place node; ``accuracy_m`` is that start's distance to the node.
    """

    __tablename__ = "places"
    __table_args__ = (Index("ix_places_name_key", "name_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    name_key: Mapped[str] = mapped_column(String(255), nullable=False)  # normalised for matching
    admin_district: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    admin_region: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy_m: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    route_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Place id={self.id} name={self.name!r}>"


class Breakpoint(Base):
    """
    Route progress nodes and story chapters.
    """

    __tablename__ = "breakpoints"
    __table_args__ = (Index("ix_breakpoints_route_id", "route_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_id: Mapped[int] = mapped_column(
        ForeignKey("routes.id", ondelete="CASCADE"),
        nullable=False,
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    poi_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    poi_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    main_quest_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Position along the route in metres (derived OSM breakpoints; NULL for legacy hand-made ones)
    at_m: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    route: Mapped[Route] = relationship("Route", back_populates="breakpoints")
    mini_quests: Mapped[list[MiniQuest]] = relationship(
        "MiniQuest",
        back_populates="breakpoint",
        cascade="all, delete-orphan",
        order_by="MiniQuest.id",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Breakpoint id={self.id} route_id={self.route_id} idx={self.order_index}>"


class MiniQuest(Base):
    """
    Optional mini quests that fire at a specific breakpoint.
    """

    __tablename__ = "mini_quests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    breakpoint_id: Mapped[int] = mapped_column(
        ForeignKey("breakpoints.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    xp_reward: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    breakpoint: Mapped[Breakpoint] = relationship("Breakpoint", back_populates="mini_quests")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<MiniQuest id={self.id} breakpoint_id={self.breakpoint_id}>"


class Souvenir(Base):
    """
    Completion record capturing XP gain and AI generated summaries.
    """

    __tablename__ = "souvenirs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    demo_profile_id: Mapped[int] = mapped_column(
        ForeignKey("demo_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    route_id: Mapped[int] = mapped_column(
        ForeignKey("routes.id", ondelete="CASCADE"),
        nullable=False,
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    total_xp_gained: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    genai_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    xp_breakdown_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pixel_image_svg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # LLM-generated pixel art SVG

    demo_profile: Mapped[DemoProfile] = relationship("DemoProfile", back_populates="souvenirs")
    route: Mapped[Route] = relationship("Route", back_populates="souvenirs")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Souvenir id={self.id} profile_id={self.demo_profile_id} route_id={self.route_id}>"


class ProfileFeedback(Base):
    """
    👎 feedback captured from US-08 to drive adaptive re-ranking.
    """

    __tablename__ = "profile_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    demo_profile_id: Mapped[int] = mapped_column(
        ForeignKey("demo_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    route_id: Mapped[int] = mapped_column(
        ForeignKey("routes.id", ondelete="CASCADE"),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(String(100), nullable=False)
    # When the feedback was given; drives the time decay in the recommender.
    # SQLite stores CURRENT_TIMESTAMP (UTC) and returns naive datetimes: treat them as UTC.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    demo_profile: Mapped[DemoProfile] = relationship("DemoProfile", back_populates="feedback_entries")
    route: Mapped[Route] = relationship("Route", back_populates="feedback_entries")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<ProfileFeedback id={self.id} reason={self.reason!r}>"


class Achievement(Base):
    """
    Achievement definitions for gamification.
    """

    __tablename__ = "achievements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    achievement_key: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    icon: Mapped[str] = mapped_column(String(10), nullable=False)
    # route_count, route_type, level, xp, distance
    condition_type: Mapped[str] = mapped_column(String(20), nullable=False)
    condition_value: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string

    profile_achievements: Mapped[list[ProfileAchievement]] = relationship(
        "ProfileAchievement",
        back_populates="achievement",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Achievement id={self.id} key={self.achievement_key!r}>"


class ProfileAchievement(Base):
    """
    User achievement unlock records.
    """

    __tablename__ = "profile_achievements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    demo_profile_id: Mapped[int] = mapped_column(
        ForeignKey("demo_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    achievement_id: Mapped[int] = mapped_column(
        ForeignKey("achievements.id", ondelete="CASCADE"),
        nullable=False,
    )
    unlocked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    demo_profile: Mapped[DemoProfile] = relationship("DemoProfile", back_populates="achievements")
    achievement: Mapped[Achievement] = relationship("Achievement", back_populates="profile_achievements")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ProfileAchievement id={self.id} profile_id={self.demo_profile_id} "
            f"achievement_id={self.achievement_id}>"
        )


