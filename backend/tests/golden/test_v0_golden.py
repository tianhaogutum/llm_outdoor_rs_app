"""
Regression test: v0_rules must reproduce the ranking of the original scorer exactly.

The golden file was produced by ``make_golden_v0.py`` from the pre-refactor scorer (git ``main``) on the
111 seed routes, for every fitness level x adventure type of the questionnaire, per category and overall,
with no feedback. Both the pure function and the DB path are checked against it, bit for bit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.services import recommendation_service as rs
from tests.conftest import GOLDEN_FILE
from tests.golden.make_golden_v0 import (
    ADVENTURE_TYPES,
    CATEGORIES,
    FITNESS_LEVELS,
    golden_user_vector,
    ranking_row,
)

GOLDEN = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
CASES = GOLDEN["cases"]
CASE_IDS = [f"{c['fitness']}-{c['adventure_type']}-{c['category'] or 'all'}" for c in CASES]
FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_golden_covers_the_full_grid():
    grid = {(c["fitness"], c["adventure_type"], c["category"]) for c in CASES}
    assert grid == {(f, a, c) for f in FITNESS_LEVELS for a in ADVENTURE_TYPES for c in CATEGORIES}
    assert all(len(c["ranking"]) == 111 for c in CASES if c["category"] is None)


def test_golden_user_vectors_match_questionnaire_mapping():
    for case in CASES:
        assert golden_user_vector(case["fitness"], case["adventure_type"]) == case["user_vector"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_pure_ranker_matches_golden(case, seed_records):
    routes = rs.routes_in_category(sorted(seed_records, key=lambda r: r["id"]), case["category"])
    ranked = rs.rank_routes_v0(case["user_vector"], routes, [], now=FIXED_NOW)
    assert [ranking_row(r.route_id, r.score, r.breakdown) for r in ranked] == case["ranking"]


def test_pure_ranker_limit_is_a_prefix(seed_records):
    case = CASES[0]
    routes = sorted(seed_records, key=lambda r: r["id"])
    top = rs.rank_routes_v0(case["user_vector"], routes, [], limit=20)
    assert [r.route_id for r in top] == [row[0] for row in case["ranking"][:20]]


def test_ties_keep_input_order(seed_records):
    """Equal scores keep the input order: that is why the DB path orders candidates by id."""
    case = next(c for c in CASES if c["category"] is None)
    routes = sorted(seed_records, key=lambda r: r["id"])
    forward = rs.rank_routes_v0(case["user_vector"], routes, [])
    backward = rs.rank_routes_v0(case["user_vector"], list(reversed(routes)), [])
    assert [r.score for r in forward] == [r.score for r in backward]
    assert [r.route_id for r in forward] != [r.route_id for r in backward]  # the grid has ties


async def test_db_path_matches_golden(seeded_db_copy):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.models.entities import DemoProfile

    engine = create_async_engine(f"sqlite+aiosqlite:///{seeded_db_copy}")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as db:
            profiles = {}
            for case in CASES:
                key = (case["fitness"], case["adventure_type"])
                if key not in profiles:
                    profile = DemoProfile(user_vector_json=json.dumps(case["user_vector"]), total_xp=0, level=1)
                    db.add(profile)
                    await db.commit()
                    profiles[key] = profile.id
                routes = await rs.get_recommended_routes(
                    db=db, profile_id=profiles[key], category=case["category"], limit=10_000, now=FIXED_NOW
                )
                got = [ranking_row(r.id, r.recommendation_score, r.recommendation_score_breakdown) for r in routes]
                assert got == case["ranking"], CASE_IDS[CASES.index(case)]
                db.expunge_all()
    finally:
        await engine.dispose()
