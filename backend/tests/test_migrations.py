"""Alembic migrations: profile_feedback.created_at; OSM catalogue columns, geometries, places, profile slots."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import run_alembic

PREVIOUS_HEAD = "e4fda692220c"
CATALOG_REVISION = "9b4e2d7f1a36"
BEFORE_CATALOG = "c1a7c85ec9fa"
NEW_ROUTE_COLUMNS = {"source", "source_url", "activity", "start_lat", "start_lon", "is_loop", "descent_m",
                     "locality", "admin_district", "admin_region", "facts_json"}
NEW_PROFILE_COLUMNS = {"home_name", "home_lat", "home_lon", "travel_radius_km", "distance_cap_km", "intent_text"}


def columns(db: Path, table: str) -> dict[str, tuple]:
    conn = sqlite3.connect(db)
    try:
        # cid, name, type, notnull, dflt_value, pk
        return {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_upgrade_on_empty_db_adds_created_at(tmp_path):
    db = tmp_path / "empty.db"
    run_alembic(db, "upgrade", "head")
    col = columns(db, "profile_feedback")["created_at"]
    assert col[3] == 1  # NOT NULL
    assert "CURRENT_TIMESTAMP" in col[4]


def test_existing_feedback_rows_get_a_timestamp_and_downgrade_removes_it(tmp_path):
    db = tmp_path / "old.db"
    run_alembic(db, "upgrade", PREVIOUS_HEAD)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO demo_profiles (id) VALUES (1)")
    conn.execute("INSERT INTO routes (id, title) VALUES (7, 'r')")
    conn.execute("INSERT INTO profile_feedback (demo_profile_id, route_id, reason) VALUES (1, 7, 'too-hard')")
    conn.commit()
    conn.close()

    run_alembic(db, "upgrade", "head")
    conn = sqlite3.connect(db)
    (created_at,) = conn.execute("SELECT created_at FROM profile_feedback").fetchone()
    conn.execute("INSERT INTO profile_feedback (demo_profile_id, route_id, reason) VALUES (1, 7, 'too-far')")
    conn.commit()
    new_rows = conn.execute("SELECT created_at FROM profile_feedback WHERE reason = 'too-far'").fetchall()
    conn.close()
    assert created_at is not None and len(created_at) == 19  # 'YYYY-MM-DD HH:MM:SS', UTC
    assert new_rows[0][0] is not None

    run_alembic(db, "downgrade", PREVIOUS_HEAD)
    assert "created_at" not in columns(db, "profile_feedback")
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM profile_feedback").fetchone()[0] == 2
    conn.close()


def test_models_match_migrations(tmp_path):
    db = tmp_path / "check.db"
    run_alembic(db, "upgrade", "head")
    result = run_alembic(db, "check")
    assert "No new upgrade operations detected" in result.stdout + result.stderr


def tables(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def indexes(db: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {r[1] for r in conn.execute(f"PRAGMA index_list({table})")}
    finally:
        conn.close()


def counts(db: Path, *names: str) -> list[int]:
    conn = sqlite3.connect(db)
    try:
        return [conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in names]
    finally:
        conn.close()


def test_catalog_migration_up_and_down_on_empty_db(tmp_path):
    db = tmp_path / "empty.db"
    run_alembic(db, "upgrade", "head")
    assert NEW_ROUTE_COLUMNS <= set(columns(db, "routes"))
    assert NEW_PROFILE_COLUMNS <= set(columns(db, "demo_profiles"))
    assert "at_m" in columns(db, "breakpoints") and "ix_breakpoints_route_id" in indexes(db, "breakpoints")
    assert {"route_geometries", "places"} <= tables(db)
    assert all(columns(db, "routes")[c][3] == 0 for c in NEW_ROUTE_COLUMNS)  # all nullable

    run_alembic(db, "downgrade", BEFORE_CATALOG)
    assert not NEW_ROUTE_COLUMNS & set(columns(db, "routes"))
    assert not NEW_PROFILE_COLUMNS & set(columns(db, "demo_profiles"))
    assert "at_m" not in columns(db, "breakpoints")
    assert not {"route_geometries", "places"} & tables(db)
    run_alembic(db, "upgrade", "head")  # and back up again
    assert NEW_ROUTE_COLUMNS <= set(columns(db, "routes"))


def test_catalog_migration_on_legacy_seeded_db_keeps_all_rows(seeded_db_copy):
    """Downgrading and re-upgrading a seeded DB keeps routes, breakpoints, quests and user data."""
    conn = sqlite3.connect(seeded_db_copy)
    conn.execute("INSERT INTO demo_profiles (id, home_name, home_lat, home_lon, intent_text) "
                 "VALUES (99, 'Bamberg', 49.9, 10.9, 'old town run')")
    conn.execute("INSERT INTO profile_feedback (demo_profile_id, route_id, reason) VALUES (99, 1362328, 'too-far')")
    conn.commit()
    conn.close()
    before = counts(seeded_db_copy, "routes", "breakpoints", "mini_quests", "demo_profiles", "profile_feedback")

    run_alembic(seeded_db_copy, "downgrade", BEFORE_CATALOG)
    assert counts(seeded_db_copy, "routes", "breakpoints", "mini_quests", "demo_profiles", "profile_feedback") == before
    run_alembic(seeded_db_copy, "upgrade", CATALOG_REVISION)
    assert counts(seeded_db_copy, "routes", "breakpoints", "mini_quests", "demo_profiles", "profile_feedback") == before
    conn = sqlite3.connect(seeded_db_copy)
    assert conn.execute("SELECT home_name FROM demo_profiles WHERE id = 99").fetchone() == (None,)  # dropped
    conn.close()


def test_catalog_migration_downgrade_on_osm_seeded_db(tmp_path):
    from scripts.seed_from_catalog import load_catalog
    from tests.app.conftest import CATALOG_DIR

    db = tmp_path / "osm.db"
    run_alembic(db, "upgrade", "head")
    load_catalog(db, CATALOG_DIR, limit=40)
    before = counts(db, "routes", "breakpoints")
    run_alembic(db, "downgrade", BEFORE_CATALOG)
    assert counts(db, "routes", "breakpoints") == before
    assert not {"route_geometries", "places"} & tables(db)
