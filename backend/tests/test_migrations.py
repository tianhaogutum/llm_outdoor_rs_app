"""Alembic migrations: profile_feedback.created_at."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import run_alembic

PREVIOUS_HEAD = "e4fda692220c"


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
