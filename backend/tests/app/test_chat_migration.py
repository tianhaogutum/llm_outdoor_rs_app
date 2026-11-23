"""Migration a7c3e9d2b5f4: demo_profiles.activity and chat_state_json (nullable), reversible."""

from __future__ import annotations

import sqlite3

from tests.conftest import run_alembic

PREVIOUS = "9b4e2d7f1a36"


def profile_columns(db) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(demo_profiles)")}
    finally:
        conn.close()


def test_chat_columns_upgrade_and_downgrade(tmp_path):
    db = tmp_path / "m.db"
    run_alembic(db, "upgrade", PREVIOUS)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO demo_profiles (id, intent_text) VALUES (1, 'lake')")
    conn.commit()
    conn.close()
    run_alembic(db, "upgrade", "head")
    assert {"activity", "chat_state_json"} <= profile_columns(db)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT intent_text, activity, chat_state_json FROM demo_profiles").fetchone() == (
        "lake", None, None)
    conn.close()
    run_alembic(db, "downgrade", PREVIOUS)
    assert not {"activity", "chat_state_json"} & profile_columns(db)
