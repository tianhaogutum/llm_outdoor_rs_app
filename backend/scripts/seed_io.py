"""
Shared helpers for the route seed file (``backend/data/seed/routes.jsonl``).

Seed format: one route per line, JSON object with the ``routes`` columns (model column order), then
``breakpoints``: a list ordered by ``order_index``, each with the ``breakpoints`` columns except
``route_id`` and a nested ``mini_quests`` list ordered by id (``mini_quests`` columns except
``breakpoint_id``). Foreign keys are implied by the nesting.

User data (profiles, souvenirs, feedback, achievements unlocked) is not seed data. Achievement
definitions are seeded by the app on startup.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from sqlalchemy.engine import make_url

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SEED_FILE = BACKEND_DIR / "data" / "seed" / "routes.jsonl"
DEFAULT_DB_FILE = BACKEND_DIR / "data" / "app.db"

ROUTE_TABLE = "routes"
BREAKPOINT_TABLE = "breakpoints"
MINI_QUEST_TABLE = "mini_quests"
SEED_TABLES = (ROUTE_TABLE, BREAKPOINT_TABLE, MINI_QUEST_TABLE)

# Keys holding nested children, and the foreign-key column each child level gets from its parent.
BREAKPOINTS_KEY = "breakpoints"
MINI_QUESTS_KEY = "mini_quests"
BREAKPOINT_PARENT_COLUMN = "route_id"
MINI_QUEST_PARENT_COLUMN = "breakpoint_id"

# Columns added for the OSM catalogue (migration 9b4e2d7f1a36). The legacy seed format predates them,
# so the exporter writes them only when they hold a value: a DB seeded from the legacy file still
# exports to that file byte for byte, and an OSM-seeded DB exports its catalogue facts.
EXPORT_IF_SET_COLUMNS = {
    ROUTE_TABLE: frozenset({
        "source", "source_url", "activity", "start_lat", "start_lon", "is_loop", "descent_m",
        "locality", "admin_district", "admin_region", "facts_json",
    }),
    BREAKPOINT_TABLE: frozenset({"at_m"}),
    MINI_QUEST_TABLE: frozenset(),
}


def model_columns(table: str) -> list[str]:
    """Column names of ``table`` in the order the SQLAlchemy model declares them."""
    from app.models.entities import Breakpoint, MiniQuest, Route

    models = {ROUTE_TABLE: Route, BREAKPOINT_TABLE: Breakpoint, MINI_QUEST_TABLE: MiniQuest}
    return [c.name for c in models[table].__table__.columns]


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Column names of ``table`` as they exist in the database."""
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def resolve_db_path(db: str | os.PathLike | None) -> Path:
    """
    The SQLite file to use: ``db`` if given, else the one named by ``DATABASE_URL``
    (env or ``backend/.env``, as the app reads it), else ``backend/data/app.db``.
    """
    if db:
        return Path(db).resolve()
    from app.settings import get_settings

    url = make_url(get_settings().database_url)
    if not url.drivername.startswith("sqlite") or not url.database:
        raise SystemExit(f"DATABASE_URL is not a SQLite file URL: {url.drivername}; pass --db")
    return Path(url.database).resolve()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
