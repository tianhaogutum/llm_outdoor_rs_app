"""
Export all routes (with breakpoints and mini quests) from a SQLite DB to the JSONL seed file.

Deterministic: routes by id, breakpoints by order_index (then id), quests by id; keys in model column
order; UTF-8 without ASCII escaping. Catalogue-era columns are written only when set (see
``EXPORT_IF_SET_COLUMNS``). Route geometries and places are not exported (they are rebuilt from the
OSM catalogue by ``scripts/seed_from_catalog.py``). User data is not exported.

Usage (from ``backend/``)::

    python scripts/export_seed.py [--db data/app.db] [--out data/seed/routes.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seed_io import (  # noqa: E402
    BREAKPOINT_PARENT_COLUMN,
    BREAKPOINT_TABLE,
    BREAKPOINTS_KEY,
    DEFAULT_SEED_FILE,
    EXPORT_IF_SET_COLUMNS,
    MINI_QUEST_PARENT_COLUMN,
    MINI_QUEST_TABLE,
    MINI_QUESTS_KEY,
    ROUTE_TABLE,
    connect,
    model_columns,
    resolve_db_path,
    table_columns,
)


def _columns(conn: sqlite3.Connection, table: str, exclude: str | None = None) -> list[str]:
    """Model column order, restricted to columns the DB actually has."""
    present = set(table_columns(conn, table))
    return [c for c in model_columns(table) if c in present and c != exclude]


def _row_dict(row: sqlite3.Row, columns: list[str], table: str) -> dict:
    """``columns`` of ``row``; catalogue-era columns (EXPORT_IF_SET_COLUMNS) only when not NULL."""
    optional = EXPORT_IF_SET_COLUMNS[table]
    return {c: row[c] for c in columns if c not in optional or row[c] is not None}


def iter_route_records(conn: sqlite3.Connection) -> Iterator[dict]:
    route_cols = _columns(conn, ROUTE_TABLE)
    bp_cols = _columns(conn, BREAKPOINT_TABLE, exclude=BREAKPOINT_PARENT_COLUMN)
    quest_cols = _columns(conn, MINI_QUEST_TABLE, exclude=MINI_QUEST_PARENT_COLUMN)

    quests_by_bp: dict[int, list[dict]] = {}
    for row in conn.execute(
        f"SELECT {MINI_QUEST_PARENT_COLUMN}, {', '.join(quest_cols)} FROM {MINI_QUEST_TABLE} ORDER BY id"
    ):
        quests_by_bp.setdefault(row[0], []).append(_row_dict(row, quest_cols, MINI_QUEST_TABLE))

    bps_by_route: dict[int, list[dict]] = {}
    for row in conn.execute(
        f"SELECT {BREAKPOINT_PARENT_COLUMN}, {', '.join(bp_cols)} FROM {BREAKPOINT_TABLE} ORDER BY order_index, id"
    ):
        bp = _row_dict(row, bp_cols, BREAKPOINT_TABLE)
        bp[MINI_QUESTS_KEY] = quests_by_bp.get(row["id"], [])
        bps_by_route.setdefault(row[0], []).append(bp)

    for row in conn.execute(f"SELECT {', '.join(route_cols)} FROM {ROUTE_TABLE} ORDER BY id"):
        record = _row_dict(row, route_cols, ROUTE_TABLE)
        record[BREAKPOINTS_KEY] = bps_by_route.get(row["id"], [])
        yield record


def dumps_record(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(", ", ": "))


def export_seed(db_path: Path, out_path: Path) -> int:
    """Write the seed file; returns the number of routes written."""
    conn = connect(db_path)
    try:
        lines = [dumps_record(r) for r in iter_route_records(conn)]
    finally:
        conn.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export routes to the JSONL seed file.")
    parser.add_argument("--db", help="SQLite file (default: DATABASE_URL, else data/app.db)")
    parser.add_argument("--out", type=Path, default=DEFAULT_SEED_FILE)
    args = parser.parse_args()
    db_path = resolve_db_path(args.db)
    n = export_seed(db_path, args.out)
    print(f"exported {n} routes from {db_path} to {args.out}")


if __name__ == "__main__":
    main()
