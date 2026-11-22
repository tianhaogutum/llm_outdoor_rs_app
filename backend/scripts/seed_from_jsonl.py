"""
Fill an empty, migrated SQLite DB with routes from a JSONL seed file.

Each line is one route (see ``scripts/seed_io.py`` for the format). Keys that are not columns of the
target table are skipped and counted, not fatal: richer catalogue files carry extra fields. Breakpoints
or quests without an ``id`` get one from the database.

Usage (from ``backend/``, after ``alembic upgrade head``)::

    python scripts/seed_from_jsonl.py [--file data/seed/routes.jsonl] [--db data/app.db]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seed_io import (  # noqa: E402
    BREAKPOINT_PARENT_COLUMN,
    BREAKPOINT_TABLE,
    BREAKPOINTS_KEY,
    DEFAULT_SEED_FILE,
    MINI_QUEST_PARENT_COLUMN,
    MINI_QUEST_TABLE,
    MINI_QUESTS_KEY,
    ROUTE_TABLE,
    SEED_TABLES,
    connect,
    resolve_db_path,
    table_columns,
)


class SeedError(RuntimeError):
    pass


@dataclass
class LoadStats:
    routes: int = 0
    breakpoints: int = 0
    mini_quests: int = 0
    unknown_keys: Counter = field(default_factory=Counter)  # "table.key" -> occurrences

    @property
    def unknown_total(self) -> int:
        return sum(self.unknown_keys.values())


def _insert(conn: sqlite3.Connection, table: str, columns: set[str], record: dict, stats: LoadStats,
            nested: tuple[str, ...] = (), parent: tuple[str, object] | None = None) -> int:
    values = {}
    for key, value in record.items():
        if key in nested:
            continue
        if key in columns:
            values[key] = value
        else:
            stats.unknown_keys[f"{table}.{key}"] += 1
    if parent is not None:
        values[parent[0]] = parent[1]
    if not values:
        raise SeedError(f"{table}: record has no known columns")
    cols = list(values)
    cur = conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [values[c] for c in cols],
    )
    return values["id"] if values.get("id") is not None else cur.lastrowid


def load_seed(db_path: Path, seed_file: Path) -> LoadStats:
    """Load ``seed_file`` into the empty, migrated DB at ``db_path`` in one transaction."""
    if not db_path.exists():
        raise SeedError(f"{db_path} does not exist; run `alembic upgrade head` first")
    conn = connect(db_path)
    stats = LoadStats()
    try:
        columns = {t: set(table_columns(conn, t)) for t in SEED_TABLES}
        missing = [t for t, cols in columns.items() if not cols]
        if missing:
            raise SeedError(f"tables missing in {db_path}: {missing}; run `alembic upgrade head` first")
        for table in SEED_TABLES:
            if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                raise SeedError(f"table {table} in {db_path} is not empty; seed only fills an empty DB")

        with conn:  # one transaction: all or nothing
            with seed_file.open(encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        route = json.loads(line)
                        route_id = _insert(conn, ROUTE_TABLE, columns[ROUTE_TABLE], route, stats,
                                           nested=(BREAKPOINTS_KEY,))
                        stats.routes += 1
                        for bp in route.get(BREAKPOINTS_KEY) or []:
                            bp_id = _insert(conn, BREAKPOINT_TABLE, columns[BREAKPOINT_TABLE], bp, stats,
                                            nested=(MINI_QUESTS_KEY,), parent=(BREAKPOINT_PARENT_COLUMN, route_id))
                            stats.breakpoints += 1
                            for quest in bp.get(MINI_QUESTS_KEY) or []:
                                _insert(conn, MINI_QUEST_TABLE, columns[MINI_QUEST_TABLE], quest, stats,
                                        parent=(MINI_QUEST_PARENT_COLUMN, bp_id))
                                stats.mini_quests += 1
                    except (json.JSONDecodeError, sqlite3.Error, SeedError) as exc:
                        raise SeedError(f"{seed_file}:{line_no}: {exc}") from exc
    finally:
        conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill an empty, migrated DB from a JSONL route file.")
    parser.add_argument("--file", type=Path, default=DEFAULT_SEED_FILE, help="JSONL seed file")
    parser.add_argument("--db", help="SQLite file (default: DATABASE_URL, else data/app.db)")
    args = parser.parse_args()
    db_path = resolve_db_path(args.db)
    try:
        stats = load_seed(db_path, args.file)
    except SeedError as exc:
        raise SystemExit(f"seed failed: {exc}") from exc
    print(
        f"seeded {db_path}: {stats.routes} routes, {stats.breakpoints} breakpoints, "
        f"{stats.mini_quests} mini_quests from {args.file}"
    )
    if stats.unknown_keys:
        print(f"warning: skipped {stats.unknown_total} values of keys that are not DB columns:", file=sys.stderr)
        for key, count in sorted(stats.unknown_keys.items()):
            print(f"  {key}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
