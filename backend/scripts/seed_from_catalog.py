"""
Fill an empty, migrated SQLite DB from the OSM route catalogue (``backend/data/catalog``).

Reads ``routes.jsonl.gz`` and ``geometry.jsonl.gz`` (docs/DATA.md) and writes, in one transaction:

- ``routes``: one row per catalogue route (:func:`scripts.osm_catalog.route_row`);
- ``route_geometries``: the simplified line and elevation profile;
- ``breakpoints``: start, real named features in order along the route, finish
  (:func:`scripts.osm_catalog.derive_breakpoints`); ids assigned in route-id order;
- ``places``: the catalogue's localities for the home-location lookup.

Deterministic: the same catalogue files give the same rows (same ids, same values) on every run.
Use ``--limit N`` to load only the first N routes by id (tests, quick experiments).

Usage (from ``backend/``, after ``alembic upgrade head``)::

    python scripts/seed_from_catalog.py [--catalog data/catalog] [--db data/app.db] [--limit N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.osm_catalog import build_places, derive_breakpoints, route_row  # noqa: E402
from scripts.seed_io import BACKEND_DIR, connect, resolve_db_path, table_columns  # noqa: E402

DEFAULT_CATALOG_DIR = BACKEND_DIR / "data" / "catalog"
ROUTES_FILE = "routes.jsonl.gz"
GEOMETRY_FILE = "geometry.jsonl.gz"

CATALOG_TABLES = ("routes", "breakpoints", "mini_quests", "route_geometries", "places")
BREAKPOINT_COLUMNS = ("route_id", "order_index", "poi_name", "poi_type", "latitude", "longitude", "at_m")
GEOMETRY_COLUMNS = ("route_id", "coords_json", "ele_profile_json", "ele_step_m")
PLACE_COLUMNS = ("name", "name_key", "admin_district", "admin_region", "lat", "lon", "accuracy_m", "route_count")


class CatalogSeedError(RuntimeError):
    pass


@dataclass
class CatalogStats:
    routes: int = 0
    breakpoints: int = 0
    geometries: int = 0
    places: int = 0
    seconds: float = 0.0


def iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _insert_many(conn: sqlite3.Connection, table: str, columns: tuple[str, ...], rows: list[tuple]) -> None:
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})", rows
    )


def load_catalog(db_path: Path, catalog_dir: Path = DEFAULT_CATALOG_DIR, limit: int | None = None) -> CatalogStats:
    """Load the catalogue into the empty, migrated DB at ``db_path``; all or nothing."""
    started = time.perf_counter()
    if not db_path.exists():
        raise CatalogSeedError(f"{db_path} does not exist; run `alembic upgrade head` first")
    for name in (ROUTES_FILE, GEOMETRY_FILE):
        if not (catalog_dir / name).exists():
            raise CatalogSeedError(f"{catalog_dir / name} not found")

    records = sorted(iter_jsonl_gz(catalog_dir / ROUTES_FILE), key=lambda r: r["id"])
    if limit is not None:
        records = records[:limit]
    wanted = {r["id"] for r in records}
    geometries = {g["id"]: g for g in iter_jsonl_gz(catalog_dir / GEOMETRY_FILE) if g["id"] in wanted}
    missing = sorted(wanted - geometries.keys())
    if missing:
        raise CatalogSeedError(f"{len(missing)} routes have no geometry, e.g. {missing[:5]}")

    route_rows, geometry_rows, breakpoint_rows = [], [], []
    for record in records:
        geom = geometries[record["id"]]
        route_rows.append(route_row(record))
        geometry_rows.append((
            record["id"],
            json.dumps(geom["coords"], separators=(",", ":")),
            json.dumps(geom.get("ele_profile_m"), separators=(",", ":")) if geom.get("ele_profile_m") else None,
            geom.get("ele_profile_step_m"),
        ))
        for bp in derive_breakpoints(record, geom["coords"]):
            breakpoint_rows.append((record["id"], *(bp[c] for c in BREAKPOINT_COLUMNS[1:])))
    places = build_places(records)

    conn = connect(db_path)
    try:
        present = {t: set(table_columns(conn, t)) for t in CATALOG_TABLES}
        missing_tables = [t for t, cols in present.items() if not cols]
        if missing_tables:
            raise CatalogSeedError(
                f"tables missing in {db_path}: {missing_tables}; run `alembic upgrade head` first"
            )
        route_columns = tuple(route_rows[0]) if route_rows else ()
        unknown = [c for c in route_columns if c not in present["routes"]]
        if unknown:
            raise CatalogSeedError(f"routes table lacks columns {unknown}; run `alembic upgrade head` first")
        for table in CATALOG_TABLES:
            if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                raise CatalogSeedError(f"table {table} in {db_path} is not empty; seed only fills an empty DB")

        with conn:  # one transaction
            _insert_many(conn, "routes", route_columns, [tuple(r[c] for c in route_columns) for r in route_rows])
            _insert_many(conn, "route_geometries", GEOMETRY_COLUMNS, geometry_rows)
            _insert_many(conn, "breakpoints", BREAKPOINT_COLUMNS, breakpoint_rows)
            _insert_many(conn, "places", PLACE_COLUMNS, [tuple(p[c] for c in PLACE_COLUMNS) for p in places])
        conn.execute("ANALYZE")
    except sqlite3.Error as exc:
        raise CatalogSeedError(str(exc)) from exc
    finally:
        conn.close()

    return CatalogStats(
        routes=len(route_rows), breakpoints=len(breakpoint_rows), geometries=len(geometry_rows),
        places=len(places), seconds=time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill an empty, migrated DB from the OSM route catalogue.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_DIR, help="catalogue directory")
    parser.add_argument("--db", help="SQLite file (default: DATABASE_URL, else data/app.db)")
    parser.add_argument("--limit", type=int, help="load only the first N routes by id")
    args = parser.parse_args()
    db_path = resolve_db_path(args.db)
    try:
        stats = load_catalog(db_path, args.catalog, args.limit)
    except CatalogSeedError as exc:
        raise SystemExit(f"seed failed: {exc}") from exc
    print(
        f"seeded {db_path}: {stats.routes} routes, {stats.breakpoints} breakpoints, "
        f"{stats.geometries} geometries, {stats.places} places from {args.catalog} in {stats.seconds:.1f} s"
    )


if __name__ == "__main__":
    main()
