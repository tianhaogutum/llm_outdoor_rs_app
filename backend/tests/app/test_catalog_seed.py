"""Seeding the app DB from the OSM catalogue: counts, field spot checks, determinism, guards."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.export_seed import export_seed
from scripts.osm_catalog import build_places, display_facts, route_row
from scripts.seed_from_catalog import CATALOG_TABLES, CatalogSeedError, load_catalog
from scripts.seed_from_jsonl import load_seed
from tests.app.conftest import build_osm_db, read_jsonl_gz
from tests.conftest import run_alembic


def rows(db: Path, sql: str, *params) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def scalar(db: Path, sql: str, *params):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


def content_digest(db: Path) -> str:
    """Hash of every row of the seeded tables in primary-key order."""
    h = hashlib.sha256()
    conn = sqlite3.connect(db)
    try:
        for table in CATALOG_TABLES:
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1"):
                h.update(repr(row).encode())
    finally:
        conn.close()
    return h.hexdigest()


def test_full_catalogue_counts(full_osm_db, catalog_records):
    n = len(catalog_records)
    assert n == 9347
    assert scalar(full_osm_db, "SELECT count(*) FROM routes") == n
    assert scalar(full_osm_db, "SELECT count(*) FROM route_geometries") == n
    assert scalar(full_osm_db, "SELECT count(*) FROM routes WHERE source = 'osm'") == n
    assert scalar(full_osm_db, "SELECT count(*) FROM places") == len(build_places(catalog_records))
    # every route has start and finish, at most 6 stops in between
    per_route = rows(full_osm_db, "SELECT route_id, count(*) AS n FROM breakpoints GROUP BY route_id")
    assert len(per_route) == n
    assert min(r["n"] for r in per_route) >= 2 and max(r["n"] for r in per_route) <= 8
    # the three category tabs are all populated
    by_cat = {r["category_name"]: r["n"]
              for r in rows(full_osm_db, "SELECT category_name, count(*) AS n FROM routes GROUP BY 1")}
    assert by_cat["Hiking trail"] + by_cat["Theme trail"] == 7648
    assert by_cat["Jogging"] + by_cat["Trail running"] == 55
    assert by_cat["Cycling"] + by_cat["Long distance cycling"] + by_cat["Mountainbiking"] == 1644


def test_spot_checks_against_docs(osm_db):
    """Values quoted in docs/DATA.md (spot-check table) arrive unchanged in the app DB."""
    eibsee = rows(osm_db, "SELECT * FROM routes WHERE id = 1356555")[0]
    assert eibsee["title"] == "Eibsee Rundwanderweg"
    assert eibsee["category_name"] == "Hiking trail" and eibsee["activity"] == "hiking"
    assert eibsee["length_meters"] == 6962.0
    assert (eibsee["elevation"], eibsee["descent_m"]) == (113, 116)  # elevation holds total ascent
    assert eibsee["is_loop"] == 1
    assert (eibsee["locality"], eibsee["admin_district"], eibsee["admin_region"]) == (
        "Grainau", "Landkreis Garmisch-Partenkirchen", "Oberbayern")
    assert eibsee["location"] == "Grainau, Landkreis Garmisch-Partenkirchen"
    assert eibsee["source"] == "osm"
    assert eibsee["source_url"] == "https://www.openstreetmap.org/relation/1356555"
    assert eibsee["start_lat"] is not None and eibsee["start_lon"] is not None
    assert "Round trip" in json.loads(eibsee["tags_json"])
    assert eibsee["xp_required"] == 0 and eibsee["base_xp_reward"] > 0
    facts = json.loads(eibsee["facts_json"])
    assert {h["name"] for h in facts["highlights"]} >= {"Eibsee", "Frillensee"}
    assert facts["access"]["train_station"]["name"] == "Eibsee"
    assert (facts["min_ele_m"], facts["max_ele_m"]) == (983, 1037)

    steig = rows(osm_db, "SELECT * FROM routes WHERE id = 3201006")[0]
    assert (steig["elevation"], steig["descent_m"], steig["is_loop"], steig["difficulty"]) == (0, 1532, 0, 3)
    ammersee = rows(osm_db, "SELECT * FROM routes WHERE id = 551290")[0]
    assert (ammersee["category_name"], ammersee["elevation"], ammersee["descent_m"]) == ("Cycling", 289, 282)


def test_route_row_matches_catalogue_record(catalog_records):
    record = next(r for r in catalog_records if r["id"] == 26937)
    row = route_row(record)
    assert row["short_description"] == record["description"]
    assert row["difficulty"] == record["difficulty"] and row["duration_min"] == record["duration_min"]
    assert json.loads(row["tags_json"]) == record["tags"]
    assert json.loads(row["facts_json"]) == display_facts(record)


def test_display_facts_are_compact(catalog_records):
    """facts_json is bounded and about half the size of the raw ``features`` blob it replaces."""
    stored = [len(route_row(r)["facts_json"].encode()) for r in catalog_records[:1000]]
    raw = [len(json.dumps(r["features"], ensure_ascii=False, separators=(",", ":")).encode())
           for r in catalog_records[:1000]]
    ratios = sorted(a / b for a, b in zip(stored, raw, strict=True))
    assert ratios[len(ratios) // 2] < 0.6
    assert max(stored) < 1600
    for r in catalog_records[:1000]:
        facts = display_facts(r)
        assert None not in facts.values() and [] not in facts.values() and {} not in facts.values()
        assert len(facts.get("highlights", [])) <= 8


def test_seed_is_deterministic(tmp_path, catalog_subset_dir, osm_db):
    again = build_osm_db(tmp_path / "again.db", catalog_subset_dir)
    assert content_digest(again) == content_digest(osm_db)


def test_seed_refuses_non_empty_db(osm_db_copy, catalog_subset_dir):
    with pytest.raises(CatalogSeedError, match="not empty"):
        load_catalog(osm_db_copy, catalog_subset_dir)


def test_seed_refuses_unmigrated_db(tmp_path, catalog_subset_dir):
    db = tmp_path / "blank.db"
    sqlite3.connect(db).close()
    with pytest.raises(CatalogSeedError, match="alembic upgrade head"):
        load_catalog(db, catalog_subset_dir)


def test_seed_limit_loads_first_routes_by_id(tmp_path, catalog_subset_dir):
    db = tmp_path / "limited.db"
    run_alembic(db, "upgrade", "head")
    stats = load_catalog(db, catalog_subset_dir, limit=5)
    ids = [r["id"] for r in read_jsonl_gz(catalog_subset_dir / "routes.jsonl.gz")]
    assert stats.routes == 5
    assert [r["id"] for r in rows(db, "SELECT id FROM routes ORDER BY id")] == sorted(ids)[:5]


def test_osm_db_exports_and_reloads_through_the_seed_file_format(osm_db, tmp_path):
    """The JSONL seed format carries the catalogue columns too (geometry and places are rebuilt)."""
    out = tmp_path / "osm.jsonl"
    n = export_seed(osm_db, out)
    assert n == scalar(osm_db, "SELECT count(*) FROM routes")
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert first["source"] == "osm" and "facts_json" in first and first["breakpoints"][0]["at_m"] == 0
    rebuilt = tmp_path / "rebuilt.db"
    run_alembic(rebuilt, "upgrade", "head")
    load_seed(rebuilt, out)
    for table in ("routes", "breakpoints"):
        assert rows(rebuilt, f"SELECT * FROM {table} ORDER BY id") == rows(osm_db, f"SELECT * FROM {table} ORDER BY id")
