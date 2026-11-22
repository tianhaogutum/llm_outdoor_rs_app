"""Seed file <-> database round trip."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from scripts.export_seed import export_seed
from scripts.seed_from_jsonl import SeedError, load_seed
from scripts.seed_io import SEED_TABLES
from tests.conftest import SEED_FILE, build_seeded_db, run_alembic


def table_rows(db_path: Path, table: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY id")]
    finally:
        conn.close()


def test_seed_file_shape(seed_records):
    assert len(seed_records) == 111
    ids = [r["id"] for r in seed_records]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert sum(len(r["breakpoints"]) for r in seed_records) == 15
    assert sum(len(bp["mini_quests"]) for r in seed_records for bp in r["breakpoints"]) == 9
    assert sorted({r["difficulty"] for r in seed_records}) == [0, 1, 2, 3]
    for r in seed_records:
        idx = [bp["order_index"] for bp in r["breakpoints"]]
        assert idx == sorted(idx)
    # UTF-8, not ASCII-escaped
    assert "\\u00" not in SEED_FILE.read_text(encoding="utf-8")


def test_seeded_db_counts(seeded_db):
    assert [len(table_rows(seeded_db, t)) for t in SEED_TABLES] == [111, 15, 9]


def test_export_of_seeded_db_reproduces_seed_file_byte_for_byte(seeded_db, tmp_path):
    out = tmp_path / "routes.jsonl"
    assert export_seed(seeded_db, out) == 111
    assert out.read_bytes() == SEED_FILE.read_bytes()


def test_double_round_trip_keeps_rows_identical(seeded_db, tmp_path):
    out = tmp_path / "routes.jsonl"
    export_seed(seeded_db, out)
    rebuilt = build_seeded_db(tmp_path / "rebuilt.db", out)
    for table in SEED_TABLES:
        assert table_rows(rebuilt, table) == table_rows(seeded_db, table)


@pytest.mark.skipif(not os.environ.get("SEED_REFERENCE_DB"), reason="set SEED_REFERENCE_DB to the original app.db")
def test_rebuilt_db_matches_reference_db(seeded_db):
    """Run with SEED_REFERENCE_DB=/path/to/original/app.db to check the seed against the DB it came from."""
    reference = Path(os.environ["SEED_REFERENCE_DB"])
    for table in SEED_TABLES:
        assert table_rows(seeded_db, table) == table_rows(reference, table), table


def test_loader_skips_unknown_keys_and_assigns_missing_ids(tmp_path):
    record = {
        "id": 42,
        "title": "Extra-rich route",
        "difficulty": 2,
        "osm_relation_id": 123,  # unknown
        "geometry": {"type": "LineString"},  # unknown
        "breakpoints": [
            {
                "order_index": 0,
                "poi_name": "Start",
                "elevation_m": 512,  # unknown, no id
                "mini_quests": [{"task_description": "Look around", "xp_reward": 5, "lang": "en"}],
            },
            {"order_index": 1, "poi_name": "End", "elevation_m": 600, "mini_quests": []},
        ],
    }
    seed = tmp_path / "rich.jsonl"
    seed.write_text(json.dumps(record) + "\n\n", encoding="utf-8")
    db = tmp_path / "rich.db"
    run_alembic(db, "upgrade", "head")
    stats = load_seed(db, seed)

    assert (stats.routes, stats.breakpoints, stats.mini_quests) == (1, 2, 1)
    assert dict(stats.unknown_keys) == {
        "routes.osm_relation_id": 1,
        "routes.geometry": 1,
        "breakpoints.elevation_m": 2,
        "mini_quests.lang": 1,
    }
    assert stats.unknown_total == 5
    bps = table_rows(db, "breakpoints")
    assert [(b["route_id"], b["order_index"]) for b in bps] == [(42, 0), (42, 1)]
    quests = table_rows(db, "mini_quests")
    assert quests[0]["breakpoint_id"] == bps[0]["id"]


def test_loader_refuses_non_empty_db(seeded_db_copy):
    with pytest.raises(SeedError, match="not empty"):
        load_seed(seeded_db_copy, SEED_FILE)


def test_loader_refuses_unmigrated_db(tmp_path):
    db = tmp_path / "blank.db"
    sqlite3.connect(db).close()
    with pytest.raises(SeedError, match="alembic upgrade head"):
        load_seed(db, SEED_FILE)


def test_loader_is_all_or_nothing(tmp_path):
    seed = tmp_path / "bad.jsonl"
    seed.write_text(json.dumps({"id": 1, "title": "ok"}) + "\n" + json.dumps({"id": 2}) + "\n", encoding="utf-8")
    db = tmp_path / "bad.db"
    run_alembic(db, "upgrade", "head")
    with pytest.raises(SeedError, match="bad.jsonl:2"):
        load_seed(db, seed)
    assert table_rows(db, "routes") == []
