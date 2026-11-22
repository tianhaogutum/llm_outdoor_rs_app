"""Schema check over the committed catalogue (no network, no rebuild)."""

import gzip
import json
from pathlib import Path

import pytest

from pipeline.tags import TAG_NAMES

CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog"
CATEGORIES = {
    "Hiking trail",
    "Theme trail",
    "Cycling",
    "Long distance cycling",
    "Mountainbiking",
    "Jogging",
    "Trail running",
}
ACTIVITY_OF = {
    "Hiking trail": "hiking",
    "Theme trail": "hiking",
    "Cycling": "cycling",
    "Long distance cycling": "cycling",
    "Mountainbiking": "mtb",
    "Jogging": "running",
    "Trail running": "running",
}
REQUIRED = {
    "id": int,
    "source": str,
    "title": str,
    "description": str,
    "description_source": str,
    "category_name": str,
    "activity": str,
    "tags": list,
    "difficulty": int,
    "difficulty_source": str,
    "length_m": int,
    "ascent_m": int,
    "descent_m": int,
    "min_ele_m": int,
    "max_ele_m": int,
    "duration_min": int,
    "start_lat": float,
    "start_lon": float,
    "end_lat": float,
    "end_lon": float,
    "centroid_lat": float,
    "centroid_lon": float,
    "is_loop": bool,
    "features": dict,
    "osm_tags": dict,
}
# bounding box of Bavaria (47.27..50.57 N, 8.97..13.84 E) with a small margin
LAT_RANGE = (47.2, 50.6)
LON_RANGE = (8.9, 13.9)


@pytest.fixture(scope="module")
def records():
    p = CATALOG / "routes.jsonl.gz"
    if not p.exists():
        pytest.fail("committed catalogue missing: run `python -m pipeline.build`")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_size_and_sorted_unique(records):
    assert len(records) >= 3000
    ids = [r["id"] for r in records]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_required_fields_and_types(records):
    for r in records:
        for k, t in REQUIRED.items():
            assert k in r, (r["id"], k)
            v = r[k]
            if t is float:
                assert isinstance(v, (int, float)), (r["id"], k, v)
            else:
                assert isinstance(v, t) and not (t is int and isinstance(v, bool)), (r["id"], k, v)


def test_ranges(records):
    for r in records:
        rid = r["id"]
        assert r["source"] == "osm"
        assert r["category_name"] in CATEGORIES
        assert r["activity"] == ACTIVITY_OF[r["category_name"]]
        assert 0 <= r["difficulty"] <= 3
        assert r["title"].strip()
        assert len(r["description"]) >= 80
        assert r["description_source"] in ("generated", "osm+generated")
        assert 1000 <= r["length_m"] <= 800_000
        assert 0 <= r["ascent_m"] <= 30_000 and 0 <= r["descent_m"] <= 30_000
        assert -10 <= r["min_ele_m"] <= r["max_ele_m"] <= 3000, rid
        assert r["duration_min"] > 0
        for k in ("start", "end", "centroid"):
            assert LAT_RANGE[0] <= r[f"{k}_lat"] <= LAT_RANGE[1], (rid, k)
            assert LON_RANGE[0] <= r[f"{k}_lon"] <= LON_RANGE[1], (rid, k)
        assert set(r["tags"]) <= set(TAG_NAMES), rid
        assert len(r["tags"]) == len(set(r["tags"]))
        assert ("Linear route" in r["tags"]) == (not r["is_loop"])


def test_geometry_file_matches(records):
    p = CATALOG / "geometry.jsonl.gz"
    with gzip.open(p, "rt") as fh:
        geoms = [json.loads(line) for line in fh]
    assert [g["id"] for g in geoms] == [r["id"] for r in records]
    for g, r in zip(geoms[:500], records[:500], strict=True):
        assert len(g["coords"]) >= 2
        lon, lat = g["coords"][0]
        assert abs(lat - r["start_lat"]) < 1e-3 and abs(lon - r["start_lon"]) < 1e-3


def test_manifest(records):
    m = json.loads((CATALOG / "manifest.json").read_text())
    assert m["counts"]["final"] == len(records)
    assert m["inputs"]["osm"]["sha256"] and m["inputs"]["osm"]["file"].endswith(".osm.pbf")
    assert (CATALOG / "ATTRIBUTION.md").exists()
