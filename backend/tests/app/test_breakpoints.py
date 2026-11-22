"""Breakpoints derived from a route's real named features (scripts/osm_catalog.py:derive_breakpoints)."""

from __future__ import annotations

import math
import sqlite3

import pytest

from app.services.place_service import name_key
from scripts.osm_catalog import (
    MAX_INTERMEDIATE_BREAKPOINTS,
    MIN_GAP_M,
    cumulative_lengths,
    derive_breakpoints,
    haversine_m,
    point_along,
)
from tests.app.conftest import CATALOG_DIR, read_jsonl_gz

# A straight 4 km line going east along 48° N (1 km ≈ 0.013436° of longitude there).
LINE = [[11.0 + i * 0.013436, 48.0] for i in range(5)]
LENGTH = round(cumulative_lengths(LINE)[-1])


def record(on_route: dict, **extra) -> dict:
    base = {"id": 1, "length_m": LENGTH, "is_loop": False, "locality": "Anfang", "locality_dist_m": 300,
            "end_locality": "Ende", "features": {"on_route": on_route}}
    return {**base, **extra}


def feat(name: str, at_m: float, **extra) -> dict:
    return {"name": name, "dist_m": 10, "at_m": at_m, **extra}


def dist_to_line_m(lat: float, lon: float, coords) -> float:
    """Distance from a point to a [[lon, lat], ...] polyline (local equirectangular approximation)."""
    kx = math.cos(math.radians(lat)) * 111_320.0
    ky = 110_540.0
    best = math.inf
    for (x1, y1), (x2, y2) in zip(coords, coords[1:], strict=False):
        ax, ay = (x1 - lon) * kx, (y1 - lat) * ky
        bx, by = (x2 - lon) * kx, (y2 - lat) * ky
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / seg2))
        best = min(best, math.hypot(ax + t * dx, ay + t * dy))
    return best


def test_start_and_finish_only_without_named_features():
    bps = derive_breakpoints(record({"restaurant": [{"name": None, "at_m": 500}]}), LINE)
    assert [b["poi_type"] for b in bps] == ["start", "finish"]
    assert bps[0]["poi_name"] == "Start: Anfang" and bps[1]["poi_name"] == "Finish: Ende"
    assert (bps[0]["latitude"], bps[0]["longitude"]) == (LINE[0][1], LINE[0][0])
    assert (bps[1]["latitude"], bps[1]["longitude"]) == (LINE[-1][1], LINE[-1][0])
    assert [b["at_m"] for b in bps] == [0, LENGTH]


def test_features_ordered_along_route_with_interpolated_coordinates():
    bps = derive_breakpoints(record({
        "church": [feat("St. Martin", 3000)],
        "lake": [feat("Badesee", 1000, area_ha=3)],
        "peak": [feat("Hoher Berg", 2000, ele_m=900)],
    }), LINE)
    assert [b["poi_name"] for b in bps] == ["Start: Anfang", "Badesee", "Hoher Berg", "St. Martin", "Finish: Ende"]
    assert [b["order_index"] for b in bps] == list(range(5))
    assert [b["poi_type"] for b in bps[1:-1]] == ["lake", "peak", "church"]
    for bp, km in zip(bps[1:-1], (1, 2, 3), strict=True):
        assert bp["latitude"] == pytest.approx(48.0, abs=1e-6)
        assert bp["longitude"] == pytest.approx(LINE[km][0], abs=2e-5)  # ≈ 1.5 m


def test_positions_are_scaled_to_the_simplified_line():
    # the catalogue length is 10% longer than the served (simplified) line: at_m is scaled down
    bps = derive_breakpoints(record({"lake": [feat("Mitte", LENGTH * 1.1 / 2)]}, length_m=LENGTH * 1.1), LINE)
    mid = bps[1]
    assert mid["longitude"] == pytest.approx((LINE[0][0] + LINE[-1][0]) / 2, abs=2e-5)
    assert mid["at_m"] == round(LENGTH * 1.1 / 2)


def test_duplicate_names_are_dropped_case_and_accent_insensitively():
    bps = derive_breakpoints(record({
        "castle": [feat("Schloss Fürstenried", 1000)],
        "museum": [feat("SCHLOSS FUERSTENRIED", 2500)],
        "lake": [feat("Weiher", 2000), feat("Weiher", 3000)],
    }), LINE)
    names = [b["poi_name"] for b in bps[1:-1]]
    assert names == ["Schloss Fürstenried", "Weiher"]
    assert len({name_key(n) for n in names}) == len(names)


def test_better_feature_wins_when_two_are_too_close():
    gap = max(MIN_GAP_M, 0.04 * LENGTH)
    bps = derive_breakpoints(record({
        "restaurant": [feat("Gasthof", 2000)],
        "viewpoint": [feat("Aussicht", 2000 + gap / 2)],
    }), LINE)
    assert [b["poi_name"] for b in bps[1:-1]] == ["Aussicht"]


def test_features_at_the_ends_do_not_duplicate_start_or_finish():
    bps = derive_breakpoints(record({"lake": [feat("Startsee", 20), feat("Zielsee", LENGTH - 20)]}), LINE)
    assert [b["poi_type"] for b in bps] == ["start", "finish"]


def test_intermediate_stops_are_capped_and_best_types_are_kept():
    many = {"church": [feat(f"Kirche {i}", 300 + i * 250) for i in range(12)], "peak": [feat("Gipfel", 3500)]}
    bps = derive_breakpoints(record(many), LINE)
    assert len(bps) == MAX_INTERMEDIATE_BREAKPOINTS + 2
    assert "Gipfel" in [b["poi_name"] for b in bps]


def test_loop_and_far_locality_labels():
    bps = derive_breakpoints(record({}, is_loop=True, locality="Dorf", locality_dist_m=5000, end_locality=None), LINE)
    assert bps[0]["poi_name"] == "Start: near Dorf"
    assert bps[-1]["poi_name"] == "Finish: near Dorf (back at the start)"


def test_point_along_clamps_and_interpolates():
    cum = cumulative_lengths(LINE)
    assert point_along(LINE, cum, -5) == (LINE[0][1], LINE[0][0])
    assert point_along(LINE, cum, 1e9) == (LINE[-1][1], LINE[-1][0])
    lat, lon = point_along(LINE, cum, cum[2])
    assert (lat, lon) == pytest.approx((LINE[2][1], LINE[2][0]))
    assert haversine_m(LINE[0][0], LINE[0][1], LINE[1][0], LINE[1][1]) == pytest.approx(1000, rel=1e-3)


def test_real_routes_invariants(catalog_subset_dir):
    """On real catalogue routes: ordered, on the served line, real names only, no duplicates."""
    records = {r["id"]: r for r in read_jsonl_gz(catalog_subset_dir / "routes.jsonl.gz")}
    lines = {g["id"]: g["coords"] for g in read_jsonl_gz(catalog_subset_dir / "geometry.jsonl.gz")}
    assert len(records) > 100
    with_stops = 0
    for rid, rec in records.items():
        coords = lines[rid]
        bps = derive_breakpoints(rec, coords)
        assert bps[0]["poi_type"] == "start" and bps[-1]["poi_type"] == "finish"
        assert [b["order_index"] for b in bps] == list(range(len(bps)))
        ats = [b["at_m"] for b in bps]
        assert ats == sorted(ats) and len(set(ats[1:-1])) == len(ats) - 2
        names = [b["poi_name"] for b in bps[1:-1]]
        assert len({name_key(n) for n in names}) == len(names)
        real = {(t, i["name"]) for t, items in rec["features"]["on_route"].items() for i in items if i.get("name")}
        assert {(b["poi_type"], b["poi_name"]) for b in bps[1:-1]} <= real
        for b in bps:
            assert dist_to_line_m(b["latitude"], b["longitude"], coords) < 2.0, (rid, b)
        with_stops += len(bps) > 2
    assert with_stops > len(records) * 0.7


def test_seeded_breakpoints_match_derivation(osm_db):
    """What the seeder stored is exactly derive_breakpoints() for that route (spot check: Eibsee)."""
    rec = next(r for r in read_jsonl_gz(CATALOG_DIR / "routes.jsonl.gz") if r["id"] == 1356555)
    coords = next(g["coords"] for g in read_jsonl_gz(CATALOG_DIR / "geometry.jsonl.gz") if g["id"] == 1356555)
    expected = derive_breakpoints(rec, coords)
    conn = sqlite3.connect(osm_db)
    conn.row_factory = sqlite3.Row
    stored = [dict(r) for r in conn.execute(
        "SELECT order_index, poi_name, poi_type, latitude, longitude, at_m FROM breakpoints "
        "WHERE route_id = 1356555 ORDER BY order_index")]
    conn.close()
    assert stored == expected
    assert [b["poi_name"] for b in stored] == [
        "Start: near Grainau", "Seilbahn Zugspitze Talstation", "Frillensee", "Wankle",
        "Finish: near Grainau (back at the start)",
    ]
