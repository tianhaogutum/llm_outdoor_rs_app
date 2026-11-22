"""
Pure transforms from the OSM route catalogue (``backend/data/catalog``, docs/DATA.md) to app rows.

No I/O and no database here: ``scripts/seed_from_catalog.py`` reads the files and writes the rows,
tests call these functions directly.

- :func:`route_row` maps a catalogue record onto the ``routes`` columns.
- :func:`display_facts` builds the compact ``facts_json`` the UI shows (highlights, access, surface).
- :func:`derive_breakpoints` turns the route's real named features into ordered breakpoints.
- :func:`build_places` aggregates the catalogue's localities for the home-location lookup.

Breakpoints are materialised at seed time rather than lazily on first use: 9,347 routes give about
60k rows (a few MB, well under a second to insert), every existing code path that reads
``route.breakpoints`` (recommendation payloads, story generation, the simulator) keeps working
unchanged, the result is deterministic per seed, and no GET request has to write to the database.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Optional

from app.services.place_service import name_key
from scripts.calculate_route_xp import calculate_base_xp_reward

OSM_RELATION_URL = "https://www.openstreetmap.org/relation/{id}"

# Feature types in order of interest for a walker: a breakpoint is a real named place on the route,
# and when a route passes many, the more distinctive ones win. Every type of features.on_route is
# listed; unknown future types sort last.
FEATURE_PRIORITY: tuple[str, ...] = (
    "peak", "alpine_hut", "wilderness_hut", "waterfall", "gorge", "glacier", "lake", "viewpoint",
    "castle", "ruins", "monastery", "cave", "saddle", "geological", "river", "museum",
    "archaeological_site", "natural_monument_tree", "beer_garden", "swimming", "aerialway_station",
    "zoo", "bird_hide", "monument", "attraction", "spring", "picnic_site", "shelter", "playground",
    "church", "restaurant", "campsite", "drinking_water",
)
_PRIORITY = {t: i for i, t in enumerate(FEATURE_PRIORITY)}

MAX_INTERMEDIATE_BREAKPOINTS = 6  # plus start and finish: the simulator map has layouts up to 10
MIN_GAP_M = 150.0  # breakpoints closer than this along the route are merged (the better one is kept)
MIN_GAP_SHARE = 0.04  # ... or closer than this share of the route length
MAX_HIGHLIGHTS = 8  # named highlights kept in facts_json
NEAR_LOCALITY_M = 2000  # farther than this from the locality node, the start is "near" it


# --------------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def cumulative_lengths(coords: Sequence[Sequence[float]]) -> list[float]:
    """Distance from the first vertex to each vertex of a ``[[lon, lat], ...]`` line, in metres."""
    out = [0.0]
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:], strict=False):
        out.append(out[-1] + haversine_m(lon1, lat1, lon2, lat2))
    return out


def point_along(coords: Sequence[Sequence[float]], cum: Sequence[float], dist_m: float) -> tuple[float, float]:
    """``(lat, lon)`` of the point ``dist_m`` metres along the line (clamped to its ends)."""
    if not coords:
        raise ValueError("empty line")
    if dist_m <= 0 or len(coords) == 1:
        lon, lat = coords[0]
        return lat, lon
    if dist_m >= cum[-1]:
        lon, lat = coords[-1]
        return lat, lon
    i = bisect.bisect_right(cum, dist_m) - 1  # segment i .. i+1
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 0 else (dist_m - cum[i]) / seg
    lon = coords[i][0] + t * (coords[i + 1][0] - coords[i][0])
    lat = coords[i][1] + t * (coords[i + 1][1] - coords[i][1])
    return lat, lon


# --------------------------------------------------------------------------------------------------
# breakpoints
# --------------------------------------------------------------------------------------------------


def _feature_candidates(record: Mapping[str, Any]) -> list[dict]:
    """Named on-route features with a position, best first (priority, then position, then name)."""
    on_route = (record.get("features") or {}).get("on_route") or {}
    out = []
    for ftype, items in on_route.items():
        for item in items or []:
            name = (item.get("name") or "").strip()
            at = item.get("at_m")
            if not name or at is None:
                continue
            out.append({"type": ftype, "name": name, "at_m": float(at)})
    out.sort(key=lambda c: (_PRIORITY.get(c["type"], len(_PRIORITY)), c["at_m"], c["name"]))
    return out


def _place_label(locality: Optional[str], dist_m: Optional[int]) -> Optional[str]:
    if not locality:
        return None
    if dist_m is not None and dist_m > NEAR_LOCALITY_M:
        return f"near {locality}"
    return locality


def derive_breakpoints(
    record: Mapping[str, Any],
    coords: Sequence[Sequence[float]],
    max_intermediate: int = MAX_INTERMEDIATE_BREAKPOINTS,
) -> list[dict]:
    """
    Ordered breakpoints for an OSM route: start, up to ``max_intermediate`` real named features in
    order along the route, finish.

    Rules:

    * Intermediate stops come only from ``features.on_route`` (named OSM objects within the type's
      distance of the line); nothing is invented. Candidates are taken best type first
      (:data:`FEATURE_PRIORITY`), skipping a name already chosen (case/diacritics-insensitive) and
      anything closer than ``max(MIN_GAP_M, MIN_GAP_SHARE * length)`` along the route to the start,
      the finish or an already chosen stop. The chosen stops are then ordered by ``at_m``.
    * ``at_m`` is measured on the pipeline's full-resolution line; the committed geometry is
      simplified (Douglas-Peucker 10 m) and slightly shorter, so positions are scaled by the ratio of
      the two lengths before interpolating. Coordinates therefore lie on the served line.
    * Start and finish are the line's first and last vertex, named after the catalogue's
      ``locality`` / ``end_locality`` (the nearest OSM place node).

    Returns dicts with the ``breakpoints`` columns (``order_index``, ``poi_name``, ``poi_type``,
    ``latitude``, ``longitude``, ``at_m``); ``main_quest_snippet`` is left for the story generator.
    """
    if not coords:
        raise ValueError(f"route {record.get('id')}: no geometry")
    cum = cumulative_lengths(coords)
    length = float(record.get("length_m") or cum[-1] or 0.0)
    scale = (cum[-1] / length) if length > 0 else 1.0
    min_gap = max(MIN_GAP_M, MIN_GAP_SHARE * length)

    chosen: list[dict] = []
    seen_names: set[str] = set()
    for cand in _feature_candidates(record):
        if len(chosen) >= max_intermediate:
            break
        at = min(max(cand["at_m"], 0.0), length)
        key = name_key(cand["name"])
        if key in seen_names:
            continue
        if at < min_gap or length - at < min_gap:
            continue
        if any(abs(at - c["at_m"]) < min_gap for c in chosen):
            continue
        seen_names.add(key)
        chosen.append({**cand, "at_m": at})
    chosen.sort(key=lambda c: (c["at_m"], c["name"]))

    start_label = _place_label(record.get("locality"), record.get("locality_dist_m"))
    if record.get("is_loop"):
        finish_label = f"{start_label} (back at the start)" if start_label else "Back at the start"
    else:
        finish_label = record.get("end_locality") or start_label

    s_lon, s_lat = coords[0]
    e_lon, e_lat = coords[-1]
    rows = [{
        "poi_name": f"Start: {start_label}" if start_label else "Start",
        "poi_type": "start",
        "latitude": s_lat,
        "longitude": s_lon,
        "at_m": 0,
    }]
    for c in chosen:
        lat, lon = point_along(coords, cum, c["at_m"] * scale)
        rows.append({
            "poi_name": c["name"],
            "poi_type": c["type"],
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "at_m": int(round(c["at_m"])),
        })
    rows.append({
        "poi_name": f"Finish: {finish_label}" if finish_label else "Finish",
        "poi_type": "finish",
        "latitude": e_lat,
        "longitude": e_lon,
        "at_m": int(round(length)),
    })
    for i, row in enumerate(rows):
        row["order_index"] = i
    return rows


# --------------------------------------------------------------------------------------------------
# display facts and route row
# --------------------------------------------------------------------------------------------------


def _nearest(items: Iterable[Mapping[str, Any]] | None, keys: tuple[str, ...]) -> Optional[dict]:
    best = None
    for item in items or []:
        if item.get("dist_m") is None:
            continue
        if best is None or item["dist_m"] < best["dist_m"]:
            best = item
    if best is None:
        return None
    return {k: best[k] for k in keys if best.get(k) is not None}


def highlights(record: Mapping[str, Any], limit: int = MAX_HIGHLIGHTS) -> list[dict]:
    """Distinct named on-route features, best type first: ``[{type, name, at_m}]``."""
    out, seen = [], set()
    for cand in _feature_candidates(record):
        key = name_key(cand["name"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": cand["type"], "name": cand["name"], "at_m": int(round(cand["at_m"]))})
        if len(out) >= limit:
            break
    return out


def display_facts(record: Mapping[str, Any]) -> dict:
    """
    The compact facts the UI shows, a few hundred bytes instead of the ~3 KB ``features`` blob.
    Keys with no value are omitted. All values are copied or rounded from the record.
    """
    features = record.get("features") or {}
    near_start = features.get("near_start") or {}
    near_end = features.get("near_end") or {}
    profile = features.get("profile") or {}
    tags = record.get("osm_tags") or {}

    access = {
        "train_station": _nearest(near_start.get("train_station"), ("name", "dist_m")),
        "tram_stop": _nearest(near_start.get("tram_stop"), ("name", "dist_m")),
        "bus_stop": _nearest(near_start.get("bus_stop"), ("name", "dist_m")),
        "parking": _nearest(near_start.get("parking"), ("name", "dist_m", "fee", "capacity")),
    }
    end_access = {
        "train_station": _nearest(near_end.get("train_station"), ("name", "dist_m")),
        "tram_stop": _nearest(near_end.get("tram_stop"), ("name", "dist_m")),
        "bus_stop": _nearest(near_end.get("bus_stop"), ("name", "dist_m")),
    }
    surface = {k: round(profile[k], 2) for k in ("paved_share", "unpaved_share", "path_share", "track_share",
                                                 "road_share") if profile.get(k) is not None}
    facts = {
        "highlights": highlights(record),
        "access": {k: v for k, v in access.items() if v},
        "end_access": {k: v for k, v in end_access.items() if v},
        "surface": surface,
        "min_ele_m": record.get("min_ele_m"),
        "max_ele_m": record.get("max_ele_m"),
        "max_grade": record.get("max_grade"),
        "est_days": record.get("est_days") if (record.get("est_days") or 0) > 1 else None,
        "end_locality": record.get("end_locality"),
        "protected_areas": [p["name"] for p in features.get("protected_areas") or [] if p.get("name")],
        "description_source": record.get("description_source"),
        "ref": tags.get("ref"),
        "network": tags.get("network"),
        "operator": tags.get("operator"),
        "website": tags.get("website") or tags.get("url"),
    }
    return {k: v for k, v in facts.items() if v not in (None, [], {}, "")}


def _base_xp_reward(difficulty: Optional[int], length_m: Optional[float], duration_min: Optional[int],
                    ascent_m: Optional[int]) -> int:
    """Same formula as the legacy routes' ``base_xp_reward`` (scripts/calculate_route_xp.py)."""
    return calculate_base_xp_reward(SimpleNamespace(
        difficulty=difficulty, length_meters=length_m, duration_min=duration_min, elevation=ascent_m,
    ))


def route_row(record: Mapping[str, Any]) -> dict:
    """``routes`` columns for one catalogue record (storytelling columns stay NULL)."""
    locality, district = record.get("locality"), record.get("admin_district")
    location = ", ".join(p for p in (locality, district) if p) or None
    tags = record.get("tags") or []
    return {
        "id": record["id"],
        "title": record["title"],
        "category_name": record.get("category_name"),
        "length_meters": float(record["length_m"]) if record.get("length_m") is not None else None,
        "duration_min": record.get("duration_min"),
        "difficulty": record.get("difficulty"),
        "short_description": record.get("description"),
        "tags_json": json.dumps(tags, ensure_ascii=False),
        "location": location,
        "elevation": record.get("ascent_m"),  # elevation == total ascent (see Route docstring)
        "gpx_data_raw": None,
        "xp_required": 0,
        "base_xp_reward": _base_xp_reward(record.get("difficulty"), record.get("length_m"),
                                          record.get("duration_min"), record.get("ascent_m")),
        "source": record.get("source") or "osm",
        "source_url": OSM_RELATION_URL.format(id=record["id"]),
        "activity": record.get("activity"),
        "start_lat": record.get("start_lat"),
        "start_lon": record.get("start_lon"),
        "is_loop": record.get("is_loop"),
        "descent_m": record.get("descent_m"),
        "locality": locality,
        "admin_district": district,
        "admin_region": record.get("admin_region"),
        "facts_json": json.dumps(display_facts(record), ensure_ascii=False, separators=(",", ":")),
    }


# --------------------------------------------------------------------------------------------------
# places
# --------------------------------------------------------------------------------------------------


def build_places(records: Iterable[Mapping[str, Any]]) -> list[dict]:
    """
    One place per (locality, district) named as a route start's locality.

    Coordinates: the start of the route that lies closest to the locality's place node (smallest
    ``locality_dist_m``, ties by lower id); ``accuracy_m`` is that distance. Sorted by name key,
    district, name, so the table content is deterministic.
    """
    best: dict[tuple[str, Optional[str]], dict] = {}
    for r in records:
        name = r.get("locality")
        if not name or r.get("start_lat") is None or r.get("start_lon") is None:
            continue
        key = (name, r.get("admin_district"))
        dist = r.get("locality_dist_m")
        rank = (dist if dist is not None else math.inf, r["id"])
        cur = best.get(key)
        if cur is None:
            best[key] = {"rank": rank, "record": r, "count": 1}
        else:
            cur["count"] += 1
            if rank < cur["rank"]:
                cur["rank"], cur["record"] = rank, r
    places = []
    for (name, district), entry in best.items():
        r = entry["record"]
        places.append({
            "name": name,
            "name_key": name_key(name),
            "admin_district": district,
            "admin_region": r.get("admin_region"),
            "lat": r["start_lat"],
            "lon": r["start_lon"],
            "accuracy_m": r.get("locality_dist_m"),
            "route_count": entry["count"],
        })
    places.sort(key=lambda p: (p["name_key"], p["admin_district"] or "", p["name"]))
    return places
