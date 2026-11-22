"""Read the Geofabrik PBF with pyosmium into compact Python structures.

Five streaming passes, each pre-filtered in C++ so Python only sees what it
needs (route/boundary/water relations, their member ways, POIs, node coords):

1. relations: route/superroute relations of interest, admin boundaries
   (admin_level 5 and 6), named water and protected-area multipolygons
2. ways referenced by those relations (node refs + selected tags)
3. ways that are POIs or polygons themselves (lakes, huts, castles, parking,
   named rivers, protected areas, ...)
4. nodes that carry POI tags (peaks, huts, stations, places, ...)
5. coordinates of every node referenced by a kept way

The result is pickled to ``data/raw/work/`` keyed by the extract name and the
extraction version, so later stages can be re-run without re-parsing.
"""

from __future__ import annotations

import pickle
import time
from array import array
from pathlib import Path

import numpy as np
import osmium
import osmium.filter as ofilter

from . import config

EXTRACT_VERSION = 3

# tags kept on route member ways
WAY_KEYS = (
    "highway",
    "surface",
    "tracktype",
    "sac_scale",
    "mtb:scale",
    "smoothness",
    "trail_visibility",
    "access",
    "foot",
    "bicycle",
    "dog",
    "name",
    "route",
    "railway",
    "aerialway",
    "bridge",
    "tunnel",
    "wheelchair",
    "lit",
    "via_ferrata_scale",
)
POI_KEYS = (
    "natural",
    "tourism",
    "amenity",
    "historic",
    "leisure",
    "waterway",
    "aerialway",
    "railway",
    "public_transport",
    "highway",
    "place",
    "boundary",
    "mountain_pass",
    "geological",
    "water",
    "heritage",
)

POI_RULES = {
    # key: set of values that are interesting (None = any value)
    "natural": {
        "peak",
        "volcano",
        "saddle",
        "waterfall",
        "cave_entrance",
        "gorge",
        "arch",
        "rock",
        "stone",
        "cliff",
        "water",
        "spring",
        "glacier",
        "tree",
        "beach",
        "sinkhole",
        "valley",
        "ridge",
    },
    "waterway": {"waterfall", "river", "rapids"},
    "tourism": {"alpine_hut", "wilderness_hut", "viewpoint", "museum", "attraction", "zoo", "picnic_site", "camp_site"},
    "amenity": {
        "restaurant",
        "cafe",
        "biergarten",
        "pub",
        "ice_cream",
        "parking",
        "place_of_worship",
        "monastery",
        "shelter",
        "drinking_water",
        "toilets",
    },
    "historic": {
        "castle",
        "ruins",
        "monastery",
        "archaeological_site",
        "monument",
        "city_gate",
        "fort",
        "manor",
        "palace",
        "church",
        "tower",
        "mine",
        "battlefield",
    },
    "leisure": {"playground", "bathing_place", "swimming_area", "nature_reserve", "bird_hide", "beach_resort"},
    "aerialway": {"station"},
    "railway": {"station", "halt", "tram_stop"},
    "public_transport": {"station", "platform", "stop_position"},
    "highway": {"bus_stop"},
    "place": {"city", "town", "village", "suburb", "hamlet", "isolated_dwelling", "locality"},
    "boundary": {"protected_area", "national_park"},
    "mountain_pass": {"yes"},
    "geological": None,
}

POI_TAG_KEEP = (
    "name",
    "ele",
    "natural",
    "tourism",
    "amenity",
    "historic",
    "leisure",
    "waterway",
    "aerialway",
    "railway",
    "public_transport",
    "highway",
    "place",
    "boundary",
    "water",
    "protect_class",
    "castle_type",
    "religion",
    "building",
    "train",
    "bus",
    "station",
    "subway",
    "tram",
    "light_rail",
    "fee",
    "capacity",
    "population",
    "wikidata",
    "wikipedia",
    "geological",
    "mountain_pass",
    "cuisine",
    "heritage",
    "access",
    "parking",
    "intermittent",
    "seasonal",
    "drinking_water",
    "opening_hours",
    "religion",
    "denomination",
    "admin_level",
)


def _tags(obj, keys=None) -> dict:
    if keys is None:
        return {t.k: t.v for t in obj.tags}
    return {k: obj.tags[k] for k in keys if k in obj.tags}


def is_poi(tags: dict) -> bool:
    for k, vals in POI_RULES.items():
        v = tags.get(k)
        if v is None:
            continue
        if vals is None or v in vals:
            return True
    return False


def _is_route_rel(tags) -> bool:
    t = tags.get("type")
    if t == "route":
        return tags.get("route") in config.ROUTE_TYPES
    if t == "superroute":
        r = tags.get("route")
        return r is None or r in config.ROUTE_TYPES
    return False


def _is_area_rel(tags) -> str | None:
    if tags.get("type") not in ("multipolygon", "boundary"):
        return None
    if tags.get("boundary") == "administrative" and tags.get("admin_level") in ("4", "5", "6"):
        return "admin"
    if (
        "name" in tags
        and tags.get("natural") == "water"
        and tags.get("water") not in ("river", "canal", "stream", "moat", "wastewater")
    ):
        return "water"
    if "name" in tags and (
        tags.get("leisure") == "nature_reserve" or tags.get("boundary") in ("national_park", "protected_area")
    ):
        return "protected"
    return None


def extract(pbf: Path, cache_dir: Path, log=print) -> dict:
    cache = cache_dir / f"{pbf.name.split('.')[0]}.extract-v{EXTRACT_VERSION}.pkl"
    if cache.exists():
        log(f"[extract] using cached {cache.name}")
        with cache.open("rb") as fh:
            return pickle.load(fh)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- pass 1: relations ------------------------------------------------
    routes: dict[int, dict] = {}
    areas: dict[int, dict] = {}
    for rel in osmium.FileProcessor(str(pbf), osmium.osm.RELATION).with_filter(ofilter.KeyFilter("type")):
        tags = _tags(rel)
        if _is_route_rel(tags):
            routes[rel.id] = {
                "tags": tags,
                "members": [(m.type, m.ref, m.role) for m in rel.members],
            }
        else:
            kind = _is_area_rel(tags)
            if kind:
                areas[rel.id] = {
                    "kind": kind,
                    "tags": {k: v for k, v in tags.items() if k in POI_TAG_KEEP or k == "name:de"},
                    "members": [(m.ref, m.role) for m in rel.members if m.type == "w"],
                }
    log(f"[extract] pass 1 relations: {len(routes)} route, {len(areas)} area  ({time.time() - t0:.0f}s)")

    member_ways = set()
    for r in routes.values():
        for t, ref, _ in r["members"]:
            if t == "w":
                member_ways.add(ref)
    area_ways = set()
    for a in areas.values():
        for ref, _ in a["members"]:
            area_ways.add(ref)

    # ---- pass 2: member ways ---------------------------------------------
    ways: dict[int, tuple] = {}  # id -> (np.int64 node refs, tags dict)
    wanted = member_ways | area_ways
    for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(ofilter.IdFilter(wanted)):
        refs = np.fromiter((n.ref for n in w.nodes), dtype=np.int64)
        tags = _tags(w, WAY_KEYS) if w.id in member_ways else {}
        ways[w.id] = (refs, tags)
    log(f"[extract] pass 2 member ways: {len(ways)} of {len(wanted)} referenced  ({time.time() - t0:.0f}s)")

    # ---- pass 3: POI / polygon ways ---------------------------------------
    poi_ways: dict[int, tuple] = {}
    for w in osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(ofilter.KeyFilter(*POI_KEYS)):
        tags = _tags(w)
        if not is_poi(tags):
            continue
        # unnamed things are only useful as parking/bus stops/playgrounds/water polygons...
        if "name" not in tags and not (
            tags.get("amenity") in ("parking", "drinking_water", "toilets", "shelter")
            or tags.get("leisure") in ("playground", "bathing_place")
            or tags.get("tourism") == "viewpoint"
        ):
            continue
        if tags.get("waterway") == "river" or tags.get("natural") == "water" or len(w.nodes) >= 2:
            refs = np.fromiter((n.ref for n in w.nodes), dtype=np.int64)
            poi_ways[w.id] = (refs, {k: v for k, v in tags.items() if k in POI_TAG_KEEP})
    log(f"[extract] pass 3 POI ways: {len(poi_ways)}  ({time.time() - t0:.0f}s)")

    # ---- pass 4: POI nodes ------------------------------------------------
    pid, plon, plat, ptags = array("q"), array("d"), array("d"), []
    for n in osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(ofilter.KeyFilter(*POI_KEYS)):
        tags = _tags(n)
        if not is_poi(tags):
            continue
        if "name" not in tags and not (
            tags.get("amenity") in ("parking", "drinking_water", "toilets", "shelter")
            or tags.get("leisure") in ("playground", "bathing_place")
            or tags.get("highway") == "bus_stop"
            or tags.get("public_transport") == "platform"
            or tags.get("tourism") == "viewpoint"
            or tags.get("waterway") == "waterfall"
            or tags.get("natural") in ("peak", "waterfall", "cave_entrance", "spring")
        ):
            continue
        if not n.location.valid():
            continue
        pid.append(n.id)
        plon.append(n.location.lon)
        plat.append(n.location.lat)
        ptags.append({k: v for k, v in tags.items() if k in POI_TAG_KEEP})
    log(f"[extract] pass 4 POI nodes: {len(pid)}  ({time.time() - t0:.0f}s)")

    # ---- pass 5: node coordinates -----------------------------------------
    need = set()
    for refs, _ in ways.values():
        need.update(refs.tolist())
    for refs, _ in poi_ways.values():
        need.update(refs.tolist())
    nid, nlon, nlat = array("q"), array("d"), array("d")
    for n in osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(ofilter.IdFilter(need)):
        if n.location.valid():
            nid.append(n.id)
            nlon.append(n.location.lon)
            nlat.append(n.location.lat)
    ids = np.frombuffer(nid, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    node_ids = ids[order]
    node_lon = np.frombuffer(nlon, dtype=np.float64)[order]
    node_lat = np.frombuffer(nlat, dtype=np.float64)[order]
    log(f"[extract] pass 5 node coords: {len(node_ids)} of {len(need)}  ({time.time() - t0:.0f}s)")

    out = {
        "routes": routes,
        "areas": areas,
        "ways": ways,
        "poi_ways": poi_ways,
        "poi_nodes": {
            "id": np.frombuffer(pid, dtype=np.int64).copy(),
            "lon": np.frombuffer(plon, dtype=np.float64).copy(),
            "lat": np.frombuffer(plat, dtype=np.float64).copy(),
            "tags": ptags,
        },
        "nodes": {"id": node_ids, "lon": node_lon, "lat": node_lat},
        "extract_seconds": round(time.time() - t0, 1),
    }
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    return out


class NodeIndex:
    """Sorted-array lookup of node coordinates."""

    def __init__(self, nodes: dict):
        self.ids = nodes["id"]
        self.lon = nodes["lon"]
        self.lat = nodes["lat"]

    def lookup(self, refs: np.ndarray):
        """Returns (lon, lat, ok_mask) for the given node ids."""
        pos = np.searchsorted(self.ids, refs)
        pos = np.clip(pos, 0, len(self.ids) - 1)
        ok = self.ids[pos] == refs
        return self.lon[pos], self.lat[pos], ok
