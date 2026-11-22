"""Polygons from OSM: admin boundaries, named lakes, protected areas.

Multipolygon relations are assembled with shapely.polygonize over the member
ways (outer and inner separately); closed ways become polygons directly.
All polygons are in the pipeline's local metric projection.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from .geo import LocalProjection
from .osm_extract import NodeIndex


def way_xy(refs: np.ndarray, nodes: NodeIndex, proj: LocalProjection):
    lon, lat, ok = nodes.lookup(refs)
    if not ok.all():
        return None
    x, y = proj.fwd(lon, lat)
    return np.column_stack([x, y])


def relation_polygon(members, ways, nodes, proj):
    """members: [(way_id, role)]. Returns (geometry | None, complete: bool)."""
    outers, inners = [], []
    complete = True
    for ref, role in members:
        w = ways.get(ref)
        if w is None:
            complete = False
            continue
        xy = way_xy(w[0], nodes, proj)
        if xy is None or len(xy) < 2:
            complete = False
            continue
        (inners if role == "inner" else outers).append(LineString(xy))
    if not outers:
        return None, False
    outer = shapely.unary_union(list(shapely.polygonize(outers).geoms))
    if outer.is_empty:
        return None, False
    if inners:
        inner = shapely.unary_union(list(shapely.polygonize(inners).geoms))
        if not inner.is_empty:
            outer = outer.difference(inner)
    outer = shapely.make_valid(outer)
    return (outer if not outer.is_empty else None), complete


def closed_way_polygon(refs, nodes, proj):
    if len(refs) < 4 or refs[0] != refs[-1]:
        return None
    xy = way_xy(refs, nodes, proj)
    if xy is None:
        return None
    p = Polygon(xy)
    if not p.is_valid:
        p = shapely.make_valid(p)
    return p if not p.is_empty and p.area > 0 else None


LAKE_WATER_VALUES = {None, "lake", "reservoir", "pond", "oxbow", "lagoon", "natural", "yes"}


def build_areas(ex: dict, nodes: NodeIndex, proj: LocalProjection, log=print) -> dict:
    admin5, admin6, admin4 = [], [], []
    lakes, protected = [], []
    incomplete_admin = 0
    for rid in sorted(ex["areas"]):
        a = ex["areas"][rid]
        tags = a["tags"]
        geom, complete = relation_polygon(a["members"], ex["ways"], nodes, proj)
        if geom is None:
            if a["kind"] == "admin":
                incomplete_admin += 1
            continue
        name = tags.get("name")
        rec = {"osm": f"r{rid}", "name": name, "geom": geom, "tags": tags}
        if a["kind"] == "admin":
            if not complete:
                incomplete_admin += 1
                continue  # truncated at the extract boundary (e.g. Austrian districts)
            lvl = tags.get("admin_level")
            (admin4 if lvl == "4" else admin5 if lvl == "5" else admin6).append(rec)
        elif a["kind"] == "water":
            if tags.get("water") in LAKE_WATER_VALUES:
                lakes.append(rec)
        elif a["kind"] == "protected":
            protected.append(rec)
    for wid in sorted(ex["poi_ways"]):
        refs, tags = ex["poi_ways"][wid]
        name = tags.get("name")
        if not name:
            continue
        is_lake = tags.get("natural") == "water" and tags.get("water") in LAKE_WATER_VALUES
        is_prot = tags.get("leisure") == "nature_reserve" or tags.get("boundary") in ("protected_area", "national_park")
        if not (is_lake or is_prot):
            continue
        p = closed_way_polygon(refs, nodes, proj)
        if p is None:
            continue
        rec = {"osm": f"w{wid}", "name": name, "geom": p, "tags": tags}
        (lakes if is_lake else protected).append(rec)
    log(
        f"[areas] admin4={len(admin4)} admin5={len(admin5)} admin6={len(admin6)} "
        f"(skipped incomplete {incomplete_admin}) lakes={len(lakes)} protected={len(protected)}"
    )
    return {"admin4": admin4, "admin5": admin5, "admin6": admin6, "lakes": lakes, "protected": protected}
