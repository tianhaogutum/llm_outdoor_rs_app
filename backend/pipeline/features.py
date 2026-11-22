"""Named features along a route: classification of OSM POIs and proximity queries.

Each POI is classified into exactly one feature type (first matching rule
wins, see ``classify``). A feature counts as "on the route" when its
geometry lies within the type's distance threshold of the route line
(``ON_ROUTE_M``). Start-related features (stations, bus stops, parking) are
measured from the start point instead (``NEAR_START_M``).
"""

from __future__ import annotations

import re

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from .areas import way_xy
from .geo import LocalProjection
from .osm_extract import NodeIndex

GORGE_RE = re.compile(r"(klamm|schlucht|tobel|gorge)", re.I)
GEOTOP_RE = re.compile(r"geotop", re.I)

# distance from the route line (m) within which a feature counts as on the route
ON_ROUTE_M = {
    "peak": 75,
    "saddle": 50,
    "alpine_hut": 100,
    "wilderness_hut": 75,
    "shelter": 50,
    "restaurant": 75,
    "beer_garden": 75,
    "viewpoint": 50,
    "waterfall": 150,
    "gorge": 150,
    "cave": 100,
    "geological": 100,
    "lake": 100,
    "river": 60,
    "glacier": 300,
    "spring": 50,
    "castle": 150,
    "ruins": 100,
    "monastery": 150,
    "church": 75,
    "museum": 100,
    "archaeological_site": 100,
    "monument": 75,
    "attraction": 75,
    "playground": 75,
    "swimming": 150,
    "zoo": 200,
    "bird_hide": 100,
    "drinking_water": 50,
    "picnic_site": 50,
    "campsite": 150,
    "aerialway_station": 150,
    "natural_monument_tree": 50,
}
# distance from the start point (m)
NEAR_START_M = {"train_station": 1000, "bus_stop": 300, "tram_stop": 500, "parking": 300}

ROUTE_TYPES_ORDER = list(ON_ROUTE_M)


def classify(t: dict) -> str | None:
    """Map OSM tags to one feature type (or None)."""
    n = t.get("natural")
    tour = t.get("tourism")
    am = t.get("amenity")
    hist = t.get("historic")
    lei = t.get("leisure")
    ww = t.get("waterway")
    name = t.get("name", "")
    if n in ("peak", "volcano"):
        return "peak"
    if n == "saddle" or t.get("mountain_pass") == "yes":
        return "saddle"
    if tour == "alpine_hut":
        return "alpine_hut"
    if tour == "wilderness_hut":
        return "wilderness_hut"
    if am == "biergarten":
        return "beer_garden"
    if am in ("restaurant", "cafe", "pub", "ice_cream"):
        return "restaurant"
    if tour == "viewpoint":
        return "viewpoint"
    if ww == "waterfall" or n == "waterfall":
        return "waterfall"
    if n == "gorge" or (name and GORGE_RE.search(name) and (tour == "attraction" or n in ("valley", "gorge"))):
        return "gorge"
    if n == "cave_entrance":
        return "cave"
    if "geological" in t or n in ("arch", "rock", "stone", "sinkhole", "cliff") or (name and GEOTOP_RE.search(name)):
        return "geological" if (name or "geological" in t) else None
    if n == "glacier":
        return "glacier"
    if n == "spring":
        return "spring" if name else None
    if n == "tree":
        return "natural_monument_tree" if name else None
    if hist in ("castle", "palace", "manor", "fort"):
        return "castle"
    if hist == "ruins":
        return "ruins"
    if am == "monastery" or hist == "monastery":
        return "monastery"
    if tour == "museum":
        return "museum"
    if hist == "archaeological_site":
        return "archaeological_site"
    if hist in ("monument", "city_gate", "tower", "mine", "battlefield"):
        return "monument"
    if am == "place_of_worship" or hist == "church":
        return "church" if name else None
    if lei == "playground":
        return "playground"
    if lei in ("bathing_place", "swimming_area", "beach_resort") or n == "beach":
        return "swimming"
    if tour == "zoo":
        return "zoo"
    if lei == "bird_hide":
        return "bird_hide"
    if am == "drinking_water":
        return "drinking_water"
    if tour == "picnic_site":
        return "picnic_site"
    if tour == "camp_site":
        return "campsite"
    if t.get("aerialway") == "station":
        return "aerialway_station"
    if am == "shelter":
        return "shelter"
    if tour == "attraction":
        return "attraction" if name else None
    # start-related
    rw = t.get("railway")
    if rw in ("station", "halt") or (t.get("public_transport") == "station" and t.get("train") == "yes"):
        if t.get("station") == "subway":
            return "tram_stop"
        return "train_station"
    if rw == "tram_stop" or t.get("station") in ("subway", "light_rail"):
        return "tram_stop"
    if t.get("highway") == "bus_stop" or (t.get("public_transport") == "platform" and t.get("bus") == "yes"):
        return "bus_stop"
    if am == "parking" and t.get("access") not in ("private", "no", "customers"):
        return "parking"
    if t.get("place") in ("city", "town", "village", "suburb", "hamlet"):
        return "place"
    return None


def _ele(t):
    v = t.get("ele")
    if not v:
        return None
    m = re.match(r"^\s*(-?\d+(?:[.,]\d+)?)", v)
    if not m:
        return None
    try:
        e = float(m.group(1).replace(",", "."))
    except ValueError:
        return None
    return round(e) if -50 < e < 5000 else None


class FeatureIndex:
    def __init__(self, ex: dict, nodes: NodeIndex, areas: dict, proj: LocalProjection, log=print):
        self.proj = proj
        geoms, kinds, names, extra = [], [], [], []
        pn = ex["poi_nodes"]
        x, y = proj.fwd(pn["lon"], pn["lat"])
        pts = shapely.points(np.column_stack([x, y]))
        for i in range(len(pn["id"])):
            t = pn["tags"][i]
            k = classify(t)
            if k is None:
                continue
            geoms.append(pts[i])
            kinds.append(k)
            names.append(t.get("name"))
            extra.append({"osm": f"n{int(pn['id'][i])}", "ele": _ele(t), "tags": t})
        n_nodes = len(geoms)
        lake_way_ids = {a["osm"] for a in areas["lakes"]}
        for wid in sorted(ex["poi_ways"]):
            refs, t = ex["poi_ways"][wid]
            if f"w{wid}" in lake_way_ids:
                continue
            if t.get("waterway") == "river":
                if not t.get("name"):
                    continue
                xy = way_xy(refs, nodes, proj)
                if xy is None or len(xy) < 2:
                    continue
                geoms.append(LineString(xy))
                kinds.append("river")
                names.append(t["name"])
                extra.append({"osm": f"w{wid}", "ele": None, "tags": t})
                continue
            if (
                t.get("natural") == "water"
                or t.get("leisure") == "nature_reserve"
                or t.get("boundary") in ("protected_area", "national_park")
            ):
                continue  # polygons handled in areas
            k = classify(t)
            if k is None:
                continue
            xy = way_xy(refs, nodes, proj)
            if xy is None or len(xy) == 0:
                continue
            if len(xy) >= 4 and refs[0] == refs[-1]:
                g = shapely.Polygon(xy)
                if not g.is_valid:
                    g = g.buffer(0)
            elif len(xy) >= 2:
                g = LineString(xy)
            else:
                g = Point(xy[0])
            # a gorge or river as line/area keeps its shape; a hut building is used as is
            geoms.append(g)
            kinds.append(k)
            names.append(t.get("name"))
            extra.append({"osm": f"w{wid}", "ele": _ele(t), "tags": t})
        for a in areas["lakes"]:
            if a["geom"].area < 10_000:  # < 1 ha: ponds
                continue
            geoms.append(a["geom"])
            kinds.append("lake")
            names.append(a["name"])
            extra.append({"osm": a["osm"], "ele": None, "tags": a["tags"], "area_ha": round(a["geom"].area / 1e4, 1)})
        self.geoms = np.array(geoms, dtype=object)
        self.kinds = np.array(kinds)
        self.names = names
        self.extra = extra
        self.tree = STRtree(self.geoms)
        self.n_by_kind = {k: int(v) for k, v in zip(*np.unique(self.kinds, return_counts=True), strict=True)}
        log(f"[features] {len(geoms)} classified features ({n_nodes} from nodes): {self.n_by_kind}")

        # places for locality (points only)
        pl = [i for i, k in enumerate(kinds) if k == "place" and names[i]]
        self.place_idx = np.array(pl, dtype=int)
        self.place_tree = STRtree(self.geoms[self.place_idx]) if pl else None

        self.protected = areas["protected"]
        self.protected_tree = STRtree([a["geom"] for a in self.protected]) if self.protected else None

    # ------------------------------------------------------------------
    def along_route(self, line: LineString, cum_line: LineString | None = None):
        """All classified features within their threshold of the line.

        Returns {kind: [ {name, dist_m, at_m, ele, osm, ...} ]} sorted by position."""
        maxd = max(ON_ROUTE_M.values())
        idx = self.tree.query(line, predicate="dwithin", distance=maxd)
        out: dict[str, list] = {}
        if len(idx) == 0:
            return out
        dists = shapely.distance(self.geoms[idx], line)
        for i, d in zip(idx.tolist(), dists.tolist(), strict=True):
            k = self.kinds[i]
            thr = ON_ROUTE_M.get(k)
            if thr is None or d > thr:
                continue
            g = self.geoms[i]
            if k == "river":
                # require the route to follow the river for >= 1 km
                along = line.intersection(g.buffer(thr)).length
                if along < 1000:
                    continue
                at = line.project(g.interpolate(g.project(line.interpolate(0))))
                rec = {"name": self.names[i], "along_m": round(along), "osm": self.extra[i]["osm"]}
                out.setdefault(k, []).append((at, rec))
                continue
            # position along the route of the nearest point
            near = nearest_points(line, g)[0]
            at = line.project(near)
            rec = {"name": self.names[i], "dist_m": round(d), "osm": self.extra[i]["osm"]}
            e = self.extra[i].get("ele")
            if e is not None:
                rec["ele_m"] = e
            if "area_ha" in self.extra[i]:
                rec["area_ha"] = self.extra[i]["area_ha"]
            out.setdefault(k, []).append((at, rec))
        res = {}
        for k, lst in out.items():
            lst.sort(key=lambda t: (t[0], t[1]["osm"]))
            # merge repeated names (a river or lake hit several times) keep first
            seen = set()
            items = []
            for at, rec in lst:
                key = rec["name"] or rec["osm"]
                if rec["name"] and key in seen:
                    continue
                seen.add(key)
                rec["at_m"] = round(at)
                items.append(rec)
            res[k] = items
        return res

    def near_point(self, xy, kinds_thr: dict):
        p = Point(xy)
        maxd = max(kinds_thr.values())
        idx = self.tree.query(p, predicate="dwithin", distance=maxd)
        out: dict[str, list] = {}
        if len(idx) == 0:
            return out
        dists = shapely.distance(self.geoms[idx], p)
        for i, d in zip(idx.tolist(), dists.tolist(), strict=True):
            k = self.kinds[i]
            thr = kinds_thr.get(k)
            if thr is None or d > thr:
                continue
            rec = {"name": self.names[i], "dist_m": round(d), "osm": self.extra[i]["osm"]}
            if k == "parking":
                cap = self.extra[i]["tags"].get("capacity")
                if cap and cap.isdigit():
                    rec["capacity"] = int(cap)
                fee = self.extra[i]["tags"].get("fee")
                if fee:
                    rec["fee"] = fee
            out.setdefault(k, []).append(rec)
        for k in out:
            out[k].sort(key=lambda r: (r["dist_m"], r["osm"]))
        return out

    def nearest_place(self, xy, kinds=("city", "town", "village"), max_m=15_000):
        if self.place_tree is None:
            return None
        p = Point(xy)
        idx = self.place_tree.query(p, predicate="dwithin", distance=max_m)
        best = None
        for j in idx.tolist():
            i = self.place_idx[j]
            t = self.extra[i]["tags"]
            if t.get("place") not in kinds:
                continue
            d = self.geoms[i].distance(p)
            key = (d, self.extra[i]["osm"])
            if best is None or key < best[0]:
                best = (key, i)
        if best is None:
            return None
        i = best[1]
        return {
            "name": self.names[i],
            "place": self.extra[i]["tags"].get("place"),
            "dist_m": round(best[0][0]),
            "osm": self.extra[i]["osm"],
        }

    def protected_share(self, line: LineString):
        if self.protected_tree is None:
            return []
        idx = self.protected_tree.query(line, predicate="intersects")
        res = []
        L = line.length
        for j in sorted(idx.tolist()):
            a = self.protected[j]
            try:
                inside = line.intersection(a["geom"]).length
            except shapely.errors.GEOSException:
                continue
            share = inside / L if L else 0
            if share >= 0.10:
                t = a["tags"]
                kind = (
                    "national_park"
                    if t.get("boundary") == "national_park"
                    else "nature_reserve"
                    if t.get("leisure") == "nature_reserve" or t.get("protect_class") in ("1", "1a", "1b", "4")
                    else "protected_area"
                )
                res.append({"name": a["name"], "kind": kind, "share": round(share, 2), "osm": a["osm"]})
        # one entry per name
        seen, out = set(), []
        for r in sorted(res, key=lambda r: (-r["share"], r["osm"])):
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            out.append(r)
        return out
