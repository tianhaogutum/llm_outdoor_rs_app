"""Build the route catalogue.

    cd backend && python -m pipeline.build [--extract bayern-261101]

Stages: download (cached) -> extract (cached) -> polygons and feature index
-> per-route assembly and filters -> dedup -> elevation calibration ->
attributes, tags, descriptions -> write catalogue + manifest + report.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
from collections import Counter, defaultdict

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point

from . import PIPELINE_VERSION, attributes, config, describe, download
from . import dem as demmod
from . import tags as tagmod
from .areas import build_areas
from .dedup import find_duplicates
from .features import NEAR_START_M, FeatureIndex
from .geo import LocalProjection, WayPiece, is_loop, point_dist_m, resample, self_retrace_share, simplify, stitch
from .osm_extract import EXTRACT_VERSION, NodeIndex, extract

KEEP_OSM_TAGS = (
    "name",
    "name:de",
    "name:en",
    "ref",
    "route",
    "type",
    "network",
    "network:type",
    "operator",
    "osmc:symbol",
    "symbol",
    "colour",
    "website",
    "url",
    "wikidata",
    "wikipedia",
    "from",
    "to",
    "via",
    "roundtrip",
    "distance",
    "length",
    "ascent",
    "descent",
    "description",
    "description:de",
    "description:en",
    "state",
    "dog",
    "wheelchair",
    "educational",
    "pilgrimage",
    "duration",
    "sac_scale",
    "mtb:scale",
    "survey:date",
    "check_date",
)

DROP_REASONS = [
    "inactive_state",
    "unnamed",
    "no_member_ways",
    "incomplete_in_extract",
    "stitch_failed",
    "too_short",
    "too_long",
    "outside_region",
    "superroute_single_child",
    "no_elevation",
    "duplicate",
]


def _log(msg):
    print(msg, flush=True)


def _r(v, nd=0):
    if v is None:
        return None
    return int(round(v)) if nd == 0 else round(float(v), nd)


def _activity(rid, routes, seen=None) -> str | None:
    r = routes[rid]
    rt = r["tags"].get("route")
    if rt in config.ROUTE_TYPES:
        return config.ROUTE_TYPES[rt]
    seen = seen or set()
    seen.add(rid)
    acts = Counter()
    for t, ref, _ in r["members"]:
        if t == "r" and ref in routes and ref not in seen:
            a = _activity(ref, routes, seen)
            if a:
                acts[a] += 1
    return acts.most_common(1)[0][0] if acts else None


def _expand(rid, routes, seen):
    """Main member ways in order, recursing into child route relations."""
    r = routes[rid]
    main, children, missing_rel = [], [], 0
    for t, ref, role in r["members"]:
        role = (role or "").strip().lower()
        if t == "w":
            if role in config.MAIN_ROLES:
                main.append(ref)
        elif t == "r":
            if role not in config.MAIN_ROLES and role != "stage":
                continue
            if ref not in routes:
                missing_rel += 1
                continue
            if ref in seen:
                continue
            seen.add(ref)
            children.append(ref)
            m2, _, mr2 = _expand(ref, routes, seen)
            main += m2
            missing_rel += mr2
    return main, children, missing_rel


def _title(tags):
    for k in ("name", "name:de", "name:en"):
        v = tags.get(k)
        if v and v.strip():
            return " ".join(v.split())
    return None


def calibrate_dem(cands, log=_log):
    """Pick smoothing window and hysteresis on the calibration half of the
    routes that carry an OSM ascent tag; report on the validation half."""
    windows = [0.0, 60.0, 100.0, 160.0, 240.0, 320.0, 400.0]
    hysts = [0.0, 2.0, 5.0, 10.0, 15.0]
    tagged = [c for c in cands if c["osm_ascent_m"] and c["osm_ascent_m"] >= 50]
    calib = [c for c in tagged if int(hashlib.md5(str(c["id"]).encode()).hexdigest(), 16) % 2 == 0]
    valid = [c for c in tagged if c not in calib]
    grid = []
    for w in windows:
        for h in hysts:
            errs = []
            for c in calib:
                zs = demmod.smooth(c["z"], c["d"], w)
                up, _ = demmod.ascent_descent(zs, h)
                errs.append((up - c["osm_ascent_m"]) / c["osm_ascent_m"])
            e = np.abs(errs) if errs else np.array([np.nan])
            grid.append(
                {
                    "window_m": w,
                    "hysteresis_m": h,
                    "median_abs_rel_err": float(np.median(e)),
                    "median_rel_err": float(np.median(errs)) if errs else None,
                }
            )
    best = min(grid, key=lambda g: (g["median_abs_rel_err"], g["window_m"], g["hysteresis_m"]))
    log(
        f"[dem] calibration n={len(calib)} best window={best['window_m']} hyst={best['hysteresis_m']} "
        f"MdAPE={best['median_abs_rel_err']:.3f}"
    )
    return best, grid, calib, valid


def error_summary(pairs):
    """pairs: list of (ours, reference). Relative error stats."""
    if not pairs:
        return {"n": 0}
    rel = np.array([(o - ref) / ref for o, ref in pairs])
    a = np.abs(rel)
    absd = np.array([o - ref for o, ref in pairs])
    return {
        "n": len(pairs),
        "median_rel_err": round(float(np.median(rel)), 3),
        "mean_rel_err": round(float(np.mean(rel)), 3),
        "median_abs_rel_err": round(float(np.median(a)), 3),
        "p90_abs_rel_err": round(float(np.quantile(a, 0.9)), 3),
        "share_within_10pct": round(float((a <= 0.10).mean()), 3),
        "share_within_20pct": round(float((a <= 0.20).mean()), 3),
        "median_abs_err": round(float(np.median(np.abs(absd))), 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", default=config.OSM_EXTRACT)
    ap.add_argument("--limit", type=int, default=0, help="debug: process only the first N route relations")
    args = ap.parse_args(argv)
    t_start = time.time()
    timings = {}

    # ---- inputs ----------------------------------------------------------
    pbf = download.osm_extract(args.extract, log=_log)
    import osmium

    hdr = osmium.io.Reader(str(pbf), osmium.osm.osm_entity_bits.NOTHING).header()
    osm_header = {
        "osmosis_replication_timestamp": hdr.get("osmosis_replication_timestamp"),
        "osmosis_replication_sequence_number": hdr.get("osmosis_replication_sequence_number"),
        "osmosis_replication_base_url": hdr.get("osmosis_replication_base_url"),
    }
    ex = extract(pbf, config.WORK_DIR, log=_log)
    timings["extract_s"] = round(time.time() - t_start, 1)
    nodes = NodeIndex(ex["nodes"])
    proj = LocalProjection(48.5, 11.5)

    t0 = time.time()
    areas = build_areas(ex, nodes, proj, log=_log)
    lvl = {"4": areas["admin4"], "5": areas["admin5"], "6": areas["admin6"]}[config.REGION_ADMIN_LEVEL]
    by_name = {a["name"]: a for a in lvl}
    missing = [n for n in config.REGION_ADMIN_NAMES if n not in by_name]
    if missing:
        raise SystemExit(
            f"region parts not found among admin_level={config.REGION_ADMIN_LEVEL} boundaries: "
            f"{missing}; have {sorted(by_name)}"
        )
    region = shapely.unary_union([by_name[n]["geom"] for n in config.REGION_ADMIN_NAMES])
    shapely.prepare(region)
    a5_list = sorted(areas["admin5"], key=lambda a: a["name"])
    a5_tree = STRtree([a["geom"] for a in a5_list])
    a6_list = sorted(areas["admin6"], key=lambda a: a["osm"])
    a6_tree = STRtree([a["geom"] for a in a6_list])
    feat = FeatureIndex(ex, nodes, areas, proj, log=_log)
    timings["index_s"] = round(time.time() - t0, 1)

    # ---- DEM -------------------------------------------------------------
    used_tiles: dict[str, str | None] = {}

    def tile_loader(name):
        p = download.dem_tile(name, log=_log)
        used_tiles[name] = None if p is None else p.name
        return None if p is None else demmod.read_hgt(p)

    dem = demmod.DEM(tile_loader)

    routes = ex["routes"]
    ways = ex["ways"]

    # parent/child relations and way usage (popularity proxy)
    parents = defaultdict(list)
    for rid, r in routes.items():
        for t, ref, _ in r["members"]:
            if t == "r" and ref in routes:
                parents[ref].append(rid)
    way_use = Counter()
    for r in routes.values():
        if r["tags"].get("type") == "route" and _title(r["tags"]):
            for w in {ref for t, ref, role in r["members"] if t == "w"}:
                way_use[w] += 1

    # ---- per-route assembly --------------------------------------------------
    t0 = time.time()
    drops = Counter()
    drop_ids = defaultdict(list)
    stitch_stats = Counter()
    region_counts = Counter()  # by admin5 of start, after quality filters, before region filter
    cands = []
    ids = sorted(routes)
    if args.limit:
        ids = ids[: args.limit]

    def drop(reason, rid):
        drops[reason] += 1
        drop_ids[reason].append(rid)

    for n_done, rid in enumerate(ids):
        if n_done and n_done % 2000 == 0:
            _log(f"[routes] {n_done}/{len(ids)}  kept so far {len(cands)}  ({time.time() - t0:.0f}s)")
        r = routes[rid]
        tags = r["tags"]
        if tags.get("state") in config.INACTIVE_STATES or any(
            k.split(":")[0] in config.INACTIVE_STATES for k in tags if k.endswith(":route") or k.endswith(":type")
        ):
            drop("inactive_state", rid)
            continue
        title = _title(tags)
        if not title:
            drop("unnamed", rid)
            continue
        activity = _activity(rid, routes)
        if activity is None:
            drop("no_member_ways", rid)
            continue
        main_ways, children, missing_rel = _expand(rid, routes, {rid})
        if tags.get("type") == "superroute" or children:
            if len(children) == 1 and not [1 for t, _, _ in r["members"] if t == "w"]:
                drop("superroute_single_child", rid)
                continue
        if not main_ways:
            drop("no_member_ways", rid)
            continue
        missing_w = sum(1 for w in main_ways if w not in ways)
        if missing_w or missing_rel:
            drop("incomplete_in_extract", rid)
            continue
        pieces = []
        incomplete = False
        for w in main_ways:
            refs, _ = ways[w]
            lon, lat, ok = nodes.lookup(refs)
            if not ok.all():
                incomplete = True
                break
            pieces.append(WayPiece(w, refs.tolist(), list(zip(lon.tolist(), lat.tolist(), strict=True))))
        if incomplete:
            drop("incomplete_in_extract", rid)
            continue
        res = stitch(pieces, config.STITCH_GAP_TOLERANCE_M)
        if res is None:
            drop("no_member_ways", rid)
            continue
        stitch_stats["assembled_attempts"] += 1
        if res.method == "ordered":
            stitch_stats["ordered_single_segment"] += 1
        else:
            stitch_stats["graph_assembly"] += 1
        if res.coverage < config.STITCH_MIN_COVERAGE or res.gap_m > max(
            config.STITCH_MAX_GAP_SHARE * res.length_m, config.STITCH_MAX_GAP_M
        ):
            stitch_stats["failed"] += 1
            drop("stitch_failed", rid)
            continue
        if res.method == "graph":
            stitch_stats["graph_accepted"] += 1
            if res.n_chains > 1:
                stitch_stats["accepted_with_dropped_branches"] += 1
            if res.n_gaps_bridged:
                stitch_stats["accepted_with_bridged_gaps"] += 1
        coords = res.coords
        length = res.length_m
        lo_lim, hi_lim = config.LENGTH_LIMITS_M[activity]
        if length < lo_lim:
            drop("too_short", rid)
            continue
        if length > hi_lim:
            drop("too_long", rid)
            continue

        # region membership
        xy = proj.fwd_coords(coords)
        sxy = xy[0]
        samp, _ = resample(coords, 100.0)
        sxy_all = proj.fwd_coords(samp)
        a5 = [a5_list[j]["name"] for j in a5_tree.query(Point(sxy), predicate="intersects").tolist()]
        start_a5 = a5[0] if a5 else None
        region_counts[start_a5 or "outside Bavaria"] += 1
        inside = shapely.contains_xy(region, sxy_all[:, 0], sxy_all[:, 1]).mean()
        if not shapely.contains_xy(region, sxy[0], sxy[1]) or inside < config.REGION_MIN_INSIDE_SHARE:
            drop("outside_region", rid)
            continue

        # elevation samples (raw; smoothing chosen later)
        pts, d = resample(coords, config.DEM_SAMPLE_SPACING_M)
        z = dem.sample(pts[:, 0], pts[:, 1])
        z, n_void = demmod.fill_voids(z, d)
        if np.isnan(z).all():
            drop("no_elevation", rid)
            continue

        # per-way metres actually used by the main line (a way may be used
        # partly after splitting, or twice in an out-and-back)
        way_ids_seq = [w for w, _ in res.way_seq]
        way_len = [L for _, L in res.way_seq]
        way_tags = [ways[w][1] for w in way_ids_seq]
        prof = attributes.way_profile(way_tags, way_len)
        tot = sum(way_len) or 1.0
        prof["other_routes_on_same_ways"] = round(
            sum((way_use[w] - 1) * L for w, L in zip(way_ids_seq, way_len, strict=True)) / tot, 2
        )

        loop = is_loop(coords, config.LOOP_MAX_ENDPOINT_DIST_M)
        retrace = self_retrace_share(coords, config.OUT_AND_BACK_TOL_M, 300.0, proj) if loop else 0.0

        cands.append(
            {
                "id": rid,
                "title": title,
                "activity": activity,
                "tags": tags,
                "coords": coords,
                "xy": xy,
                "length_m": length,
                "pts": pts,
                "d": d,
                "z": z,
                "void_share": n_void / len(z),
                "profile": prof,
                "is_loop": loop,
                "retrace": retrace,
                "stitch": res,
                "children": children,
                "start_a5": start_a5,
                "osm_ascent_m": attributes.parse_ele_tag(tags.get("ascent")),
                "osm_descent_m": attributes.parse_ele_tag(tags.get("descent")),
                "osm_length_m": attributes.parse_length_tag(tags.get("distance") or tags.get("length")),
            }
        )
    timings["routes_s"] = round(time.time() - t0, 1)
    _log(f"[routes] candidates after filters: {len(cands)}; drops: {dict(drops)}")

    # ---- dedup ------------------------------------------------------------
    t0 = time.time()
    items = []
    for c in cands:
        line = LineString(c["xy"]) if len(c["xy"]) >= 2 else None
        samp, _ = resample(c["coords"], 50.0)
        items.append(
            {
                "id": c["id"],
                "activity": c["activity"],
                "line": line,
                "samples": proj.fwd_coords(samp),
                "length_m": c["length_m"],
                "osm_tags": c["tags"],
            }
        )
    dup_of, pairs_checked = find_duplicates(items, config.DEDUP_MATCH_TOL_M, config.DEDUP_MIN_OVERLAP)
    merged = defaultdict(list)
    for a, b in dup_of.items():
        # resolve chains a -> b -> c
        while b in dup_of:
            b = dup_of[b]
        merged[b].append(a)
        drop("duplicate", a)
    cands = [c for c in cands if c["id"] not in dup_of]
    timings["dedup_s"] = round(time.time() - t0, 1)
    _log(f"[dedup] dropped {len(dup_of)} duplicates ({pairs_checked} pairs compared)")

    # ---- elevation calibration ------------------------------------------------
    best, grid, calib, valid = calibrate_dem(cands)
    win, hyst = best["window_m"], best["hysteresis_m"]

    # ---- final records ----------------------------------------------------------
    t0 = time.time()
    kept_ids = {c["id"] for c in cands}
    records = []
    geoms = []
    details = []
    for c in cands:
        rid = c["id"]
        coords = c["coords"]
        zs = demmod.smooth(c["z"], c["d"], win)
        up, down = demmod.ascent_descent(zs, hyst)
        prof = c["profile"]
        act = c["activity"]
        tags = c["tags"]
        length = c["length_m"]
        line = LineString(c["xy"])
        on = feat.along_route(line)
        sxy, exy = c["xy"][0], c["xy"][-1]
        near_start = feat.near_point(sxy, NEAR_START_M)
        near_end = feat.near_point(exy, NEAR_START_M) if not c["is_loop"] else None
        prot = feat.protected_share(line)
        loc = feat.nearest_place(sxy) or feat.nearest_place(sxy, ("suburb", "hamlet"), 5000)
        end_loc = (
            None if c["is_loop"] else (feat.nearest_place(exy) or feat.nearest_place(exy, ("suburb", "hamlet"), 5000))
        )
        a6 = [a6_list[j] for j in a6_tree.query(Point(sxy), predicate="intersects").tolist()]
        district = a6[0]["name"] if a6 else None
        gcen = line.centroid
        gcen_lon, gcen_lat = proj.inv(gcen.x, gcen.y)

        cap = config.FEATURE_LIST_CAP
        # lists keep named items only (unnamed shelters, playgrounds, ... are in the counts)
        on_capped = {k: [x for x in v if x.get("name")][:cap] for k, v in sorted(on.items())}
        on_capped = {k: v for k, v in on_capped.items() if v}
        counts = {k: len(v) for k, v in sorted(on.items())}
        near_start_c = {k: v[:3] for k, v in sorted(near_start.items())}
        near_end_c = None
        if near_end is not None:
            near_end_c = {
                k: v[:1] for k, v in sorted(near_end.items()) if k in ("train_station", "bus_stop", "tram_stop")
            }
        km = length / 1000.0
        poi_per_km = round(
            sum(
                len(on.get(k, []))
                for k in (
                    "restaurant",
                    "beer_garden",
                    "alpine_hut",
                    "parking",
                    "playground",
                    "viewpoint",
                    "attraction",
                    "museum",
                    "castle",
                )
            )
            / max(km, 0.1),
            2,
        )
        parking_cap = sum(p.get("capacity", 0) for p in near_start.get("parking", []))

        dif = attributes.difficulty(act, length, up, prof, down)
        cat = attributes.category(act, tags.get("route"), c["title"], tags, length, prof["unpaved_share"], up)
        dur = attributes.duration_min(act, length, up, down)
        days = attributes.est_days(dur, act)
        osm_tags = {k: tags[k] for k in KEEP_OSM_TAGS if k in tags}
        stage_of = sorted(p for p in parents.get(rid, []) if p in kept_ids)
        stages = [ch for ch in c["children"] if ch in kept_ids]
        res = c["stitch"]
        rec = {
            "id": rid,
            "source": "osm",
            "osm_type": "relation",
            "title": c["title"],
            "description": None,
            "description_source": None,
            "category_name": cat,
            "activity": act,
            "osm_route": tags.get("route") or "superroute",
            "tags": [],
            "difficulty": dif["difficulty"],
            "difficulty_source": dif["difficulty_source"],
            "difficulty_technical": dif["difficulty_technical"],
            "difficulty_effort": dif["difficulty_effort"],
            "max_grade": dif["max_grade"],
            "length_m": _r(length),
            "ascent_m": _r(up),
            "descent_m": _r(down),
            "min_ele_m": _r(float(np.min(zs))),
            "max_ele_m": _r(float(np.max(zs))),
            "start_ele_m": _r(float(zs[0])),
            "max_grade_200m_pct": _r(_max_grade(zs, c["d"]) * 100, 1),
            "duration_min": dur,
            "est_days": days,
            "start_lat": round(coords[0][1], config.COORD_DECIMALS),
            "start_lon": round(coords[0][0], config.COORD_DECIMALS),
            "end_lat": round(coords[-1][1], config.COORD_DECIMALS),
            "end_lon": round(coords[-1][0], config.COORD_DECIMALS),
            "centroid_lat": round(float(gcen_lat), config.COORD_DECIMALS),
            "centroid_lon": round(float(gcen_lon), config.COORD_DECIMALS),
            "is_loop": c["is_loop"],
            "start_end_dist_m": _r(point_dist_m(coords[0], coords[-1])),
            "locality": loc["name"] if loc else None,
            "locality_dist_m": loc["dist_m"] if loc else None,
            "end_locality": end_loc["name"] if end_loc else None,
            "admin_district": district,
            "admin_region": c["start_a5"],
            "stage_of": stage_of,
            "stages": stages,
            "duplicates_merged": sorted(merged.get(rid, [])),
            "features": {
                "on_route": on_capped,
                "on_route_counts": counts,
                "near_start": near_start_c,
                "near_start_counts": {k: len(v) for k, v in sorted(near_start.items())},
                "near_end": near_end_c,
                "protected_areas": prot,
                "profile": prof,
                "popularity_proxies": {
                    "other_routes_on_same_ways": prof.pop("other_routes_on_same_ways"),
                    "pois_per_km": poi_per_km,
                    "parking_capacity_near_start": parking_cap,
                    "has_wikidata": "wikidata" in tags or "wikipedia" in tags,
                },
            },
            "osm_tags": osm_tags,
            "osm_length_m": _r(c["osm_length_m"]),
            "osm_ascent_m": _r(c["osm_ascent_m"]),
            "osm_descent_m": _r(c["osm_descent_m"]),
            "stitch_coverage": round(res.coverage, 3),
            "_quality": {
                "stitch_method": res.method,
                "stitch_gaps_bridged": res.n_gaps_bridged,
                "stitch_gap_m": _r(res.gap_m),
                "member_ways": res.n_input_ways,
                "ways_not_in_main_line": len(res.dropped_way_ids),
                "dem_void_share": round(c["void_share"], 4),
                "retrace_share": round(c["retrace"], 3),
            },
            "_internal": {"retrace_share": c["retrace"]},
        }
        rec["tags"] = tagmod.derive_tags(rec)
        rec["description"], rec["description_source"] = describe.compose(rec)
        del rec["_internal"]
        # OSM ids of the listed features and the stitching details go to the side file
        osm_ids = {}
        for k, lst in rec["features"]["on_route"].items():
            osm_ids[k] = [it.pop("osm") for it in lst]
        for k, lst in rec["features"]["near_start"].items():
            osm_ids["start:" + k] = [it.pop("osm") for it in lst]
        for k, lst in (rec["features"]["near_end"] or {}).items():
            osm_ids["end:" + k] = [it.pop("osm") for it in lst]
        for pa in rec["features"]["protected_areas"]:
            osm_ids.setdefault("protected_areas", []).append(pa.pop("osm"))
        details.append(
            {"id": rid, "feature_osm_ids": osm_ids, "quality": rec.pop("_quality"), "main_line_way_ids": res.way_ids}
        )
        records.append(rec)

        simp = simplify(coords, config.GEOMETRY_SIMPLIFY_TOL_M, proj)
        # coarse elevation profile every 200 m for UIs
        step = max(1, int(round(200 / config.DEM_SAMPLE_SPACING_M)))
        geoms.append(
            {
                "id": rid,
                "coords": [[round(x, config.COORD_DECIMALS), round(y, config.COORD_DECIMALS)] for x, y in simp],
                "ele_profile_m": [int(round(v)) for v in zs[::step]],
                "ele_profile_step_m": step * config.DEM_SAMPLE_SPACING_M,
            }
        )
    timings["records_s"] = round(time.time() - t0, 1)

    records.sort(key=lambda r: r["id"])
    geoms.sort(key=lambda g: g["id"])
    details.sort(key=lambda g: g["id"])

    # ---- validation ------------------------------------------------------------
    by_id = {r["id"]: r for r in records}
    valid_ids = {c["id"] for c in valid}
    calib_ids = {c["id"] for c in calib}
    asc_valid = [(by_id[i]["ascent_m"], by_id[i]["osm_ascent_m"]) for i in sorted(valid_ids)]
    asc_calib = [(by_id[i]["ascent_m"], by_id[i]["osm_ascent_m"]) for i in sorted(calib_ids)]
    len_pairs = [(r["length_m"], r["osm_length_m"]) for r in records if r["osm_length_m"] and r["osm_length_m"] >= 500]
    # raw (unsmoothed, no hysteresis) ascent for comparison
    raw_pairs = []
    for c in cands:
        if c["id"] in valid_ids:
            u, _ = demmod.ascent_descent(c["z"], 0.0)
            raw_pairs.append((u, c["osm_ascent_m"]))
    validation = {
        "ascent_vs_osm_tag_validation_half": error_summary(asc_valid),
        "ascent_vs_osm_tag_calibration_half": error_summary(asc_calib),
        "ascent_raw_dem_no_smoothing_validation_half": error_summary(raw_pairs),
        "length_vs_osm_distance_tag": error_summary(len_pairs),
        "dem_calibration": {"chosen": best, "grid": grid},
    }

    # ---- write -------------------------------------------------------------------
    config.CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    # gzip with mtime=0 so the bytes depend on the content only (31 MB of JSONL becomes about 5 MB in git)
    routes_path = config.CATALOG_DIR / "routes.jsonl.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        for rec in records:
            gz.write((json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    routes_path.write_bytes(buf.getvalue())
    geom_path = config.CATALOG_DIR / "geometry.jsonl.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        for g in geoms:
            gz.write((json.dumps(g, separators=(",", ":")) + "\n").encode())
    geom_path.write_bytes(buf.getvalue())
    det_path = config.CATALOG_DIR / "details.jsonl.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        for g in details:
            gz.write((json.dumps(g, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    det_path.write_bytes(buf.getvalue())

    dem_files = []
    for name in sorted(used_tiles):
        fn = used_tiles[name]
        if fn is None:
            dem_files.append({"tile": name, "available": False})
            continue
        p = config.DEM_DIR / fn
        dem_files.append(
            {
                "tile": name,
                "file": fn,
                "url": config.DEM_URL_TEMPLATE.format(lat_dir=name[:3], name=name),
                "sha256": download.file_hash(p),
                "bytes": p.stat().st_size,
            }
        )

    counts = {
        "relations_considered": len(ids),
        "dropped": {k: drops.get(k, 0) for k in DROP_REASONS},
        "final": len(records),
        "by_activity": dict(sorted(Counter(r["activity"] for r in records).items())),
        "by_category": dict(sorted(Counter(r["category_name"] for r in records).items())),
        "by_difficulty": {str(k): v for k, v in sorted(Counter(r["difficulty"] for r in records).items())},
    }
    manifest = {
        "catalogue": "outdoor routes, " + config.REGION_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "extract_version": EXTRACT_VERSION,
        "inputs": {
            "osm": {
                "file": pbf.name,
                "url": config.OSM_URL_TEMPLATE.format(name=args.extract),
                "md5": download.file_hash(pbf, "md5"),
                "sha256": download.file_hash(pbf),
                "bytes": pbf.stat().st_size,
                **osm_header,
                "licence": "ODbL 1.0, (c) OpenStreetMap contributors",
            },
            "dem": {
                "source": "Mapzen/Tilezen Terrain Tiles on AWS Open Data (skadi, 1 arc-second HGT), mostly SRTM",
                "tiles": dem_files,
                "licence": "see ATTRIBUTION.md (SRTM public domain courtesy of USGS; "
                "EU-DEM/Copernicus; Austria DGM, data.gv.at)",
            },
        },
        "region": {
            "name": config.REGION_NAME,
            "admin_level": config.REGION_ADMIN_LEVEL,
            "admin_names": list(config.REGION_ADMIN_NAMES),
            "area_km2": round(region.area / 1e6),
        },
        "parameters": {**config.parameters(), "DEM_SMOOTH_WINDOW_M_CHOSEN": win, "DEM_HYSTERESIS_M_CHOSEN": hyst},
        "counts": counts,
        "outputs": {
            "routes.jsonl.gz": {"sha256": download.file_hash(routes_path), "bytes": routes_path.stat().st_size},
            "geometry.jsonl.gz": {"sha256": download.file_hash(geom_path), "bytes": geom_path.stat().st_size},
            "details.jsonl.gz": {"sha256": download.file_hash(det_path), "bytes": det_path.stat().st_size},
        },
    }
    report = {
        "drop_counts": counts["dropped"],
        "drop_examples": {k: v[:15] for k, v in sorted(drop_ids.items())},
        "stitching": dict(stitch_stats),
        "dedup": {"pairs_compared": pairs_checked, "dropped": len(dup_of)},
        "region_selection_counts_by_start_admin5": dict(sorted(region_counts.items())),
        "validation": validation,
        "dem_void_samples": dem.void_samples,
        "dem_missing_tile_samples": dem.missing_tile_samples,
        "feature_index_counts": feat.n_by_kind,
        "timings_s": {**timings, "total_s": round(time.time() - t_start, 1)},
    }
    (config.CATALOG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (config.WORK_DIR / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    # the report without timings is deterministic and committed next to the catalogue
    rep_c = {k: v for k, v in report.items() if k != "timings_s"}
    (config.CATALOG_DIR / "build_report.json").write_text(json.dumps(rep_c, indent=2, ensure_ascii=False) + "\n")
    _write_attribution()
    _log(
        f"[done] {len(records)} routes -> {routes_path} ({routes_path.stat().st_size / 1e6:.1f} MB), "
        f"geometry {geom_path.stat().st_size / 1e6:.1f} MB, total {time.time() - t_start:.0f}s"
    )


def _max_grade(zs, d):
    if len(d) < 2 or d[-1] < 200:
        return 0.0
    step = float(np.median(np.diff(d)))
    k = max(1, int(round(200 / step)))
    if len(zs) <= k:
        return 0.0
    g = np.abs(zs[k:] - zs[:-k]) / np.maximum(d[k:] - d[:-k], 1.0)
    return float(np.max(g))


ATTRIBUTION = """# Data attribution and licences

## Route data: OpenStreetMap

Route relations, way attributes and points of interest are from
OpenStreetMap, (c) OpenStreetMap contributors, available under the
Open Database Licence (ODbL) 1.0: https://www.openstreetmap.org/copyright

The files in this directory (`routes.jsonl.gz`, `geometry.jsonl.gz`, `details.jsonl.gz`) are a
Derivative Database of OpenStreetMap and are themselves licensed under the
ODbL 1.0. Any public use must credit "(c) OpenStreetMap contributors" and
keep the data under the ODbL. Extract obtained from Geofabrik GmbH
(https://download.geofabrik.de/); the exact file and checksum are in
`manifest.json`.

Generated descriptions are derived from the same data and are part of the
derivative database.

## Elevation: Mapzen / Tilezen Terrain Tiles (AWS Open Data)

Elevation (ascent, descent, min/max elevation, profiles) is sampled from the
Terrain Tiles "skadi" 1 arc-second HGT tiles
(https://registry.opendata.aws/terrain-tiles/), licence and attribution as
documented at https://github.com/tilezen/joerd/blob/master/docs/attribution.md.
The tiles over Bavaria are composited mainly from SRTM; the documented
attribution for the sources that cover this region is:

* SRTM terrain data courtesy of the U.S. Geological Survey (public domain).
* Europe terrain data produced using Copernicus data and information funded
  by the European Union - EU-DEM layers.
* Austria terrain data (c) offene Daten Oesterreichs - Digitales
  Gelaendemodell (DGM) Oesterreich.
* Global GMTED2010 terrain data courtesy of the U.S. Geological Survey;
  ETOPO1 U.S. National Oceanic and Atmospheric Administration.
"""


def _write_attribution():
    (config.CATALOG_DIR / "ATTRIBUTION.md").write_text(ATTRIBUTION)


if __name__ == "__main__":
    main()
