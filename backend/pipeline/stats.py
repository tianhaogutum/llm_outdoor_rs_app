"""Print every figure quoted in docs/DATA.md.

    cd backend && python -m pipeline.stats

Reads only the committed catalogue (routes.jsonl.gz, manifest.json,
build_report.json); no network, no rebuild.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter

import numpy as np

from . import config
from .tags import TAG_RULES

ON_ROUTE_REPORT = [
    "peak",
    "alpine_hut",
    "wilderness_hut",
    "restaurant",
    "beer_garden",
    "lake",
    "river",
    "waterfall",
    "gorge",
    "viewpoint",
    "castle",
    "ruins",
    "monastery",
    "church",
    "museum",
    "playground",
    "swimming",
    "aerialway_station",
    "drinking_water",
    "geological",
    "cave",
]


# hand-picked famous routes whose real length / profile is common knowledge
SPOT_CHECK_IDS = [1356555, 26937, 551290, 3190325, 223215, 3201006, 9379281, 37187]


def _pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "n/a"


def _dist(name, vals, unit=""):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if not len(a):
        print(f"  {name}: no values")
        return
    q = np.quantile(a, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print(
        f"  {name} (n={len(a)}){unit}: min {q[0]:.0f} | p10 {q[1]:.0f} | p25 {q[2]:.0f} | "
        f"median {q[3]:.0f} | p75 {q[4]:.0f} | p90 {q[5]:.0f} | max {q[6]:.0f} | mean {a.mean():.0f}"
    )


def _section(t):
    print(f"\n## {t}")


def main():
    with gzip.open(config.CATALOG_DIR / "routes.jsonl.gz", "rt", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh]
    man = json.loads((config.CATALOG_DIR / "manifest.json").read_text())
    rep = json.loads((config.CATALOG_DIR / "build_report.json").read_text())
    n = len(recs)

    _section("Inputs")
    o = man["inputs"]["osm"]
    print(f"  OSM extract: {o['file']}  md5 {o['md5']}  bytes {o['bytes']}")
    print(
        f"  replication timestamp: {o.get('osmosis_replication_timestamp')}  sequence: "
        f"{o.get('osmosis_replication_sequence_number')}"
    )
    tiles = man["inputs"]["dem"]["tiles"]
    print(f"  DEM tiles used: {len([t for t in tiles if t.get('file')])} ({', '.join(t['tile'] for t in tiles)})")
    print(
        f"  pipeline version: {man['pipeline_version']}  region: {man['region']['name']}  area "
        f"{man['region']['area_km2']} km2"
    )

    _section(
        "Region selection: routes passing all quality filters, by Regierungsbezirk of the start "
        "(before region filter and dedup)"
    )
    rc = rep["region_selection_counts_by_start_admin5"]
    for k, v in sorted(rc.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print(f"  Oberbayern + Schwaben (first-choice region): {rc.get('Oberbayern', 0) + rc.get('Schwaben', 0)}")

    _section("Filtering and dedup (route relations)")
    c = man["counts"]
    print(f"  relations considered: {c['relations_considered']}")
    for k, v in c["dropped"].items():
        print(f"  dropped {k}: {v}")
    print(f"  final routes: {c['final']}")
    s = rep["stitching"]
    att = s.get("assembled_attempts", 0)
    print(f"  stitching attempts (named, complete relations): {att}")
    print(
        f"    assembled in relation order as one segment: {s.get('ordered_single_segment', 0)} "
        f"({_pct(s.get('ordered_single_segment', 0), att)})"
    )
    print(
        f"    needed graph assembly: {s.get('graph_assembly', 0)} ({_pct(s.get('graph_assembly', 0), att)}); "
        f"of these accepted {s.get('graph_accepted', 0)}, with stray branches dropped "
        f"{s.get('accepted_with_dropped_branches', 0)}, "
        f"with bridged gaps {s.get('accepted_with_bridged_gaps', 0)}"
    )
    print(
        f"    failed (coverage < {config.STITCH_MIN_COVERAGE:.0%} or gaps too long): "
        f"{s.get('failed', 0)} ({_pct(s.get('failed', 0), att)})"
    )
    cov = [r["stitch_coverage"] for r in recs]
    print(
        f"  kept routes with stitch coverage < 1.0: {sum(1 for x in cov if x < 1)} "
        f"({_pct(sum(1 for x in cov if x < 1), n)}); median coverage of those: "
        f"{np.median([x for x in cov if x < 1]) if any(x < 1 for x in cov) else 1:.3f}"
    )
    print(f"  dedup pairs compared: {rep['dedup']['pairs_compared']}; duplicates dropped: {rep['dedup']['dropped']}")

    _section("Counts")
    for key in ("activity", "category_name", "difficulty", "difficulty_source", "description_source", "osm_route"):
        cnt = Counter(str(r[key]) for r in recs)
        print(f"  by {key}: " + ", ".join(f"{k} {v} ({_pct(v, n)})" for k, v in sorted(cnt.items())))
    print(f"  loops: {sum(r['is_loop'] for r in recs)} ({_pct(sum(r['is_loop'] for r in recs), n)})")
    print(
        f"  stages of a kept parent route: {sum(1 for r in recs if r['stage_of'])}; routes with "
        f"kept stages: {sum(1 for r in recs if r['stages'])}"
    )
    cross = Counter((r["activity"], r["difficulty"]) for r in recs)
    for act in sorted({r["activity"] for r in recs}):
        print(f"  difficulty for {act}: " + ", ".join(f"{d}: {cross.get((act, d), 0)}" for d in range(4)))

    _section("Distributions")
    for act in [None] + sorted({r["activity"] for r in recs}):
        sub = [r for r in recs if act is None or r["activity"] == act]
        print(f" [{act or 'all'}]")
        _dist("length km", [r["length_m"] / 1000 for r in sub])
        _dist("ascent m", [r["ascent_m"] for r in sub])
        _dist("duration min", [r["duration_min"] for r in sub])
        _dist("max elevation m", [r["max_ele_m"] for r in sub])
    _dist("description length (chars)", [len(r["description"]) for r in recs])
    _dist("tags per route", [len(r["tags"]) for r in recs])

    _section("Missing values (share of routes where the field is null/empty)")
    fields = [
        "locality",
        "end_locality",
        "admin_district",
        "ascent_m",
        "osm_ascent_m",
        "osm_length_m",
        "max_grade",
        "difficulty_technical",
    ]
    for f in fields:
        miss = sum(1 for r in recs if r.get(f) in (None, "", []))
        print(f"  {f}: {miss} ({_pct(miss, n)}); present {n - miss} ({_pct(n - miss, n)})")
    lin = [r for r in recs if not r["is_loop"]]
    print(f"  end_locality among linear routes: {sum(1 for r in lin if not r['end_locality'])} of {len(lin)} missing")
    for k in (
        "operator",
        "network",
        "osmc:symbol",
        "website",
        "wikidata",
        "description",
        "from",
        "to",
        "ref",
        "roundtrip",
    ):
        have = sum(1 for r in recs if k in r["osm_tags"])
        print(f"  osm tag {k}: present {have} ({_pct(have, n)})")
    st = [r["features"]["profile"]["surface_tagged_share"] for r in recs]
    print(
        f"  surface explicitly tagged: median share {np.median(st):.2f}; routes with >= 50% tagged: "
        f"{_pct(sum(x >= 0.5 for x in st), n)}"
    )
    sac = [r for r in recs if r["activity"] == "hiking"]
    print(
        "  hiking routes with any sac_scale grade >= 100 m: "
        f"{sum(1 for r in sac if r['difficulty_technical'] is not None)} of {len(sac)}"
    )

    _section("Tag frequencies")
    tc = Counter(t for r in recs for t in r["tags"])
    for t, legacy, _ in TAG_RULES:
        print(f"  {t}{' [legacy]' if legacy else ''}: {tc.get(t, 0)} ({_pct(tc.get(t, 0), n)})")
    print(
        "  routes with no tag besides shape: "
        f"{sum(1 for r in recs if set(r['tags']) <= {'Round trip', 'Linear route', 'Out and back'})}"
    )

    _section("Feature coverage (share of routes with >= 1 named/located feature of the type on the route)")
    for k in ON_ROUTE_REPORT:
        have = sum(1 for r in recs if r["features"]["on_route"].get(k))
        print(f"  {k}: {have} ({_pct(have, n)})")
    for k in ("train_station", "bus_stop", "tram_stop", "parking"):
        have = sum(1 for r in recs if r["features"]["near_start"].get(k))
        print(f"  near start {k}: {have} ({_pct(have, n)})")
    have = sum(1 for r in recs if r["features"]["protected_areas"])
    print(f"  crosses a protected area (>= 10% of length): {have} ({_pct(have, n)})")

    _section("Validation")
    v = rep["validation"]
    ch = v["dem_calibration"]["chosen"]
    print(
        f"  DEM parameters chosen on calibration half: smoothing window {ch['window_m']:.0f} m, "
        f"hysteresis {ch['hysteresis_m']:.0f} m "
        f"(calibration MdAPE {ch['median_abs_rel_err']:.3f})"
    )
    for key in (
        "ascent_vs_osm_tag_validation_half",
        "ascent_vs_osm_tag_calibration_half",
        "ascent_raw_dem_no_smoothing_validation_half",
        "length_vs_osm_distance_tag",
    ):
        e = v[key]
        if not e.get("n"):
            print(f"  {key}: n=0")
            continue
        print(
            f"  {key}: n={e['n']} median rel err {e['median_rel_err']:+.3f} | median |rel err| "
            f"{e['median_abs_rel_err']:.3f} | "
            f"p90 |rel err| {e['p90_abs_rel_err']:.3f} | within 10% {e['share_within_10pct']:.2f} | "
            f"within 20% {e['share_within_20pct']:.2f} | "
            f"median |abs err| {e['median_abs_err']:.0f} m"
        )
    print("  DEM grid (window m, hysteresis m -> calibration MdAPE):")
    for g in v["dem_calibration"]["grid"]:
        print(f"    {g['window_m']:.0f}, {g['hysteresis_m']:.0f} -> {g['median_abs_rel_err']:.3f}")
    print(f"  DEM void samples: {rep['dem_void_samples']}; samples on missing tiles: {rep['dem_missing_tile_samples']}")

    _section("Spot checks: well-known routes (relation id: title | length | ascent/descent | elevation range)")
    by_id = {r["id"]: r for r in recs}
    for rid in SPOT_CHECK_IDS:
        r = by_id.get(rid)
        if r is None:
            print(f"  {rid}: not in catalogue")
            continue
        print(
            f"  {rid}: {r['title']} | {r['length_m'] / 1000:.1f} km | +{r['ascent_m']} / -{r['descent_m']} m | "
            f"{r['min_ele_m']}-{r['max_ele_m']} m | loop {r['is_loop']} | difficulty "
            f"{r['difficulty']} | tags: {', '.join(r['tags'][:8])}"
        )
        pk = [
            f"{p['name']} (OSM ele {p['ele_m']} m)" for p in r["features"]["on_route"].get("peak", []) if p.get("ele_m")
        ]
        if pk:
            print(f"      peaks on route: {', '.join(pk[:3])}")

    _section("Files")
    for k, f in man["outputs"].items():
        print(f"  {k}: {f['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
