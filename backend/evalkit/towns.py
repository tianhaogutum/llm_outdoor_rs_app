"""Build ``eval/personas/towns.json``: real Bavarian towns as persona homes.

One-off builder, run once and committed (the output is the artefact of record):

    cd backend && python -m evalkit.towns

It reads the pipeline's cached extract of the pinned Geofabrik snapshot
(``backend/data/raw/work/*.pkl``, produced by ``make data``) and keeps every OSM
``place=city|town`` node that has a ``name`` and a ``population`` tag and lies inside
the Bavarian boundary. Each town gets its Regierungsbezirk from the admin_level=5
polygons. Data (c) OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np

from evalkit import paths


def _population(v: str | None) -> int | None:
    if not v:
        return None
    digits = "".join(ch for ch in v.split(";")[0] if ch.isdigit())
    return int(digits) if digits else None


def build() -> list[dict]:
    import shapely
    from shapely.geometry import Point

    from pipeline import config
    from pipeline.areas import build_areas
    from pipeline.geo import LocalProjection
    from pipeline.osm_extract import NodeIndex

    pkl = config.WORK_DIR / f"{config.OSM_EXTRACT}.extract-v3.pkl"
    if not pkl.exists():
        sys.exit(f"{pkl} missing: run `make data` first (towns.json is committed, rebuilding is optional)")
    with pkl.open("rb") as fh:
        ex = pickle.load(fh)
    nodes = NodeIndex(ex["nodes"])
    proj = LocalProjection(48.5, 11.5)
    areas = build_areas(ex, nodes, proj, log=lambda *a: None)
    bavaria = shapely.unary_union([a["geom"] for a in areas["admin4"] if a["name"] in config.REGION_ADMIN_NAMES])
    shapely.prepare(bavaria)
    regions = sorted(areas["admin5"], key=lambda a: a["name"])
    for r in regions:
        shapely.prepare(r["geom"])

    pn = ex["poi_nodes"]
    out = []
    for nid, lon, lat, tags in zip(pn["id"], pn["lon"], pn["lat"], pn["tags"], strict=True):
        if tags.get("place") not in ("city", "town") or not tags.get("name"):
            continue
        pop = _population(tags.get("population"))
        if not pop:
            continue
        x, y = proj.fwd(np.array([lon]), np.array([lat]))
        pt = Point(float(x[0]), float(y[0]))
        if not bavaria.contains(pt):
            continue
        region = next((r["name"] for r in regions if r["geom"].contains(pt)), None)
        if region is None:
            continue
        out.append(
            {
                "name": tags["name"],
                "osm_node": int(nid),
                "place": tags["place"],
                "population": pop,
                "lat": round(float(lat), 5),
                "lon": round(float(lon), 5),
                "region": region,
            }
        )
    out.sort(key=lambda t: (t["region"], t["name"], t["osm_node"]))
    return out


def main() -> None:
    towns = build()
    paths.TOWNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "source": "OpenStreetMap place=city|town nodes with a population tag, inside Bayern (admin_level=4), "
        "from the pinned Geofabrik extract used for the catalogue. (c) OpenStreetMap contributors, ODbL 1.0.",
        "builder": "backend/evalkit/towns.py",
        "towns": towns,
    }
    paths.TOWNS_FILE.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    by_region: dict[str, list[int]] = {}
    for t in towns:
        by_region.setdefault(t["region"], []).append(t["population"])
    print(f"{len(towns)} towns -> {paths.TOWNS_FILE}")
    for r, pops in sorted(by_region.items()):
        print(f"  {r:16s} {len(pops):4d} towns, population {sum(pops):9d}")


if __name__ == "__main__":
    main()
