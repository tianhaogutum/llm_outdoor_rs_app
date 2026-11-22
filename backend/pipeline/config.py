"""All pinned inputs and generation parameters in one place.

Every value here is copied into ``manifest.json`` so a catalogue can be traced
back to the exact inputs and thresholds that produced it.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OSM_DIR = RAW_DIR / "osm"
DEM_DIR = RAW_DIR / "dem"
WORK_DIR = RAW_DIR / "work"
CATALOG_DIR = DATA_DIR / "catalog"

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
# Geofabrik keeps dated snapshots next to "-latest": daily ones for ~10 days,
# monthly ones (YYMM01) for a few months, and the yearly 1 January snapshot
# (YY0101) indefinitely (back to 2014). We pin the 1 November 2026 monthly
# snapshot. Override with PIPELINE_OSM_EXTRACT=bayern-260101 etc.
OSM_EXTRACT = os.environ.get("PIPELINE_OSM_EXTRACT", "bayern-261101")
OSM_URL_TEMPLATE = "https://download.geofabrik.de/europe/germany/bayern/{name}.osm.pbf"
# md5 published by Geofabrik next to the file (bayern-261101.osm.pbf.md5).
OSM_EXPECTED_MD5 = {"bayern-261101": "e081102904686f1af63e61fdbb341c09"}

# Mapzen/Tilezen terrain tiles, "skadi" format = 1x1 degree SRTM-style HGT,
# 1 arc-second (3601x3601 int16 big-endian), gzip-compressed. Static since 2016.
DEM_URL_TEMPLATE = "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{lat_dir}/{name}.hgt.gz"

# --------------------------------------------------------------------------
# Region
# --------------------------------------------------------------------------
# The region is the union of the named OSM admin boundaries at REGION_ADMIN_LEVEL.
# First choice was the Bavarian Alps and foothills (admin_level 5 Oberbayern +
# Schwaben), but only ~2.2k routes there pass the quality filters (see
# build_report.json, region_selection_counts_by_start_admin5), so the region
# is the whole Free State of Bavaria (admin_level 4).
REGION_NAME = "Bavaria (Freistaat Bayern)"
REGION_ADMIN_LEVEL = "4"
REGION_ADMIN_NAMES = ("Bayern",)
# A route is in the region if its start is inside and at least this share of
# its length lies inside the region polygon.
REGION_MIN_INSIDE_SHARE = 0.75

# --------------------------------------------------------------------------
# Route selection
# --------------------------------------------------------------------------
ROUTE_TYPES = {
    "hiking": "hiking",
    "foot": "hiking",
    "walking": "hiking",
    "nordic_walking": "hiking",
    "bicycle": "cycling",
    "mtb": "mtb",
    "running": "running",
    "fitness_trail": "running",
}
# Member roles that are part of the main line. Others (alternative, excursion,
# approach, connection, platform, stop, guidepost, ...) are not stitched.
MAIN_ROLES = {"", "main", "forward", "backward", "route", "section"}
# Lifecycle states that mean "not a usable route".
INACTIVE_STATES = {"proposed", "disused", "abandoned", "construction", "planned"}

# Length limits per activity (m): shorter = fragment, longer = not a
# recommendable day / multi-day route for this catalogue.
LENGTH_LIMITS_M = {
    "hiking": (1_000, 400_000),
    "running": (1_000, 60_000),
    "cycling": (3_000, 800_000),
    "mtb": (3_000, 300_000),
}

# --------------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------------
STITCH_GAP_TOLERANCE_M = 50.0  # endpoints closer than this are joined
STITCH_MIN_COVERAGE = 0.90  # main chain must cover >= 90% of member length
STITCH_MAX_GAP_SHARE = 0.02  # sum of bridged gaps <= max(2% of line length,
STITCH_MAX_GAP_M = 100.0  #                            100 m)
LOOP_MAX_ENDPOINT_DIST_M = 300.0  # start-end distance for a loop
OUT_AND_BACK_MIN_OVERLAP = 0.70  # share of the line retraced by the line itself
OUT_AND_BACK_TOL_M = 30.0

# --------------------------------------------------------------------------
# Elevation
# --------------------------------------------------------------------------
DEM_SAMPLE_SPACING_M = 20.0  # resample the line at this spacing
# Smoothing window and hysteresis are not fixed here: build.calibrate_dem picks
# them from a grid on half of the routes that carry an OSM ascent tag
# (calibration split) and reports the error on the other half. The chosen
# values are written to manifest.json (DEM_*_CHOSEN).

# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------
DEDUP_MATCH_TOL_M = 40.0  # point-to-line distance that counts as "same path"
DEDUP_MIN_OVERLAP = 0.90  # both directions of overlap must reach this

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
GEOMETRY_SIMPLIFY_TOL_M = 10.0  # Douglas-Peucker tolerance of the stored polyline
COORD_DECIMALS = 5  # ~1 m
FEATURE_LIST_CAP = 8  # max named items stored per feature type


def parameters() -> dict:
    """Snapshot of generation parameters for the manifest."""
    keys = [k for k in globals() if k.isupper() and not k.endswith("_DIR")]
    out = {}
    for k in sorted(keys):
        v = globals()[k]
        if isinstance(v, (set, frozenset)):
            v = sorted(v)
        elif isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, Path):
            continue
        out[k] = v
    return out
