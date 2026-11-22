"""Per-route derived attributes: surface profile, difficulty, duration, category.

Every function here is pure (dicts and numbers in, dicts and numbers out) so
the rules can be unit-tested and are documented in one place.
"""

from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------
PAVED = {
    "asphalt",
    "paved",
    "concrete",
    "concrete:plates",
    "concrete:lanes",
    "paving_stones",
    "paving_stones:lanes",
    "sett",
    "cobblestone",
    "unhewn_cobblestone",
    "metal",
    "chipseal",
    "wood",
    "bricks",
    "brick",
}
UNPAVED = {
    "unpaved",
    "gravel",
    "fine_gravel",
    "compacted",
    "dirt",
    "earth",
    "ground",
    "grass",
    "mud",
    "sand",
    "rock",
    "rocks",
    "pebblestone",
    "woodchips",
    "grass_paver",
    "clay",
    "stepping_stones",
    "soil",
    "stone",
    "scree",
    "roots",
}
# smooth enough for a pushchair
STROLLER_OK = {
    "asphalt",
    "paved",
    "concrete",
    "concrete:plates",
    "paving_stones",
    "compacted",
    "fine_gravel",
    "chipseal",
    "concrete:lanes",
}
ROADS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "road",
}
BUSY_ROADS = {"trunk", "primary", "secondary", "trunk_link", "primary_link", "secondary_link"}
PATHS = {"path", "footway", "bridleway", "steps", "via_ferrata", "pedestrian", "corridor"}


def surface_class(tags: dict) -> tuple[str, bool]:
    """(paved | unpaved | unknown, explicit) for one way.

    Explicit = from the ``surface`` tag. Otherwise inferred from the highway
    type: roads and cycleways -> paved, tracktype grade1 -> paved, other
    tracks and paths -> unpaved, footways/steps/unknown -> unknown."""
    s = tags.get("surface")
    if s:
        s = s.split(";")[0].strip()
        if s in PAVED:
            return "paved", True
        if s in UNPAVED:
            return "unpaved", True
    hw = tags.get("highway")
    if hw in ROADS or hw in ("cycleway", "pedestrian", "living_street"):
        return "paved", False
    if hw == "track":
        return ("paved" if tags.get("tracktype") == "grade1" else "unpaved"), False
    if hw in ("path", "bridleway", "via_ferrata"):
        return "unpaved", False
    return "unknown", False


def stroller_ok(tags: dict) -> bool:
    if tags.get("highway") in ("steps", "via_ferrata") or tags.get("sac_scale") not in (None, "hiking"):
        return False
    s = (tags.get("surface") or "").split(";")[0]
    if s:
        return s in STROLLER_OK
    hw = tags.get("highway")
    return hw in ROADS or hw in ("cycleway", "pedestrian") or (hw == "track" and tags.get("tracktype") == "grade1")


SAC = {
    "hiking": 1,
    "mountain_hiking": 2,
    "demanding_mountain_hiking": 3,
    "alpine_hiking": 4,
    "demanding_alpine_hiking": 5,
    "difficult_alpine_hiking": 6,
}
SAC_LABEL = {v: f"T{v}" for v in SAC.values()}


def sac_level(tags: dict) -> int | None:
    if tags.get("highway") == "via_ferrata" or "via_ferrata_scale" in tags:
        return 5
    return SAC.get((tags.get("sac_scale") or "").strip())


def mtb_level(tags: dict) -> int | None:
    v = tags.get("mtb:scale")
    if not v:
        return None
    m = re.match(r"\s*(\d)", v)
    return int(m.group(1)) if m else None


def way_profile(way_tags: list[dict], way_len: list[float]) -> dict:
    """Length-weighted profile over the member ways of the main chain."""
    total = sum(way_len) or 1.0
    acc = {"paved": 0.0, "unpaved": 0.0, "unknown": 0.0}
    explicit = 0.0
    stroller = 0.0
    roads = busy = paths = tracks = cycleways = steps = 0.0
    sac_len: dict[int, float] = {}
    mtb_len: dict[int, float] = {}
    sac_tagged = mtb_tagged = 0.0
    dog_no = dog_yes = 0.0
    lit = 0.0
    via_ferrata = 0.0
    for t, L in zip(way_tags, way_len, strict=True):
        c, ex = surface_class(t)
        acc[c] += L
        if ex:
            explicit += L
        if stroller_ok(t):
            stroller += L
        hw = t.get("highway")
        if hw in ROADS:
            roads += L
        if hw in BUSY_ROADS:
            busy += L
        if hw in PATHS:
            paths += L
        if hw == "track":
            tracks += L
        if hw == "cycleway":
            cycleways += L
        if hw == "steps":
            steps += L
        s = sac_level(t)
        if s is not None:
            sac_len[s] = sac_len.get(s, 0.0) + L
            sac_tagged += L
        m = mtb_level(t)
        if m is not None:
            mtb_len[m] = mtb_len.get(m, 0.0) + L
            mtb_tagged += L
        if t.get("dog") == "no":
            dog_no += L
        elif t.get("dog") in ("yes", "leashed"):
            dog_yes += L
        if t.get("lit") == "yes":
            lit += L
        if hw == "via_ferrata" or "via_ferrata_scale" in t:
            via_ferrata += L

    def r(v):
        return round(v / total, 3)

    return {
        "paved_share": r(acc["paved"]),
        "unpaved_share": r(acc["unpaved"]),
        "unknown_surface_share": r(acc["unknown"]),
        "surface_tagged_share": r(explicit),
        "stroller_ok_share": r(stroller),
        "road_share": r(roads),
        "busy_road_share": r(busy),
        "path_share": r(paths),
        "track_share": r(tracks),
        "cycleway_share": r(cycleways),
        "steps_m": round(steps),
        "via_ferrata_m": round(via_ferrata),
        "sac_tagged_share": r(sac_tagged),
        "sac_m": {SAC_LABEL[k]: round(v) for k, v in sorted(sac_len.items())},
        "mtb_tagged_share": r(mtb_tagged),
        "mtb_scale_m": {f"S{k}": round(v) for k, v in sorted(mtb_len.items())},
        "dog_no_share": r(dog_no),
        "dog_yes_share": r(dog_yes),
        "lit_share": r(lit),
    }


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------
THEME_RE = re.compile(
    r"(lehrpfad|themenweg|erlebnisweg|erlebnispfad|kulturweg|kulturpfad|geschichtsweg|historischer weg|"
    r"sagenweg|planetenweg|skulpturenweg|kunstweg|wissensweg|besinnungsweg|kreuzweg|meditationsweg|"
    r"barfu(ß|ss)pfad|naturerlebnis|geologischer|geopfad|walderlebnis|bienenweg|märchenweg|"
    r"literaturweg|lyrikweg|philosophenweg|lehrweg|infopfad|pfad der|weg der|themenpfad|kunstpfad|"
    r"klimaweg|moorpfad|moorlehrpfad|wasserweg|mühlenweg|schmugglerweg|archäolog|römerweg|"
    r"grenzweg|bierweg|brauereiweg|schlösserweg|burgenweg|kapellenweg|keltenweg)",
    re.I,
)
# pilgrimage routes (Jakobsweg, Via Claudia, ...) are not theme trails; they get
# the "Pilgrimage route" tag instead


def category(
    activity: str, route_tag: str, name: str, tags: dict, length_m: float, unpaved_share: float, ascent_m: float
) -> str:
    """App category vocabulary (CATEGORY_MAPPING in recommendation_service)."""
    if activity == "hiking":
        if (
            THEME_RE.search(name or "")
            or tags.get("educational") == "yes"
            or tags.get("tourism") == "yes"
            and "lehr" in (name or "").lower()
        ):
            return "Theme trail"
        return "Hiking trail"
    if activity == "cycling":
        if length_m >= 100_000 or tags.get("network") in ("icn", "ncn"):
            return "Long distance cycling"
        return "Cycling"
    if activity == "mtb":
        return "Mountainbiking"
    # running
    if unpaved_share >= 0.6 and (ascent_m >= 150 or route_tag == "running" and unpaved_share >= 0.8):
        return "Trail running"
    return "Jogging"


# ---------------------------------------------------------------------------
# Difficulty (0..3)
# ---------------------------------------------------------------------------
# effort index per activity: km + ascent_m / K, thresholds for levels 1, 2, 3
EFFORT_MODEL = {
    "hiking": {"k": 100.0, "thresholds": (12.0, 22.0, 32.0)},  # "Leistungskilometer"
    "running": {"k": 100.0, "thresholds": (8.0, 15.0, 25.0)},
    "cycling": {"k": 30.0, "thresholds": (35.0, 70.0, 110.0)},
    "mtb": {"k": 20.0, "thresholds": (30.0, 50.0, 75.0)},
}
MIN_TECH_LENGTH_M = 100.0  # a grade must cover >= 100 m to count


def effort_index(activity: str, length_m: float, ascent_m: float) -> float:
    return length_m / 1000.0 + (ascent_m or 0.0) / EFFORT_MODEL[activity]["k"]


def effort_level(activity: str, length_m: float, ascent_m: float) -> int:
    e = effort_index(activity, length_m, ascent_m)
    return sum(e >= t for t in EFFORT_MODEL[activity]["thresholds"])


def technical_level(activity: str, profile: dict) -> tuple[int | None, str | None]:
    """Technical grade from OSM way tags (0..3) or None when untagged.

    hiking/running: highest SAC grade covering >= 100 m: T1 -> 0, T2 -> 1,
    T3 -> 2, T4..T6 and via ferrata -> 3. mtb/cycling: highest mtb:scale
    covering >= 100 m: S0 -> 0, S1 -> 1, S2 -> 2, S3+ -> 3."""
    if activity in ("hiking", "running"):
        grades = [int(k[1:]) for k, v in profile["sac_m"].items() if v >= MIN_TECH_LENGTH_M]
        if not grades:
            return None, None
        g = max(grades)
        return {1: 0, 2: 1, 3: 2}.get(g, 3), f"T{g}"
    grades = [int(k[1:]) for k, v in profile["mtb_scale_m"].items() if v >= MIN_TECH_LENGTH_M]
    if not grades:
        return None, None
    g = max(grades)
    return min(g, 3), f"S{g}"


def difficulty(activity: str, length_m: float, ascent_m: float, profile: dict, descent_m: float | None = None) -> dict:
    """Difficulty 0..3 = max(technical grade, effort level).

    The effort uses the larger of ascent and descent: a linear route mapped
    summit-down is as hard as the same route mapped valley-up."""
    eff = effort_level(activity, length_m, max(ascent_m or 0.0, descent_m or 0.0))
    tech, grade = technical_level(activity, profile)
    if tech is None:
        return {
            "difficulty": eff,
            "difficulty_source": "derived:effort",
            "difficulty_technical": None,
            "difficulty_effort": eff,
            "max_grade": None,
        }
    key = "sac_scale" if activity in ("hiking", "running") else "mtb:scale"
    if tech >= eff:
        src = f"osm:{key}"
    else:
        src = f"derived:effort (osm:{key} lower)"
    return {
        "difficulty": max(tech, eff),
        "difficulty_source": src,
        "difficulty_technical": tech,
        "difficulty_effort": eff,
        "max_grade": grade,
    }


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------
SPEED_MODEL = {
    # activity: (flat km/h, ascent m/h added on top, descent m/h added on top)
    "cycling": (18.0, 600.0, None),
    "mtb": (12.0, 400.0, None),
    "running": (9.0, 600.0, None),
}


def duration_min(activity: str, length_m: float, ascent_m: float, descent_m: float) -> int:
    """Hiking: DIN 33466 (4 km/h, 300 m/h up, 500 m/h down; the smaller of
    horizontal and vertical time is halved and added to the larger).
    Others: additive speed model, t = km / v_flat + ascent / v_up."""
    km = length_m / 1000.0
    up = ascent_m or 0.0
    down = descent_m or 0.0
    if activity == "hiking":
        th = km / 4.0
        tv = up / 300.0 + down / 500.0
        h = max(th, tv) + min(th, tv) / 2.0
    else:
        v, vu, _ = SPEED_MODEL[activity]
        h = km / v + up / vu
    return int(round(h * 60))


# ---------------------------------------------------------------------------
# Raw OSM numbers
# ---------------------------------------------------------------------------
_NUM = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*([a-zA-Z]*)")


def parse_length_tag(v: str | None) -> float | None:
    """OSM distance/length tag -> metres. Bare numbers are km (wiki convention)."""
    if not v:
        return None
    m = _NUM.search(v.replace(" ", " "))
    if not m:
        return None
    x = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    if unit in ("", "km", "kms", "k"):
        return x * 1000.0
    if unit == "m":
        return x
    if unit in ("mi", "mile", "miles"):
        return x * 1609.344
    return None


def parse_ele_tag(v: str | None) -> float | None:
    """OSM ascent/descent tag -> metres (bare numbers are metres)."""
    if not v:
        return None
    m = _NUM.search(v)
    if not m:
        return None
    x = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    if unit in ("", "m", "hm"):
        return x
    if unit == "km":
        return x * 1000.0
    return None


def est_days(duration_minutes: int, activity: str) -> int:
    per_day = {"hiking": 8 * 60, "running": 3 * 60, "cycling": 6 * 60, "mtb": 5 * 60}[activity]
    return max(1, math.ceil(duration_minutes / per_day))
