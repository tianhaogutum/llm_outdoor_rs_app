"""Controlled tag vocabulary and the rule behind every tag.

``TAG_RULES`` is the single source of truth: the tag name, whether it is a
legacy tag from the original Outdooractive catalogue, and a human-readable
rule. ``derive_tags`` implements exactly those rules over a route record.
No tag is assigned without a fact (an OSM tag, a feature within a distance
threshold, or a number computed from geometry/elevation).
"""

from __future__ import annotations

import re

# (tag, legacy?, rule)
TAG_RULES: list[tuple[str, bool, str]] = [
    ("Round trip", True, "start and end within 300 m of each other and the line does not mostly retrace itself"),
    (
        "Out and back",
        True,
        "start and end within 300 m and >= 70% of the line runs within 30 m of another part of "
        "itself >= 300 m away along the line",
    ),
    ("Linear route", True, "start and end more than 300 m apart"),
    ("Scenic", True, "at least one of: viewpoint, summit, lake, waterfall, gorge or glacier on the route"),
    (
        "Refreshment stops available",
        True,
        "a restaurant, cafe, pub, beer garden, ice-cream shop or alpine hut within its on-route "
        "distance (75 m; huts 100 m)",
    ),
    (
        "Family-friendly",
        True,
        "derived heuristic: hiking/running <= 7 km, ascent <= 250 m, difficulty 0, no SAC grade "
        "above T2, < 10% on busy roads, and at least one of playground, swimming spot, lake, "
        "waterfall, zoo/wildlife park, picnic site, refreshment stop or hut on the route; cycling "
        "<= 30 km, ascent <= 300 m, difficulty <= 1, < 5% on busy roads, and the same amenity condition; never for MTB",
    ),
    (
        "Cultural/historical interest",
        True,
        "a castle/palace, ruins, monastery, museum, archaeological site or historic monument "
        "within its on-route distance",
    ),
    (
        "Geological highlights",
        True,
        "a gorge, cave entrance, or named geological feature (geological=*, rock, stone, arch, "
        "cliff, sinkhole, 'Geotop') on the route",
    ),
    (
        "Flora",
        True,
        ">= 30% of the route inside a nature reserve or national park, a named natural-monument "
        "tree on the route, or a theme-trail name about plants/forest/moor",
    ),
    ("Fauna", True, "a zoo/wildlife park or bird hide on the route, or a theme-trail name about animals"),
    ("Summit route", True, "a named peak within 75 m of the route"),
    ("Dog-friendly", True, "dog=yes/leashed on the relation or on >= 50% of the route's ways, and dog=no on none"),
    (
        "Suitable for strollers",
        True,
        "hiking/running, >= 90% of the length on smooth surfaces (asphalt, paving, compacted, fine "
        "gravel), no steps, ascent <= 200 m, length <= 15 km",
    ),
    (
        "Multi-stage route",
        True,
        "relation has >= 2 stage sub-relations, or a hiking route whose DIN 33466 time exceeds one 8-hour day",
    ),
    ("Singletrail/Free ride", True, "MTB route with >= 25% of its length on paths (highway=path/footway/bridleway)"),
    ("Cableway ascent/descent", True, "an aerialway station within 150 m of the route"),
    ("Accessibility", True, "wheelchair=yes on the relation"),
    # --- new tags ---
    ("Lake", False, "a named lake/reservoir/pond >= 1 ha within 100 m of the route"),
    ("River", False, "the route runs within 60 m of a named river for >= 1 km"),
    ("Waterfall", False, "a waterfall within 150 m of the route"),
    ("Gorge", False, "natural=gorge, or a named attraction/valley called *klamm/*schlucht/*tobel, within 150 m"),
    ("Cave", False, "a cave entrance within 100 m"),
    ("Glacier", False, "a glacier within 300 m"),
    ("Hut", False, "an alpine hut (100 m) or wilderness hut (75 m) on the route"),
    ("Beer garden", False, "a beer garden within 75 m"),
    ("Viewpoint", False, "a viewpoint within 50 m"),
    ("Castle", False, "a castle, palace, manor or fort within 150 m"),
    ("Ruins", False, "ruins within 100 m"),
    ("Monastery", False, "a monastery within 150 m"),
    ("Museum", False, "a museum within 100 m"),
    ("Church/chapel", False, "a named place of worship within 75 m"),
    ("Playground", False, "a playground within 75 m"),
    ("Swimming spot", False, "a bathing place, swimming area or beach within 150 m"),
    ("Picnic area", False, "a picnic site within 50 m"),
    ("Drinking water", False, "a drinking-water point within 50 m"),
    (
        "Nature reserve",
        False,
        ">= 10% of the route inside a nature reserve (leisure=nature_reserve or protect_class 1/4)",
    ),
    ("National park", False, ">= 10% of the route inside a national park"),
    ("Train access", False, "a railway station or halt within 1 km of the start"),
    ("Bus access", False, "a bus stop within 300 m of the start"),
    ("Public transport access", False, "train within 1 km, tram/subway within 500 m or bus within 300 m of the start"),
    ("Parking at start", False, "a public parking area within 300 m of the start"),
    (
        "Alpine terrain",
        False,
        "a highest point >= 1600 m, a via ferrata, or SAC grade T3 or harder on >= 100 m with a "
        "highest point >= 1000 m",
    ),
    ("Via ferrata", False, "a via ferrata section (highway=via_ferrata or via_ferrata_scale) on the route"),
    (
        "Mostly paved",
        False,
        ">= 70% of the length on paved surfaces (explicit surface tag, else inferred from highway type)",
    ),
    ("Mostly unpaved", False, ">= 70% of the length on unpaved surfaces (same inference)"),
    ("Busy road sections", False, ">= 10% of the length on trunk/primary/secondary roads"),
    ("Flat", False, "ascent and descent both <= 150 m and <= 10 m per km"),
    ("Big climb", False, "ascent or descent >= 1000 m (linear routes can be walked either way)"),
    ("Waymarked", False, "the relation carries a waymark symbol (osmc:symbol or symbol)"),
    (
        "Pilgrimage route",
        False,
        "name contains Jakobsweg, Pilgerweg, Wallfahrt, Via ... Romea/Claudia, Camino or Kreuzweg, or pilgrimage=yes",
    ),
]
TAG_NAMES = [t for t, _, _ in TAG_RULES]
LEGACY_TAGS_NOT_ASSIGNED = {
    "Competition track": "no OSM fact distinguishes race courses on route relations",
    "Insider tip": "editorial judgement, not derivable from data",
}

FLORA_RE = re.compile(
    r"(botan|pflanz|kräuter|blumen|baum|bäume|wald|forst|moor|orchid|enzian|obst|streuobst|wiese)", re.I
)
FAUNA_RE = re.compile(
    r"(vogel|vögel|tier|wildpark|wildtier|bienen|fisch|schmetterling|biber|insekt|amphib|frosch|wild)", re.I
)
PILGRIM_RE = re.compile(
    r"(jakobsweg|pilgerweg|pilgerpfad|wallfahrt|via romea|via claudia|camino|kreuzweg|münchner jakobsweg)", re.I
)


def derive_tags(r: dict) -> list[str]:
    """r: the route record (fields as written to routes.jsonl, plus ``_internal``)."""
    f = r["features"]
    on = f.get("on_route", {})
    start = f.get("near_start", {})
    prof = f["profile"]
    rel = r["osm_tags"]
    internal = r["_internal"]
    act = r["activity"]
    tags: set[str] = set()

    def has(*kinds):
        return any(on.get(k) for k in kinds)

    # shape
    if r["is_loop"]:
        tags.add("Out and back" if internal["retrace_share"] >= 0.70 else "Round trip")
    else:
        tags.add("Linear route")

    if has("lake"):
        tags.add("Lake")
    if has("river"):
        tags.add("River")
    if has("waterfall"):
        tags.add("Waterfall")
    if has("gorge"):
        tags.add("Gorge")
    if has("cave"):
        tags.add("Cave")
    if has("glacier"):
        tags.add("Glacier")
    if has("viewpoint"):
        tags.add("Viewpoint")
    if has("peak"):
        tags.add("Summit route")
    if has("viewpoint", "peak", "lake", "waterfall", "gorge", "glacier"):
        tags.add("Scenic")
    if has("alpine_hut", "wilderness_hut"):
        tags.add("Hut")
    if has("beer_garden"):
        tags.add("Beer garden")
    if has("restaurant", "beer_garden", "alpine_hut"):
        tags.add("Refreshment stops available")
    if has("castle"):
        tags.add("Castle")
    if has("ruins"):
        tags.add("Ruins")
    if has("monastery"):
        tags.add("Monastery")
    if has("museum"):
        tags.add("Museum")
    if has("church"):
        tags.add("Church/chapel")
    if has("castle", "ruins", "monastery", "museum", "archaeological_site", "monument"):
        tags.add("Cultural/historical interest")
    if has("gorge", "cave", "geological"):
        tags.add("Geological highlights")
    if has("playground"):
        tags.add("Playground")
    if has("swimming"):
        tags.add("Swimming spot")
    if has("picnic_site"):
        tags.add("Picnic area")
    if has("drinking_water"):
        tags.add("Drinking water")
    if has("aerialway_station"):
        tags.add("Cableway ascent/descent")

    prot = f.get("protected_areas", [])
    nr_share = sum(p["share"] for p in prot if p["kind"] == "nature_reserve")
    np_share = sum(p["share"] for p in prot if p["kind"] == "national_park")
    if nr_share >= 0.10:
        tags.add("Nature reserve")
    if np_share >= 0.10:
        tags.add("National park")

    title = r["title"]
    theme = r["category_name"] == "Theme trail"
    if nr_share + np_share >= 0.30 or has("natural_monument_tree") or (theme and FLORA_RE.search(title)):
        tags.add("Flora")
    if has("zoo", "bird_hide") or (theme and FAUNA_RE.search(title)):
        tags.add("Fauna")

    # access
    if start.get("train_station"):
        tags.add("Train access")
    if start.get("bus_stop"):
        tags.add("Bus access")
    if start.get("train_station") or start.get("bus_stop") or start.get("tram_stop"):
        tags.add("Public transport access")
    if start.get("parking"):
        tags.add("Parking at start")

    # terrain
    sac = prof["sac_m"]
    hard = sum(v for k, v in sac.items() if int(k[1:]) >= 3)
    max_ele = r["max_ele_m"] if r["max_ele_m"] is not None else 0
    if max_ele >= 1600 or prof.get("via_ferrata_m", 0) > 0 or (hard >= 100 and max_ele >= 1000):
        tags.add("Alpine terrain")
    if prof.get("via_ferrata_m", 0) > 0:
        tags.add("Via ferrata")
    if prof["paved_share"] >= 0.70:
        tags.add("Mostly paved")
    if prof["unpaved_share"] >= 0.70:
        tags.add("Mostly unpaved")
    if prof["busy_road_share"] >= 0.10:
        tags.add("Busy road sections")

    km = r["length_m"] / 1000.0
    asc = r["ascent_m"] if r["ascent_m"] is not None else None
    if asc is not None:
        desc = r["descent_m"] or 0
        if max(asc, desc) <= 150 and max(asc, desc) <= 10 * km:
            tags.add("Flat")
        if max(asc, desc) >= 1000:
            tags.add("Big climb")

    # suitability (derived heuristics)
    family_amenity = has(
        "playground", "swimming", "lake", "waterfall", "zoo", "picnic_site", "restaurant", "beer_garden", "alpine_hut"
    )
    if asc is not None:
        if act in ("hiking", "running"):
            if (
                km <= 7
                and asc <= 250
                and r["difficulty"] == 0
                and prof["busy_road_share"] < 0.10
                and family_amenity
                and not any(int(k[1:]) >= 3 and v >= 100 for k, v in sac.items())
            ):
                tags.add("Family-friendly")
            if prof["stroller_ok_share"] >= 0.90 and prof["steps_m"] == 0 and asc <= 200 and km <= 15:
                tags.add("Suitable for strollers")
        elif act == "cycling":
            if km <= 30 and asc <= 300 and r["difficulty"] <= 1 and prof["busy_road_share"] < 0.05 and family_amenity:
                tags.add("Family-friendly")

    if act == "mtb" and prof["path_share"] >= 0.25:
        tags.add("Singletrail/Free ride")

    dog_rel = rel.get("dog")
    if (
        prof["dog_no_share"] == 0
        and (dog_rel in ("yes", "leashed") or prof["dog_yes_share"] >= 0.5)
        and dog_rel != "no"
    ):
        tags.add("Dog-friendly")
    if rel.get("wheelchair") == "yes":
        tags.add("Accessibility")
    if len(r.get("stages") or []) >= 2 or (act == "hiking" and r.get("est_days", 1) >= 2):
        tags.add("Multi-stage route")
    if rel.get("osmc:symbol") or rel.get("symbol"):
        tags.add("Waymarked")
    if PILGRIM_RE.search(title) or rel.get("pilgrimage") == "yes":
        tags.add("Pilgrimage route")

    unknown = tags - set(TAG_NAMES)
    assert not unknown, unknown
    return [t for t in TAG_NAMES if t in tags]
