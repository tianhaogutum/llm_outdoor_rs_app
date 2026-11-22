"""Plain-text cards: the persona as the judge (and the owner) sees it, and a factual route card.

Both are deterministic functions of the data, so prompts and cache keys are stable.
"""

from __future__ import annotations

from typing import Any

from app.recsys import RouteRecord, haversine_km
from evalkit.personas import DESIRE_LABELS, PARTY_TEXT, Persona

TECH_TEXT = {
    "hiking": {
        0: "easy paths only (SAC T1)",
        1: "mountain paths up to SAC T2",
        2: "up to SAC T3 (sure-footed, some exposure)",
        3: "alpine routes up to SAC T4 and beyond",
    },
    "mtb": {0: "MTB S0 only", 1: "up to MTB S1", 2: "up to MTB S2", 3: "MTB S3 and harder"},
    "touring": {
        0: "roads, cycleways and good tracks only; no technical trails",
        1: "easy trails at most (S0/S1)",
        2: "up to S2",
        3: "S3+",
    },
}
SHAPE_TEXT = {
    "loop": "prefers a loop (back to the car / start)",
    "linear": "prefers a one-way (linear) route",
    "any": "no preference between loop and one-way",
}
SURFACE_TEXT = {
    "paved": "prefers paved or smooth surfaces",
    "natural": "prefers natural paths and trails",
    "any": "no surface preference",
}
DISLIKE_TEXT = {
    "alpine_terrain": "alpine terrain (high, exposed, via ferrata)",
    "busy_roads": "riding or walking on busy roads",
    "crowds": "crowded, popular trails",
    "steep_sections": "steep sections",
}
FITNESS_WORDS = [(0.75, "low"), (1.5, "moderate"), (2.25, "good"), (9.0, "very fit")]


def fitness_word(f: float) -> str:
    return next(w for lim, w in FITNESS_WORDS if f < lim)


def party_text(party: dict[str, Any]) -> str:
    s = PARTY_TEXT[party["kind"]]
    extra = []
    if party.get("children_ages"):
        ages = ", ".join("baby" if a == 0 else str(a) for a in party["children_ages"])
        extra.append(f"children aged {ages}")
    if party.get("dog") and party["kind"] != "with_dog":
        extra.append("bringing a dog")
    extra.append(f"{party.get('adults', 1)} adult(s), age {party.get('age_band')}")
    return f"{s} ({'; '.join(extra)})"


def latent_lines(p: Persona, *, canonical: bool = False) -> list[str]:
    """The persona's full latent profile as short factual lines (judge card, intent writer, owner sheet).

    ``canonical=True`` lists the wishes in key order, which is the order they have after a round trip
    through ``personas.jsonl`` (written with sorted keys); the judge and owner cards use it so their text
    does not depend on in-memory dict order. The intent writer saw the generator's order (by weight)."""
    x = p.latent
    desires = dict(sorted(x["desires"].items())) if canonical else x["desires"]
    style = x["style"]
    act = {
        "hiking": "hiking / walking",
        "touring": "cycling (touring bike: roads, cycleways, tracks)",
        "mtb": "mountain biking",
    }[style]
    lines = [
        f"Activity: {act}",
        f"Home: {x['home']['town']} ({x['home']['region']}, Bavaria); travels by {x['transport']} "
        f"at most {x['max_travel_km']:.0f} km (straight line) to the start",
        f"Party: {party_text(x['party'])}",
        f"Fitness: {fitness_word(x['fitness'])}",
    ]
    lo, hi = x["distance_band_km"]
    if x["multi_day"]:
        lines.append(
            f"Trip: multi-day tour over about {x['days']} days; comfortable total {lo:.0f}-{hi:.0f} km, "
            f"absolutely no more than {x['max_distance_km']:.0f} km"
        )
    else:
        lines.append(
            f"Distance: comfortable {lo:.0f}-{hi:.0f} km, absolutely no more than {x['max_distance_km']:.0f} km"
        )
    lines.append(f"Climbing: comfortable with up to about {x['max_ascent_m']} m of total ascent")
    lines.append(f"Technical level: {TECH_TEXT[style][x['technical_max']]}")
    wants = [
        f"{DESIRE_LABELS.get(k, k.replace('_', ' '))} ({'main wish' if w >= 0.6 else 'nice to have'})"
        for k, w in desires.items()
        if k not in ("train_access", "multi_day")
    ]
    if wants:
        lines.append("Wishes: " + "; ".join(wants))
    if x["desires"].get("train_access"):
        lines.append("Needs to reach the start by train (no car)")
    lines.append(f"Route shape: {SHAPE_TEXT[x['shape']]}")
    lines.append(f"Surface: {SURFACE_TEXT[x['surface']]}")
    if x["dislikes"]:
        lines.append("Dislikes: " + "; ".join(DISLIKE_TEXT[d] for d in x["dislikes"]))
    return lines


def persona_card(p: Persona, *, include_intent: bool = True, canonical: bool = True) -> str:
    lines = latent_lines(p, canonical=canonical)
    if include_intent and p.intent:
        lines.append(f'What they typed into the app: "{p.intent}"')
    return "\n".join(lines)


# --------------------------------------------------------------------------- route card
FEATURE_GROUPS = [
    ("peak", "summits"),
    ("viewpoint", "viewpoints"),
    ("lake", "lakes"),
    ("river", "rivers followed"),
    ("waterfall", "waterfalls"),
    ("gorge", "gorges"),
    ("cave", "caves"),
    ("geological", "rock features"),
    ("alpine_hut", "alpine huts"),
    ("wilderness_hut", "unstaffed huts"),
    ("restaurant", "restaurants/cafes/inns"),
    ("beer_garden", "beer gardens"),
    ("castle", "castles/palaces"),
    ("ruins", "ruins"),
    ("monastery", "monasteries"),
    ("museum", "museums"),
    ("church", "churches/chapels"),
    ("playground", "playgrounds"),
    ("swimming", "swimming spots"),
    ("zoo", "zoo/wildlife park"),
    ("bird_hide", "bird hides"),
    ("picnic_site", "picnic sites"),
    ("aerialway_station", "cable car stations"),
]


def _names(items: list[dict], n: int = 3) -> str:
    names = [i.get("name") for i in items if i.get("name")]
    out = []
    for nm in names:
        if nm not in out:
            out.append(nm)
    return ", ".join(out[:n])


def route_card(r: RouteRecord, home: tuple[float, float] | None = None, *, osm_desc_chars: int = 280) -> str:
    f = r.features or {}
    prof = f.get("profile") or {}
    lines = [f"Title: {r.title}"]
    shape = "loop" if r.is_loop else f"one-way, ends near {r.end_locality or 'elsewhere'}"
    lines.append(f"Type: {r.category_name} ({r.activity}); {shape}")
    start = f"Start: near {r.locality or '?'} ({r.admin_district or '?'}, {r.admin_region or '?'})"
    if home is not None:
        km = haversine_km(home[0], home[1], r.start_lat, r.start_lon)
        start += f"; {km:.0f} km from the person's home (straight line)"
    lines.append(start)
    dur_h = r.duration_min / 60
    days = f", about {r.est_days} days" if r.est_days > 1 else ""
    lines.append(
        f"Length {r.length_km:.1f} km; ascent {r.ascent_m} m, descent {r.descent_m} m; highest point {r.max_ele_m} m; "
        f"estimated moving time {dur_h:.1f} h{days}"
    )
    grade = f", hardest graded section {r.max_grade}" if r.max_grade else ", no technical grade mapped"
    lines.append(f"Difficulty {r.difficulty} on 0-3 (0 easy, 3 very demanding){grade}")
    surf = (
        f"Surface: paved {prof.get('paved_share', 0):.0%}, unpaved {prof.get('unpaved_share', 0):.0%}; "
        f"paths {prof.get('path_share', 0):.0%}, tracks {prof.get('track_share', 0):.0%}, "
        f"roads {prof.get('road_share', 0):.0%} (busy roads {prof.get('busy_road_share', 0):.0%})"
    )
    if prof.get("steps_m"):
        surf += f"; steps {prof['steps_m']} m"
    if prof.get("via_ferrata_m"):
        surf += f"; via ferrata {prof['via_ferrata_m']} m"
    lines.append(surf)
    on = f.get("on_route") or {}
    counts = f.get("on_route_counts") or {}
    feats = []
    for key, label in FEATURE_GROUPS:
        n = counts.get(key, 0)
        if not n:
            continue
        nm = _names(on.get(key, []))
        feats.append(f"{label}: {n}" + (f" ({nm})" if nm else ""))
    lines.append("Near the line: " + ("; ".join(feats) if feats else "nothing notable mapped"))
    prot = [a["name"] for a in f.get("protected_areas") or [] if a.get("kind") in ("nature_reserve", "national_park")]
    if prot:
        lines.append("Protected: " + ", ".join(prot[:2]))
    ns = f.get("near_start") or {}
    acc = []
    if ns.get("train_station"):
        t = ns["train_station"][0]
        acc.append(f"train station {t.get('name') or ''} {t['dist_m']} m".replace("  ", " "))
    if ns.get("bus_stop"):
        acc.append(f"bus stop {ns['bus_stop'][0]['dist_m']} m")
    if ns.get("parking"):
        acc.append(f"parking {ns['parking'][0]['dist_m']} m")
    lines.append("At the start: " + (", ".join(acc) if acc else "no station, stop or parking mapped nearby"))
    lines.append("Tags: " + ", ".join(r.tags))
    if r.osm_description:
        d = " ".join(r.osm_description.split())
        if len(d) > osm_desc_chars:
            d = d[: osm_desc_chars - 3].rstrip() + "..."
        lines.append(f"Mapper's note (OSM): {d}")
    return "\n".join(lines)
