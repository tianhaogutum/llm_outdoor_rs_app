"""Controlled vocabularies the chat may write: activities, bands, tags, clearable slots.

The tag list is the catalogue's own vocabulary (``pipeline.tags.TAG_NAMES``, docs/DATA.md section 7), so
a tag the model proposes can only ever be one that routes actually carry. Everything here is plain data
(no I/O) and is what both the prompt and the validator read, so they cannot drift apart.
"""

from __future__ import annotations

from typing import Optional

from pipeline.tags import TAG_NAMES

ACTIVITIES: tuple[str, ...] = ("hiking", "cycling", "mtb", "running")
# Activity -> UI tab / hard-filter category (recsys.constraints.CATEGORY_TABS). MTB routes live in the
# cycling tab ("Mountainbiking" is one of its category names).
ACTIVITY_TAB: dict[str, str] = {"hiking": "hiking", "cycling": "cycling", "mtb": "cycling", "running": "running"}

DIFFICULTY_BANDS: tuple[str, ...] = ("easy", "moderate", "hard")
# Same ranges as the questionnaire's fitness levels (user_profile_service.FITNESS_MAPPING), 0-3 scale.
DIFFICULTY_RANGES: dict[str, list[int]] = {"easy": [0, 1], "moderate": [1, 2], "hard": [2, 3]}
FITNESS_TO_DIFFICULTY: dict[str, str] = {"beginner": "easy", "intermediate": "moderate", "advanced": "hard"}

DISTANCE_BANDS: tuple[str, ...] = ("short", "medium", "long")
# Hiking ranges are the questionnaire's (beginner 0-8, intermediate 5-20, advanced 10-50 km); other
# activities scale them, because "a short ride" is not 8 km.
HIKING_DISTANCE_RANGES: dict[str, tuple[float, float]] = {"short": (0.0, 8.0), "medium": (5.0, 20.0),
                                                          "long": (10.0, 50.0)}
DISTANCE_SCALE: dict[str, float] = {"hiking": 1.0, "running": 1.0, "cycling": 4.0, "mtb": 2.5}
# Planning speeds (km/h) for turning a stated duration into a distance band (DATA.md section 11 uses
# 4 / 18 / 12 / 9 km/h on the flat; rounded down a little for breaks and climbs).
PLANNING_SPEED_KMH: dict[str, float] = {"hiking": 4.0, "running": 9.0, "cycling": 16.0, "mtb": 10.0}

DIFFICULTY_CHANGES: tuple[str, ...] = ("easier", "harder")
DISTANCE_CHANGES: tuple[str, ...] = ("shorter", "longer")
LANGUAGES: tuple[str, ...] = ("en", "de")

TAGS: tuple[str, ...] = tuple(TAG_NAMES)
_TAG_BY_FOLD: dict[str, str] = {t.casefold(): t for t in TAGS}

# Slots a message (or a chip in the UI) may clear explicitly.
CLEARABLE: tuple[str, ...] = (
    "activity", "difficulty", "distance_band", "duration_hours", "distance_cap_km", "place",
    "travel_radius_km", "tags", "intent",
)

# Short glosses for the prompt: what each tag means in user terms (the rule text is in DATA.md).
TAG_GLOSS: dict[str, str] = {
    "Round trip": "loop, ends where it starts",
    "Out and back": "there and back on the same way",
    "Linear route": "A to B, start and end differ",
    "Scenic": "views, summit, lake, waterfall or gorge on the way",
    "Refreshment stops available": "somewhere to eat or drink on the way (inn, cafe, hut)",
    "Family-friendly": "short, easy, with something for kids",
    "Cultural/historical interest": "castles, ruins, monasteries, museums, monuments",
    "Geological highlights": "gorge, cave, rock formations",
    "Flora": "nature reserve, notable trees, plant/forest/moor theme trail",
    "Fauna": "wildlife park, zoo, bird watching",
    "Summit route": "reaches a named peak",
    "Dog-friendly": "dogs explicitly allowed",
    "Suitable for strollers": "smooth surface, no steps, little climbing (pram, buggy, Kinderwagen)",
    "Multi-stage route": "several days or stages",
    "Singletrail/Free ride": "MTB singletrails",
    "Cableway ascent/descent": "cable car / chairlift on the route",
    "Accessibility": "wheelchair accessible",
    "Lake": "lake or reservoir", "River": "along a river", "Waterfall": "waterfall", "Gorge": "gorge, Klamm",
    "Cave": "cave", "Glacier": "glacier", "Hut": "mountain hut (Hütte, Alm)", "Beer garden": "beer garden",
    "Viewpoint": "viewpoint", "Castle": "castle or palace", "Ruins": "ruins", "Monastery": "monastery, abbey",
    "Museum": "museum", "Church/chapel": "church or chapel", "Playground": "playground",
    "Swimming spot": "place to swim, bathing spot", "Picnic area": "picnic site", "Drinking water": "fountain",
    "Nature reserve": "nature reserve", "National park": "national park",
    "Train access": "train station near the start", "Bus access": "bus stop at the start",
    "Public transport access": "reachable by public transport (train, tram or bus)",
    "Parking at start": "car park at the start", "Alpine terrain": "high alpine, exposed terrain",
    "Via ferrata": "via ferrata, Klettersteig", "Mostly paved": "mostly asphalt",
    "Mostly unpaved": "mostly gravel / trail", "Busy road sections": "sections on busy roads (usually avoided)",
    "Flat": "almost no climbing", "Big climb": "1000 m or more of climbing", "Waymarked": "signposted / waymarked",
    "Pilgrimage route": "pilgrimage way (Jakobsweg)",
}


def canonical_tag(value: object) -> Optional[str]:
    """The vocabulary spelling of ``value`` (case-insensitive), or None if it is not a vocabulary tag."""
    if not isinstance(value, str):
        return None
    return _TAG_BY_FOLD.get(" ".join(value.split()).casefold())


def distance_range(band: str, activity: Optional[str]) -> tuple[float, float]:
    """``[min, max]`` km of a distance band for an activity (hiking scale when the activity is unknown)."""
    lo, hi = HIKING_DISTANCE_RANGES[band]
    k = DISTANCE_SCALE.get(activity or "hiking", 1.0)
    return (round(lo * k, 1), round(hi * k, 1))


def band_for_km(km: float, activity: Optional[str]) -> str:
    """Distance band whose range is closest to ``km`` (ties go to the shorter band)."""
    best, best_d = DISTANCE_BANDS[0], float("inf")
    for band in DISTANCE_BANDS:
        lo, hi = distance_range(band, activity)
        mid = (lo + hi) / 2
        d = 0.0 if lo <= km <= hi else min(abs(km - lo), abs(km - hi))
        d = d * 1000 + abs(km - mid)  # inside a range beats outside; then the nearer centre wins
        if d < best_d:
            best, best_d = band, d
    return best


def band_for_range(lo: Optional[float], hi: Optional[float], activity: Optional[str]) -> Optional[str]:
    """The band whose range equals ``[lo, hi]`` exactly (how a questionnaire vector maps back), else None."""
    if lo is None or hi is None:
        return None
    for band in DISTANCE_BANDS:
        blo, bhi = distance_range(band, activity)
        if abs(blo - float(lo)) < 1e-6 and abs(bhi - float(hi)) < 1e-6:
            return band
    return None


def difficulty_for_range(rng: object) -> Optional[str]:
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        return None
    for band, (lo, hi) in DIFFICULTY_RANGES.items():
        if rng[0] == lo and rng[1] == hi:
            return band
    return None
