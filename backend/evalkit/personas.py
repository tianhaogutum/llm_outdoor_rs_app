"""Simulated users ("personas") for offline evaluation.

One seeded generator produces ``eval/personas/personas.jsonl``::

    cd backend && python -m evalkit.personas            # LLM intents come from the cache on reruns

Each persona has three layers:

- ``latent``: the structured truth about this simulated person (activity and taste, home town, travel
  radius, comfortable distance band, absolute distance cap, ascent and technical tolerance, party,
  weighted desires, dislikes). **Never shown to a system.** The judge sees it.
- ``questionnaire`` + ``user_vector``: the 3-question form answered the way a real person would,
  coarsely and with noise; the vector is exactly what ``translate_questionnaire_to_vector`` returns.
- ``profile``: slots a product form could plausibly collect (home, travel radius, distance cap); the
  hard filter may use them.
- ``intent``: one or two natural sentences written by an LLM from the latent profile (cached).

Running is not simulated: the catalogue has 55 running routes (docs/DATA.md section 12).
See docs/EVAL.md for the sampling model and its limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from app.recsys import Catalog, HardConstraints, UserQuery, get_catalog
from evalkit import paths

GENERATOR_VERSION = "personas/v1"

# --------------------------------------------------------------------------- parameters
# Every number that shapes the population lives here and is written to generator_params.json.
PARAMS: dict[str, Any] = {
    "version": GENERATOR_VERSION,
    "seed": 20261122,
    "n_personas": 300,
    "n_test": 100,
    # home towns: OSM towns/cities with population; weight = population ** town_weight_exponent.
    # 1.0 would put about a fifth of all personas in Munich; 0.5 keeps every region represented
    # while still favouring large towns.
    "town_weight_exponent": 0.5,
    "activity": {"hiking": 0.68, "cycling": 0.32},
    "cycling_style": {"touring": 0.72, "mtb": 0.28},
    "party": {
        "hiking": {
            "solo": 0.17,
            "couple": 0.22,
            "friends": 0.11,
            "family_young": 0.10,
            "family_school": 0.10,
            "stroller": 0.05,
            "with_dog": 0.13,
            "older": 0.12,
        },
        "touring": {
            "solo": 0.20,
            "couple": 0.28,
            "friends": 0.12,
            "family_young": 0.06,
            "family_school": 0.16,
            "older": 0.18,
        },
        "mtb": {"solo": 0.45, "friends": 0.37, "couple": 0.12, "family_teen": 0.06},
    },
    # true fitness on the 0-3 route-difficulty scale: normal(mean, sd) clipped
    "fitness": {
        "solo": [1.7, 0.7],
        "couple": [1.4, 0.6],
        "friends": [1.9, 0.6],
        "family_young": [1.0, 0.45],
        "family_school": [1.3, 0.5],
        "family_teen": [1.8, 0.5],
        "stroller": [0.7, 0.4],
        "with_dog": [1.4, 0.6],
        "older": [0.9, 0.5],
    },
    "mtb_fitness_bonus": 0.4,
    # comfortable distance band centre (km) = a + b * fitness, band = [0.6 c, 1.3 c]
    "distance_centre": {"hiking": [4.0, 4.5], "touring": [15.0, 22.0], "mtb": [12.0, 12.0]},
    "distance_centre_cap": {
        "family_young": {"hiking": 5.5, "touring": 18.0},
        "stroller": {"hiking": 7.0},
        "family_school": {"hiking": 10.0, "touring": 35.0},
        "older": {"hiking": 12.0, "touring": 50.0},
    },
    "distance_cap_factor": [1.25, 1.7],
    "multi_day_prob": {"hiking": 0.05, "touring": 0.10, "mtb": 0.0},
    # comfortable total ascent (m) = a + b * fitness
    "ascent": {"hiking": [150, 380], "touring": [120, 330], "mtb": [300, 450]},
    "ascent_cap": {"family_young": 250, "stroller": 150, "family_school": 450, "older": 600},
    "train_user_prob": {"default": 0.15, "city_over_100k": 0.35},
    "travel_radius_km": {
        "car": [[20, 30, 40, 60, 80], [0.12, 0.25, 0.30, 0.22, 0.11]],
        "train": [[25, 40, 60], [0.3, 0.45, 0.25]],
    },
    "min_eligible": 40,
    "min_activity_match": 10,
    "radius_widen_factor": 1.25,
    "radius_max_km": 150,
    # desire propensities (relative); 2-4 primary desires, 0-2 secondary ones
    "desire_propensity": {
        "hiking": {
            "lake": 1.0,
            "refreshment": 0.9,
            "summit": 0.9,
            "viewpoint": 1.0,
            "waterfall_gorge": 0.5,
            "quiet_nature": 0.9,
            "culture": 0.6,
            "river": 0.35,
            "cave_geology": 0.3,
            "hut": 0.6,
            "beer_garden": 0.5,
            "swimming": 0.15,
            "wildlife": 0.15,
        },
        "touring": {
            "river": 1.0,
            "lake": 0.9,
            "culture": 0.9,
            "beer_garden": 0.9,
            "refreshment": 0.7,
            "quiet_nature": 0.5,
            "viewpoint": 0.3,
            "swimming": 0.3,
        },
        "mtb": {
            "summit": 0.9,
            "quiet_nature": 0.8,
            "viewpoint": 0.7,
            "hut": 0.6,
            "refreshment": 0.3,
            "lake": 0.3,
            "cave_geology": 0.2,
        },
    },
    "party_desire_boost": {
        "family_young": {
            "playground": 1.5,
            "lake": 1.2,
            "swimming": 0.7,
            "wildlife": 0.8,
            "refreshment": 1.0,
            "summit": 0.1,
            "hut": 0.2,
        },
        "family_school": {
            "playground": 0.9,
            "lake": 1.1,
            "swimming": 0.7,
            "wildlife": 0.6,
            "cave_geology": 0.6,
            "waterfall_gorge": 0.6,
            "culture": 0.5,
        },
        "stroller": {
            "lake": 1.2,
            "refreshment": 1.2,
            "playground": 0.8,
            "summit": 0.0,
            "hut": 0.0,
            "waterfall_gorge": 0.2,
        },
        "older": {"refreshment": 1.4, "culture": 1.2, "lake": 0.9, "viewpoint": 0.9, "beer_garden": 0.8, "summit": 0.3},
        "with_dog": {"quiet_nature": 1.4, "lake": 0.9, "river": 0.9, "refreshment": 0.6},
        "friends": {"beer_garden": 1.0, "hut": 0.9, "summit": 0.9},
    },
    "shape": {
        "default": {"loop": 0.45, "linear": 0.08, "any": 0.47},
        "family": {"loop": 0.75, "linear": 0.02, "any": 0.23},
        "train": {"loop": 0.25, "linear": 0.30, "any": 0.45},
    },
    # huts are an Alpine thing: outside these regions the hut propensity is scaled down
    "alpine_regions": ["Oberbayern", "Schwaben"],
    "hut_propensity_outside_alps": 0.2,
    "questionnaire_noise": {"overestimate": 0.17, "underestimate": 0.08, "select_all_types": 0.08},
    "intent_styles": {"vague": 0.18, "short": 0.27, "medium": 0.35, "detailed": 0.20},
    "intent_mentions_home": 0.4,
    "intent_voices": {"casual": 0.3, "practical": 0.25, "enthusiastic": 0.15, "terse": 0.15, "polite": 0.15},
}

DESIRE_LABELS = {
    "lake": "a lake to walk or ride along",
    "swimming": "a place to swim",
    "refreshment": "somewhere to eat or drink along the way (inn, cafe)",
    "hut": "a mountain hut stop",
    "beer_garden": "a beer garden",
    "summit": "reaching a summit",
    "viewpoint": "good views / a viewpoint",
    "waterfall_gorge": "a waterfall or gorge",
    "cave_geology": "caves or interesting rocks",
    "quiet_nature": "quiet nature, forest, few people",
    "culture": "culture: castles, ruins, monasteries, museums",
    "river": "following a river",
    "playground": "a playground for the kids",
    "wildlife": "seeing animals (wildlife park, bird watching)",
}

PARTY_TEXT = {
    "solo": "on their own",
    "couple": "a couple",
    "friends": "a group of friends",
    "family_young": "a family with young children",
    "family_school": "a family with school-age children",
    "family_teen": "a parent with a teenager",
    "stroller": "parents pushing a stroller",
    "with_dog": "with a dog",
    "older": "older walkers (retired)",
}

FAMILY_PARTIES = {"family_young", "family_school", "family_teen", "stroller"}
NATURE_DESIRES = {
    "lake",
    "summit",
    "viewpoint",
    "waterfall_gorge",
    "quiet_nature",
    "river",
    "cave_geology",
    "wildlife",
    "swimming",
}


# --------------------------------------------------------------------------- helpers
def _choice(rng: np.random.Generator, table: dict[str, float]) -> str:
    keys = list(table)
    p = np.array([table[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _r(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def load_towns() -> list[dict]:
    return json.loads(paths.TOWNS_FILE.read_text(encoding="utf-8"))["towns"]


@dataclass
class Persona:
    persona_id: str
    split: str
    latent: dict[str, Any]
    questionnaire: dict[str, Any]
    user_vector: dict[str, Any]
    profile: dict[str, Any]
    intent: Optional[str] = None
    intent_meta: Optional[dict[str, Any]] = None
    eligible: Optional[dict[str, Any]] = None

    # what a system may see
    def to_query(self) -> UserQuery:
        return UserQuery(
            request_id=self.persona_id,
            user_vector=self.user_vector,
            category=self.latent["category"],
            home=tuple(self.profile["home"]),
            max_travel_km=self.profile["max_travel_km"],
            max_distance_km=self.profile["max_distance_km"],
            intent=self.intent,
        )

    def constraints(self) -> HardConstraints:
        return HardConstraints.from_query(self.to_query())

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "split": self.split,
            "latent": self.latent,
            "questionnaire": self.questionnaire,
            "user_vector": self.user_vector,
            "profile": self.profile,
            "intent": self.intent,
            "intent_meta": self.intent_meta,
            "eligible": self.eligible,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Persona:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


def load_personas(path=None) -> list[Persona]:
    with open(path or paths.PERSONAS_FILE, encoding="utf-8") as fh:
        return [Persona.from_dict(json.loads(line)) for line in fh if line.strip()]


# --------------------------------------------------------------------------- latent sampling
def _sample_party(rng, style: str) -> dict[str, Any]:
    kind = _choice(rng, PARAMS["party"][style])
    party: dict[str, Any] = {"kind": kind, "children_ages": [], "dog": kind == "with_dog", "stroller": False}
    if kind == "family_young":
        n = int(rng.choice([1, 2], p=[0.5, 0.5]))
        party["children_ages"] = sorted(int(a) for a in rng.integers(3, 7, size=n))
    elif kind == "family_school":
        n = int(rng.choice([1, 2, 3], p=[0.4, 0.45, 0.15]))
        party["children_ages"] = sorted(int(a) for a in rng.integers(7, 14, size=n))
    elif kind == "family_teen":
        party["children_ages"] = [int(rng.integers(12, 17))]
    elif kind == "stroller":
        party["stroller"] = True
        party["children_ages"] = [0] + ([int(rng.integers(3, 6))] if rng.random() < 0.35 else [])
    if kind in ("couple", "family_school") and rng.random() < 0.12:
        party["dog"] = True
    party["adults"] = 1 if kind in ("solo", "family_teen") else (int(rng.integers(3, 6)) if kind == "friends" else 2)
    if kind == "with_dog":
        party["adults"] = int(rng.choice([1, 2]))
    party["age_band"] = (
        "65+"
        if kind == "older"
        else str(rng.choice(["30-45", "35-50"]))
        if kind in FAMILY_PARTIES
        else str(rng.choice(["18-29", "30-44", "45-64"], p=[0.35, 0.4, 0.25]))
    )
    return party


def _sample_desires(rng, style: str, party_kind: str, region: str) -> dict[str, float]:
    prop = dict(PARAMS["desire_propensity"][style])
    for k, v in PARAMS["party_desire_boost"].get(party_kind, {}).items():
        if style == "mtb" and k in ("playground", "swimming"):
            continue
        prop[k] = v if k not in prop else prop[k] * 0.3 + v
    if region not in PARAMS["alpine_regions"] and "hut" in prop:
        prop["hut"] *= PARAMS["hut_propensity_outside_alps"]
    keys = [k for k, v in prop.items() if v > 0]
    p = np.array([prop[k] for k in keys])
    n_primary = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
    n_secondary = int(rng.choice([0, 1, 2], p=[0.3, 0.45, 0.25]))
    n_total = min(len(keys), n_primary + n_secondary)
    picked = rng.choice(len(keys), size=n_total, replace=False, p=p / p.sum())
    desires = {}
    for j, i in enumerate(picked):
        w = rng.uniform(0.6, 1.0) if j < n_primary else rng.uniform(0.2, 0.5)
        desires[keys[int(i)]] = _r(w, 2)
    return dict(sorted(desires.items(), key=lambda kv: -kv[1]))


def sample_latent(rng: np.random.Generator, towns: list[dict], weights: np.ndarray) -> dict[str, Any]:
    activity = _choice(rng, PARAMS["activity"])
    style = "hiking" if activity == "hiking" else _choice(rng, PARAMS["cycling_style"])
    town = towns[int(rng.choice(len(towns), p=weights))]
    party = _sample_party(rng, style)
    kind = party["kind"]

    mu, sd = PARAMS["fitness"][kind]
    fitness = _clip(rng.normal(mu + (PARAMS["mtb_fitness_bonus"] if style == "mtb" else 0.0), sd), 0.0, 3.0)

    # distance band and cap
    a, b = PARAMS["distance_centre"][style]
    centre = a + b * fitness
    cap_c = PARAMS["distance_centre_cap"].get(kind, {}).get(style)
    if cap_c:
        centre = min(centre, cap_c)
    centre *= float(np.exp(rng.normal(0, 0.15)))
    multi_day = bool(fitness >= 1.3 and kind not in FAMILY_PARTIES and rng.random() < PARAMS["multi_day_prob"][style])
    if multi_day:
        days = int(rng.choice([2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.1]))
        band = [centre * 0.8 * max(days - 1, 1), centre * 1.2 * days]
        cap = centre * 1.3 * (days + 1)
    else:
        days = 1
        band = [centre * 0.6, centre * 1.3]
        cap = band[1] * rng.uniform(*PARAMS["distance_cap_factor"])

    a, b = PARAMS["ascent"][style]
    max_ascent = (a + b * fitness) * float(np.exp(rng.normal(0, 0.2)))
    if kind in PARAMS["ascent_cap"]:
        max_ascent = min(max_ascent, PARAMS["ascent_cap"][kind])
    if multi_day:
        max_ascent *= days

    # technical tolerance on the 0-3 scale (SAC T1..T4+ for hiking, S0..S3+ for MTB)
    if style == "touring" or kind in FAMILY_PARTIES or kind == "older":
        technical = 0 if style != "mtb" else 1
    elif style == "mtb":
        technical = int(_clip(round(0.7 + 0.5 * fitness + rng.normal(0, 0.5)), 0, 3))
    else:
        technical = int(_clip(round(0.6 * fitness - 0.3 + rng.normal(0, 0.5)), 0, 3))

    p_train = PARAMS["train_user_prob"]["city_over_100k" if town["population"] > 100_000 else "default"]
    transport = "train" if rng.random() < p_train else "car"
    vals, probs = PARAMS["travel_radius_km"][transport]
    radius = float(rng.choice(vals, p=probs))

    desires = _sample_desires(rng, style, kind, town["region"])
    if transport == "train":
        desires["train_access"] = 1.0
    if multi_day:
        desires["multi_day"] = 1.0

    shape_table = PARAMS["shape"][
        "family" if kind in FAMILY_PARTIES or party["dog"] else "train" if transport == "train" else "default"
    ]
    shape = _choice(rng, shape_table)
    if multi_day:
        shape = "linear" if rng.random() < 0.6 else "any"

    if party["stroller"]:
        surface = "paved"
    elif style == "mtb":
        surface = "natural"
    elif style == "touring":
        surface = (
            "paved" if rng.random() < (0.6 if kind in ("older", "family_young", "family_school") else 0.35) else "any"
        )
    else:
        surface = "natural" if rng.random() < 0.3 else "any"

    dislikes = []
    if kind in FAMILY_PARTIES or kind == "older" or technical == 0 or rng.random() < 0.15:
        if style != "mtb":
            dislikes.append("alpine_terrain")
    if style == "touring" and (kind in FAMILY_PARTIES or kind == "older" or rng.random() < 0.35):
        dislikes.append("busy_roads")
    elif style == "hiking" and rng.random() < 0.1:
        dislikes.append("busy_roads")
    if "quiet_nature" in desires and desires["quiet_nature"] >= 0.6:
        dislikes.append("crowds")
    if style == "hiking" and kind in ("older", "stroller") or (style == "touring" and kind == "older"):
        dislikes.append("steep_sections")

    return {
        "activity": activity,
        "category": "hiking" if activity == "hiking" else "cycling",
        "style": style,  # hiking | touring | mtb
        "home": {
            "town": town["name"],
            "region": town["region"],
            "lat": town["lat"],
            "lon": town["lon"],
            "population": town["population"],
            "osm_node": town["osm_node"],
        },
        "transport": transport,
        "max_travel_km": radius,
        "radius_widened": False,
        "party": party,
        "fitness": _r(fitness, 2),
        "distance_band_km": [_r(band[0]), _r(band[1])],
        "max_distance_km": _r(cap),
        "multi_day": multi_day,
        "days": days,
        "max_ascent_m": int(round(max_ascent / 10.0) * 10),
        "technical_max": technical,
        "desires": desires,
        "shape": shape,
        "surface": surface,
        "dislikes": sorted(set(dislikes)),
    }


# --------------------------------------------------------------------------- questionnaire
def answer_questionnaire(rng: np.random.Generator, latent: dict[str, Any]) -> dict[str, Any]:
    f = latent["fitness"]
    level = 0 if f < 1.0 else 1 if f < 2.0 else 2
    noise = PARAMS["questionnaire_noise"]
    u = rng.random()
    misjudged = 0
    if u < noise["overestimate"]:
        misjudged = 1
    elif u < noise["overestimate"] + noise["underestimate"]:
        misjudged = -1
    level = int(_clip(level + misjudged, 0, 2))
    fitness = ["beginner", "intermediate", "advanced"][level]

    d = latent["desires"]
    kind = latent["party"]["kind"]
    types = []
    culture = d.get("culture", 0.0)
    if (culture >= 0.5 and rng.random() < 0.85) or (culture == 0 and rng.random() < 0.08):
        types.append("history-culture")
    nature = max([w for k, w in d.items() if k in NATURE_DESIRES] or [0.0])
    if (nature >= 0.6 and rng.random() < 0.85) or (nature < 0.6 and rng.random() < 0.25):
        types.append("natural-scenery")
    if (kind in FAMILY_PARTIES and rng.random() < 0.85) or (kind not in FAMILY_PARTIES and rng.random() < 0.04):
        types.append("family-fun")
    if rng.random() < noise["select_all_types"]:
        types = ["history-culture", "natural-scenery", "family-fun"]
    if not types:
        types = ["natural-scenery"]
    order = ["history-culture", "natural-scenery", "family-fun"]
    types = [t for t in order if t in types]

    if kind in FAMILY_PARTIES:
        narrative = _choice(rng, {"playful": 0.6, "adventure": 0.25, "mystery": 0.15})
    else:
        narrative = _choice(rng, {"adventure": 0.45, "mystery": 0.3, "playful": 0.25})
    return {"fitness": fitness, "type": types, "narrative": narrative, "fitness_misjudged": misjudged}


def questionnaire_vector(q: dict[str, Any]) -> dict[str, Any]:
    """Exactly the product's translation of the 3-question form."""
    from app.api.schemas import ProfileCreate
    from app.services.user_profile_service import translate_questionnaire_to_vector

    return translate_questionnaire_to_vector(
        ProfileCreate(fitness=q["fitness"], type=q["type"], narrative=q["narrative"])
    )


# --------------------------------------------------------------------------- eligibility
def activity_match_mask(catalog: Catalog, style: str) -> np.ndarray:
    want = {"hiking": "hiking", "touring": "cycling", "mtb": "mtb"}[style]
    return catalog.activity == want


def eligibility(catalog: Catalog, latent: dict[str, Any], radius: float) -> tuple[int, int]:
    hc = HardConstraints(
        category=latent["category"],
        home=(latent["home"]["lat"], latent["home"]["lon"]),
        max_travel_km=radius,
        max_distance_km=latent["max_distance_km"],
    )
    m = hc.mask(catalog)
    return int(m.sum()), int((m & activity_match_mask(catalog, latent["style"])).sum())


def ensure_eligible(catalog: Catalog, latent: dict[str, Any]) -> dict[str, Any]:
    """Widen the travel radius (people in sparse areas drive further) until the eligible set is meaningful."""
    radius = latent["max_travel_km"]
    n, n_act = eligibility(catalog, latent, radius)
    while (n < PARAMS["min_eligible"] or n_act < PARAMS["min_activity_match"]) and radius < PARAMS["radius_max_km"]:
        radius = min(PARAMS["radius_max_km"], round(radius * PARAMS["radius_widen_factor"]))
        latent["radius_widened"] = True
        n, n_act = eligibility(catalog, latent, radius)
    latent["max_travel_km"] = float(radius)
    return {"n": n, "n_activity_match": n_act}


# --------------------------------------------------------------------------- generator
def generate(
    catalog: Optional[Catalog] = None,
    towns: Optional[list[dict]] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    *,
    id_prefix: str = "P",
    split: Optional[str] = None,
) -> list[Persona]:
    """Deterministic for a given seed, town list and catalogue (no LLM: intents are added separately).

    ``split`` puts every persona in that split instead of drawing dev / test (the holdout, D28); ``id_prefix``
    names them (``H001``...). Neither changes any random draw, so the main 300 are unaffected."""
    catalog = catalog or get_catalog()
    towns = towns if towns is not None else load_towns()
    n = n or PARAMS["n_personas"]
    seed = PARAMS["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    w = np.array([t["population"] for t in towns], dtype=float) ** PARAMS["town_weight_exponent"]
    w /= w.sum()
    n_test = round(n * PARAMS["n_test"] / PARAMS["n_personas"])
    test_idx = set(int(i) for i in np.random.default_rng(seed + 1).choice(n, size=n_test, replace=False))
    style_rng = np.random.default_rng(seed + 2)

    out = []
    for i in range(n):
        latent = sample_latent(rng, towns, w)
        q = answer_questionnaire(rng, latent)
        elig = ensure_eligible(catalog, latent)
        style = _choice(style_rng, PARAMS["intent_styles"])
        mention_home = bool(style_rng.random() < PARAMS["intent_mentions_home"])
        voice = _choice(style_rng, PARAMS["intent_voices"])
        pid = f"{id_prefix}{i + 1:03d}"
        p = Persona(
            persona_id=pid,
            split=split or ("test" if i in test_idx else "dev"),
            latent=latent,
            questionnaire={k: v for k, v in q.items() if k != "fitness_misjudged"},
            user_vector=questionnaire_vector(q),
            profile={
                "home": [latent["home"]["lat"], latent["home"]["lon"]],
                "home_town": latent["home"]["town"],
                "max_travel_km": latent["max_travel_km"],
                "max_distance_km": latent["max_distance_km"],
            },
            intent_meta={
                "style": style,
                "voice": voice,
                "mention_home": mention_home,
                "fitness_misjudged": q["fitness_misjudged"],
            },
            eligible=elig,
        )
        out.append(p)
    return out


# --------------------------------------------------------------------------- statistics
def _quantiles(xs: list[float]) -> dict[str, float]:
    a = np.array(xs, dtype=float)
    return {
        k: _r(np.percentile(a, q), 1)
        for k, q in (("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("max", 100))
    }


def persona_stats(personas: list[Persona]) -> dict[str, Any]:
    c = Counter
    lat = [p.latent for p in personas]
    return {
        "n": len(personas),
        "split": dict(sorted(c(p.split for p in personas).items())),
        "activity_style": dict(sorted(c(f"{x['activity']}/{x['style']}" for x in lat).items())),
        "party": dict(sorted(c(x["party"]["kind"] for x in lat).items())),
        "with_dog": sum(1 for x in lat if x["party"]["dog"]),
        "region": dict(sorted(c(x["home"]["region"] for x in lat).items())),
        "distinct_towns": len({x["home"]["town"] for x in lat}),
        "top_towns": c(x["home"]["town"] for x in lat).most_common(8),
        "transport": dict(sorted(c(x["transport"] for x in lat).items())),
        "multi_day": sum(1 for x in lat if x["multi_day"]),
        "radius_widened": sum(1 for x in lat if x["radius_widened"]),
        "max_travel_km": _quantiles([x["max_travel_km"] for x in lat]),
        "max_distance_km": _quantiles([x["max_distance_km"] for x in lat]),
        "max_ascent_m": _quantiles([x["max_ascent_m"] for x in lat]),
        "questionnaire_fitness": dict(sorted(c(p.questionnaire["fitness"] for p in personas).items())),
        "fitness_misjudged": dict(
            sorted(c(str((p.intent_meta or {}).get("fitness_misjudged", 0)) for p in personas).items())
        ),
        "questionnaire_types": dict(sorted(c("+".join(p.questionnaire["type"]) for p in personas).items())),
        "desires": dict(sorted(c(k for x in lat for k in x["desires"]).items(), key=lambda kv: -kv[1])),
        "intent_styles": dict(sorted(c((p.intent_meta or {}).get("style") for p in personas).items())),
        "eligible_n": _quantiles([p.eligible["n"] for p in personas]),
        "eligible_activity_match": _quantiles([p.eligible["n_activity_match"] for p in personas]),
        "eligible_n_by_style": {
            s: _quantiles([p.eligible["n"] for p in personas if p.latent["style"] == s])
            for s in sorted({x["style"] for x in lat})
        },
        "eligible_activity_match_by_style": {
            s: _quantiles([p.eligible["n_activity_match"] for p in personas if p.latent["style"] == s])
            for s in sorted({x["style"] for x in lat})
        },
        "intent_words": _quantiles([len((p.intent or "").split()) for p in personas]),
    }


def write_personas(personas: list[Persona], path=None) -> str:
    path = path or paths.PERSONAS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(p.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for p in personas)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    from evalkit import intents
    from evalkit.llm import print_usage

    ap = argparse.ArgumentParser(description="generate eval/personas/personas.jsonl (deterministic from cache)")
    ap.add_argument("--n", type=int, default=None, help="pilot: only the first N personas (not written)")
    ap.add_argument("--no-intents", action="store_true", help="skip the LLM intents (dry run, not written)")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args(argv)

    personas = generate()
    if args.n:
        personas = personas[: args.n]
    client = None
    if not args.no_intents:
        client = intents.make_writer_client(max_concurrency=args.concurrency)
        intents.add_intents(personas, client)
    stats = persona_stats(personas)
    if args.n or args.no_intents:
        for p in personas[:10]:
            print(p.persona_id, p.latent["style"], p.latent["party"]["kind"], "|", p.intent)
        print(json.dumps(stats, indent=1, ensure_ascii=False)[:3000])
    else:
        sha = write_personas(personas)
        paths.PERSONA_PARAMS_FILE.write_text(
            json.dumps(
                {"params": PARAMS, "intent_writer": intents.writer_config(), "desire_labels": DESIRE_LABELS},
                indent=1,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        stats["personas_sha256"] = sha
        paths.PERSONA_STATS_FILE.write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(personas)} personas -> {paths.PERSONAS_FILE} (sha256 {sha[:12]})")
        print(json.dumps({k: stats[k] for k in ("split", "activity_style", "party", "eligible_n")}, ensure_ascii=False))
    if client is not None:
        print_usage(client)


if __name__ == "__main__":
    main()
