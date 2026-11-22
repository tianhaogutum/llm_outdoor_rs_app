"""Features for the learned ranker, built from the stages' outputs, the catalogue and the query only.

Nothing here reads a simulated user's latent profile: the inputs are :class:`UserQuery` (user vector,
tab, profile slots, intent) and catalogue facts. Missing values are NaN (LightGBM routes them to the
learned default branch). The list, with the source of each feature, is in docs/RECSYS.md.

Query parsing is deliberately simple and deterministic (regular expressions over the intent):
a distance figure ("around 8 km", "30-70 km"), loop wishes, "easy / gentle / flat" wishes, mountain
biking, and requested or negated concepts mapped to catalogue tags.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .text import split_negated

if TYPE_CHECKING:  # pragma: no cover
    from .catalog import Catalog
    from .query import UserQuery

FEATURES: tuple[str, ...] = (
    # rule sub-scores of v0_rules (questionnaire vector)
    "rule_difficulty",
    "rule_distance",
    "rule_tags",
    # retrieval stages
    "bm25",
    "bm25_rank",
    "dense",
    "dense_rank",
    "rrf",
    "fused_rank",
    "rank_api",
    # geography and length vs the profile slots
    "home_km",
    "home_ratio",
    "length_km",
    "length_cap_ratio",
    "length_band_pos",
    "intent_km_gap",
    # effort
    "ascent_m",
    "ascent_per_km",
    "ascent_fit",
    "intent_easy",
    "difficulty",
    "difficulty_in_range",
    "technical",
    "duration_h",
    # shape
    "is_loop",
    "intent_loop",
    "loop_match",
    # requested / negated concepts vs catalogue tags
    "tag_cover",
    "tag_hits",
    "neg_hits",
    # activity style
    "is_mtb",
    "intent_mtb",
    # surface, traffic, popularity proxies, data quality
    "paved_share",
    "busy_road_share",
    "popularity",
    "pois_per_km",
    "waymarked",
    "osm_description",
)

# Feature groups used in the overfitting analysis (docs/RECSYS.md, eval/results/_ltr.json)
GROUPS: dict[str, tuple[str, ...]] = {
    "slot_ratios": ("home_ratio", "length_cap_ratio"),
    "home_distance": ("home_km", "home_ratio"),
    "intent_numbers": ("intent_km_gap",),
    "rule_subscores": ("rule_difficulty", "rule_distance", "rule_tags"),
    "rank_api": ("rank_api",),
    "retrieval": ("bm25", "bm25_rank", "dense", "dense_rank", "rrf", "fused_rank"),
}

# concept -> (intent patterns, catalogue tags)
CONCEPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "lake": (r"\blakes?\b|\blakeside\b|\bpond", ("Lake",)),
    "swimming": (r"\bswim", ("Swimming spot",)),
    "castle": (r"\bcastles?\b|\bpalace", ("Castle",)),
    "ruins": (r"\bruins?\b", ("Ruins",)),
    "monastery": (r"\bmonaster|\babbey|\bconvent", ("Monastery",)),
    "museum": (r"\bmuseum", ("Museum",)),
    "culture": (r"\bculture|\bcultural|\bhistor|\bheritage", ("Cultural/historical interest",)),
    "hut": (r"\bhuts?\b|\balm\b|\bmountain inn", ("Hut",)),
    "beer_garden": (r"\bbeer ?gardens?\b|\bbiergarten", ("Beer garden",)),
    "refreshment": (
        r"\binn\b|\bpub\b|\bcaf[eé]|\bbite\b|\bfood\b|\beat\b|\bmeal|\bdrink|\brefreshment|\brestaurant",
        ("Refreshment stops available",),
    ),
    "summit": (r"\bsummits?\b|\bpeaks?\b|\btop of\b", ("Summit route",)),
    "viewpoint": (r"\bviewpoints?\b|\bviews?\b|\bpanoram", ("Viewpoint", "Scenic")),
    "waterfall": (r"\bwaterfalls?\b", ("Waterfall",)),
    "gorge": (r"\bgorges?\b|\bravine|\bcanyon", ("Gorge",)),
    "cave": (r"\bcaves?\b|\bcavern", ("Cave",)),
    "rocks": (r"\brocks?\b|\bgeolog|\bformations?\b", ("Geological highlights",)),
    "river": (r"\brivers?\b|\briverside|\bstream", ("River",)),
    "playground": (r"\bplayground", ("Playground",)),
    "family": (r"\bkids?\b|\bchild|\bfamily|\bfamilies", ("Family-friendly",)),
    "stroller": (r"\bstroller|\bpram|\bbuggy|\bpushchair", ("Suitable for strollers",)),
    "train": (r"\btrain\b|\brailway|\bstation", ("Train access",)),
    "wildlife": (r"\bwildlife|\banimals?\b|\bbirds?\b|\bzoo\b", ("Fauna",)),
    "forest": (r"\bforest|\bwoods?\b|\bwoodland|\bwooded|\bnature reserve", ("Nature reserve", "Flora")),
    "paved": (r"\bpaved\b|\bsmooth\b|\btarmac|\basphalt", ("Mostly paved",)),
    "flat": (r"\bflat\b|\blevel\b", ("Flat",)),
    "multi_day": (r"\bmulti[- ]?day|\bseveral days|\btwo days|\bthree days|\bovernight", ("Multi-stage route",)),
}
NEG_CONCEPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "busy_roads": (r"\bbusy\b|\btraffic|\broads?\b", ("Busy road sections",)),
    "steep": (r"\bsteep|\bclimb|\bstrenuous|\bhilly", ("Big climb",)),
    "alpine": (r"\balpine|\bexposed|\bexposure|\bvia ferrata|\btechnical", ("Alpine terrain", "Via ferrata")),
    "crowds": (r"\bcrowd|\bbusy trails?|\bpopular", ()),
}
_NUM = r"(\d+(?:[.,]\d+)?)"
_KM = re.compile(_NUM + r"(?:\s*(?:-|–|to|or)\s*" + _NUM + r")?\s*(?:km|kilomet)", re.IGNORECASE)
_LOOP = re.compile(r"\bloop|\bcircular|\bround[- ]?trip|\bback to (?:the )?(?:car|start)|\bround walk", re.I)
_EASY = re.compile(r"\beasy\b|\bgentle|\bflat\b|\bleisurely|\brelaxed|\bnot too steep|\blittle climbing", re.I)
_MTB = re.compile(r"\bmountain[- ]?bik|\bmtb\b|\bsingle ?trail|\btrail rid", re.I)


@dataclass(frozen=True)
class IntentInfo:
    km_target: Optional[float]
    wants_loop: bool
    wants_easy: bool
    wants_mtb: bool
    concepts: tuple[str, ...]
    negated: tuple[str, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "km_target": self.km_target,
            "wants_loop": self.wants_loop,
            "wants_easy": self.wants_easy,
            "wants_mtb": self.wants_mtb,
            "concepts": list(self.concepts),
            "negated": list(self.negated),
        }


def parse_intent(intent: Optional[str]) -> IntentInfo:
    text = intent or ""
    m = _KM.search(text)
    km = None
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", ".")) if m.group(2) else None
        km = (a + b) / 2 if b else a
    pos, removed = split_negated(text)  # negated concepts are searched in the negated spans only
    concepts = tuple(c for c, (pat, _) in CONCEPTS.items() if re.search(pat, pos, re.I))
    negated = tuple(c for c, (pat, _) in NEG_CONCEPTS.items() if re.search(pat, removed, re.I))
    return IntentInfo(
        km_target=km,
        wants_loop=bool(_LOOP.search(pos)),
        wants_easy=bool(_EASY.search(pos)),
        wants_mtb=bool(_MTB.search(text)),
        concepts=concepts,
        negated=negated,
    )


def _ascent_budget(catalog: Catalog, activity: str, drange: tuple[float, float]) -> float:
    """75th percentile of max(ascent, descent) over the catalogue routes of this activity whose difficulty
    lies in the user's difficulty range (catalogue-derived; no latent data)."""
    key = f"feat:ascent_budget:{activity}:{drange[0]}:{drange[1]}"

    def build():
        act = catalog.activity == activity
        d = catalog.difficulty
        m = act & (d >= math.floor(drange[0])) & (d <= math.ceil(drange[1]))
        if m.sum() < 20:
            m = act
        vals = np.maximum(catalog.ascent_m[m], catalog.descent_m[m])
        return float(max(np.percentile(vals, 75), 50.0)) if len(vals) else 300.0

    return catalog.memo(key, build)


def stage_arrays(n: int) -> dict[str, np.ndarray]:
    """Empty per-candidate stage arrays (NaN = stage not run / not available)."""
    return {
        k: np.full(n, np.nan) for k in ("bm25", "bm25_rank", "dense", "dense_rank", "rrf", "fused_rank", "rank_api")
    }


def build_features(
    catalog: Catalog,
    query: UserQuery,
    cand: np.ndarray,
    stages: dict[str, np.ndarray],
    home_km: Optional[np.ndarray],
    intent: Optional[IntentInfo] = None,
) -> np.ndarray:
    """Feature matrix (len(cand), len(FEATURES)) for catalogue indices ``cand``."""
    from app.services.recommendation_service import (
        calculate_difficulty_score,
        calculate_distance_score,
        calculate_tag_score,
        default_difficulty_range,
    )

    intent = intent or parse_intent(query.intent)
    uv = query.user_vector or {}
    drange = uv.get("difficulty_range") or default_difficulty_range()
    dmin = float(uv.get("min_distance_km", 0.0))
    dmax = float(uv.get("max_distance_km", 100.0))
    utags = uv.get("preferred_tags", [])
    n = len(cand)
    X = np.full((n, len(FEATURES)), np.nan, dtype=np.float64)
    col = {f: j for j, f in enumerate(FEATURES)}
    concept_tags = [set(CONCEPTS[c][1]) for c in intent.concepts]
    neg_tags = [set(NEG_CONCEPTS[c][1]) for c in intent.negated]
    for row, i in enumerate(cand):
        r = catalog.routes[int(i)]
        tags = set(r.tags)
        feat = r.features or {}
        prof = feat.get("profile") or {}
        pop = feat.get("popularity_proxies") or {}
        L = r.length_km
        asc = max(r.ascent_m, r.descent_m)
        v = X[row]
        v[col["rule_difficulty"]] = calculate_difficulty_score(drange, r.difficulty)
        v[col["rule_distance"]] = calculate_distance_score(dmin, dmax, L) if dmax > 0 else np.nan
        v[col["rule_tags"]] = calculate_tag_score(utags, [t.lower() for t in r.tags])
        if home_km is not None:
            v[col["home_km"]] = home_km[int(i)]
            if query.max_travel_km:
                v[col["home_ratio"]] = home_km[int(i)] / query.max_travel_km
        v[col["length_km"]] = L
        if query.max_distance_km:
            v[col["length_cap_ratio"]] = L / query.max_distance_km
        if dmax > dmin:
            v[col["length_band_pos"]] = (L - dmin) / (dmax - dmin)
        if intent.km_target:
            v[col["intent_km_gap"]] = abs(L - intent.km_target) / intent.km_target
        v[col["ascent_m"]] = asc
        v[col["ascent_per_km"]] = asc / max(L, 0.1)
        v[col["ascent_fit"]] = asc / _ascent_budget(catalog, r.activity, (float(drange[0]), float(drange[1])))
        v[col["intent_easy"]] = float(intent.wants_easy)
        v[col["difficulty"]] = r.difficulty
        v[col["difficulty_in_range"]] = float(drange[0] <= r.difficulty <= drange[1])
        if r.difficulty_technical is not None:
            v[col["technical"]] = r.difficulty_technical
        v[col["duration_h"]] = r.duration_min / 60.0
        v[col["is_loop"]] = float(r.is_loop)
        v[col["intent_loop"]] = float(intent.wants_loop)
        v[col["loop_match"]] = float(intent.wants_loop and r.is_loop)
        hits = sum(1 for ts in concept_tags if ts & tags)
        v[col["tag_hits"]] = hits
        if concept_tags:
            v[col["tag_cover"]] = hits / len(concept_tags)
        v[col["neg_hits"]] = sum(1 for ts in neg_tags if ts & tags)
        v[col["is_mtb"]] = float(r.activity == "mtb")
        v[col["intent_mtb"]] = float(intent.wants_mtb)
        v[col["paved_share"]] = prof.get("paved_share", np.nan)
        v[col["busy_road_share"]] = prof.get("busy_road_share", np.nan)
        v[col["popularity"]] = pop.get("other_routes_on_same_ways", np.nan)
        v[col["pois_per_km"]] = pop.get("pois_per_km", np.nan)
        v[col["waymarked"]] = float("Waymarked" in tags)
        v[col["osm_description"]] = float(r.description_source != "generated")
    for k in ("bm25", "bm25_rank", "dense", "dense_rank", "rrf", "fused_rank", "rank_api"):
        X[:, col[k]] = stages[k]
    return X
