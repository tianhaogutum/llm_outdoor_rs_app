"""Pool-only contributors and TREC-style pooling.

Pool-only contributors are **not competitors**: they never appear in result tables. Like the manual
runs in TREC, their only job is to put likely-relevant routes into the judged pool, so that the ideal
DCG and the Recall@50 denominator do not depend only on what the (weak) baselines happened to
retrieve. Both see the persona's latent profile (``pool_latent``) or its free text (``pool_intent_lexical``)
and both return only routes that pass the persona's hard constraints.

- ``pool_latent``: a hand-written utility over the latent profile and catalogue facts (distance band,
  ascent and technical fit, party suitability, dislikes, weighted desires). Multiplicative feasibility
  times desire coverage: deliberately not the functional form of ``v0_rules``.
- ``pool_intent_lexical``: BM25 of the free-text intent over title + description + tags.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

from app.recsys import Catalog, RouteRecord
from evalkit.personas import Persona

POOL_DEPTH = 10
POOL_ONLY = ("pool_latent", "pool_intent_lexical")

DESIRE_TAGS: dict[str, tuple[str, ...]] = {
    "lake": ("Lake",),
    "swimming": ("Swimming spot",),
    "refreshment": ("Refreshment stops available",),
    "hut": ("Hut",),
    "beer_garden": ("Beer garden",),
    "summit": ("Summit route",),
    "viewpoint": ("Viewpoint",),
    "waterfall_gorge": ("Waterfall", "Gorge"),
    "cave_geology": ("Cave", "Geological highlights"),
    "culture": ("Castle", "Ruins", "Monastery", "Museum"),
    "river": ("River",),
    "playground": ("Playground",),
    "wildlife": ("Fauna",),
    "train_access": ("Train access",),
    "multi_day": ("Multi-stage route",),
}


def has_desire(r: RouteRecord, key: str) -> bool:
    tags = set(r.tags)
    if key == "quiet_nature":
        prof = (r.features or {}).get("profile") or {}
        natural = prof.get("unpaved_share", 0) >= 0.6 and prof.get("road_share", 1) <= 0.15
        return bool(tags & {"Nature reserve", "National park", "Flora"}) or natural
    if key == "multi_day":
        return "Multi-stage route" in tags or r.est_days >= 2
    return bool(tags & set(DESIRE_TAGS.get(key, ())))


def latent_utility(p: Persona, r: RouteRecord) -> float:
    x = p.latent
    tags = set(r.tags)
    prof = (r.features or {}).get("profile") or {}
    f = 1.0
    # activity / style
    if x["style"] == "mtb" and r.activity != "mtb":
        f *= 0.45
    if x["style"] == "touring" and r.activity == "mtb":
        f *= 0.25
    # distance band
    lo, hi = x["distance_band_km"]
    L = r.length_km
    if L < lo:
        f *= math.exp(-2.0 * ((lo - L) / max(lo, 0.5)) ** 2)
    elif L > hi:
        f *= math.exp(-3.0 * (L - hi) / max(hi, 0.5))
    # ascent
    asc = max(r.ascent_m, r.descent_m)
    if asc > x["max_ascent_m"]:
        f *= math.exp(-2.0 * (asc - x["max_ascent_m"]) / max(x["max_ascent_m"], 50))
    # technical
    tech = r.difficulty_technical or 0
    if tech > x["technical_max"]:
        f *= 0.15 ** (tech - x["technical_max"])
    # party and dislikes
    party = x["party"]
    if party.get("stroller") and not (
        "Suitable for strollers" in tags or (prof.get("paved_share", 0) >= 0.7 and not prof.get("steps_m"))
    ):
        f *= 0.3
    if party["kind"] in ("family_young", "family_school") and "Family-friendly" in tags:
        f *= 1.15
    if "alpine_terrain" in x["dislikes"] and tags & {"Alpine terrain", "Via ferrata"}:
        f *= 0.1
    if "busy_roads" in x["dislikes"] and "Busy road sections" in tags:
        f *= 0.4
    if "steep_sections" in x["dislikes"] and r.max_grade_200m_pct > 15:
        f *= 0.7
    if (
        "crowds" in x["dislikes"]
        and ((r.features or {}).get("popularity_proxies") or {}).get("other_routes_on_same_ways", 0) > 3
    ):
        f *= 0.8
    if x["shape"] == "loop" and not r.is_loop:
        f *= 0.7
    if x["shape"] == "linear" and r.is_loop:
        f *= 0.85
    if x["surface"] == "paved" and "Mostly unpaved" in tags:
        f *= 0.6
    if x["surface"] == "natural" and "Mostly paved" in tags:
        f *= 0.6
    if x["multi_day"] and not has_desire(r, "multi_day"):
        f *= 0.6
    d = x["desires"]
    wsum = sum(d.values())
    cover = sum(w for k, w in d.items() if has_desire(r, k)) / wsum if wsum else 0.0
    return f * (0.25 + 0.75 * cover)


def feasible_indices(catalog: Catalog, p: Persona) -> np.ndarray:
    return np.flatnonzero(p.constraints().mask(catalog))


def pool_latent(catalog: Catalog, p: Persona, k: int = POOL_DEPTH) -> list[int]:
    idx = feasible_indices(catalog, p)
    scored = sorted(
        ((latent_utility(p, catalog.routes[i]), int(catalog.ids[i])) for i in idx), key=lambda t: (-t[0], t[1])
    )
    return [rid for _, rid in scored[:k]]


# --------------------------------------------------------------------------- BM25 over the intent
_TOKEN = re.compile(r"[a-zäöüß0-9]+")
_STOP = set(
    """a an and are as at be but by for from has have i i'd i'm id im in is it its me my of on or our
please so some something that the this to us we we'd we're with would you your could around about like
want looking suggest ideally nice good near maybe also there their them they km""".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str], idx: Iterable[int]) -> list[float]:
        out = []
        for i in idx:
            tf, s = self.tf[i], 0.0
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in query:
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            out.append(s)
        return out


def route_text(r: RouteRecord) -> str:
    return f"{r.title} {r.description} {' '.join(r.tags)}"


def _bm25(catalog: Catalog) -> BM25:
    return catalog.memo("evalkit:bm25", lambda: BM25([tokens(route_text(r)) for r in catalog.routes]))


def pool_intent_lexical(catalog: Catalog, p: Persona, k: int = POOL_DEPTH) -> list[int]:
    idx = feasible_indices(catalog, p)
    q = tokens(p.intent or "")
    s = _bm25(catalog).scores(q, idx)
    ranked = sorted(zip(s, (int(catalog.ids[i]) for i in idx), strict=True), key=lambda t: (-t[0], t[1]))
    return [rid for _, rid in ranked[:k]]


CONTRIBUTORS = {"pool_latent": pool_latent, "pool_intent_lexical": pool_intent_lexical}


def contributor_runs(catalog: Catalog, personas: list[Persona], k: int = POOL_DEPTH) -> dict[str, dict[str, list[int]]]:
    return {name: {p.persona_id: fn(catalog, p, k) for p in personas} for name, fn in CONTRIBUTORS.items()}


def pool_pairs(runs: dict[str, dict[str, list[int]]], depth: int = POOL_DEPTH) -> set[tuple[str, int]]:
    """Union of the top-``depth`` of every run: {(persona_id, route_id)}."""
    out = set()
    for run in runs.values():
        for pid, ids in run.items():
            out.update((pid, int(r)) for r in ids[:depth])
    return out
