"""Feedback in the multi-stage pipeline (Phase 5): signals, decay, user-representation update, route effects.

Design and parameters: docs/RECSYS.md section "Feedback and exploration"; decisions D20-D22.

Signals (:class:`~app.recsys.query.FeedbackEvent`): the four negative reasons of the product
(``too-hard``, ``too-easy``, ``too-far``, ``not-interested``) and ``completed`` (a souvenir record: the
user did the route). Every event carries its age in days and gets one weight,
``calculate_time_decay_weight(age, half_life)`` = ``0.5 ** (age / half_life)`` (the fixed true
half-life of ``recommendation_service``), the same function and half-life for both signs.

What feedback changes, always computed from the **original** query (the profile as stored, never an
already adjusted one), so it is applied exactly once:

1. **Slots** (:func:`adjust_query`): difficulty range (too-hard / too-easy shift it, width >= 1, clamped
   to 0-3), the absolute distance cap (too-hard on a long route pulls it towards that route's length),
   the travel radius (too-far pulls it towards that route's distance from home), the vector's minimum
   distance (too-easy on a short route), preferred / avoided tags (completed routes add their content
   tags; not-interested removes them unless completions outweigh). Floors: cap and radius never drop
   below half the user's own value. The returned query has ``feedback=()``: running it again changes
   nothing.
2. **Intent embedding** (Rocchio, :func:`rocchio`): ``q' = alpha q + beta c_pos - gamma c_neg`` with
   decay-weighted centroids of completed routes (``c_pos``) and not-interested routes (``c_neg``),
   renormalised. too-hard / too-easy / too-far say nothing about *content*, so they do not move the
   embedding. ``q'`` replaces the intent embedding in dense retrieval. Without an intent (questionnaire
   only) there is no ``q``: the direction ``beta c_pos - gamma c_neg`` is used only for re-scoring.
3. **Re-scoring** (:func:`rescore`, the ``feedback`` stage after the ranker): relevance = ranker score
   min-max normalised over the candidates, plus ``eta_emb`` x (cosine to the Rocchio direction, centred
   and scaled over the candidates), plus ``eta_tag`` x tag affinity (net decayed preference of the route's
   content tags), minus ``nu_dup`` x weight for near-duplicates of rejected or completed routes.
4. **Route effects**: a completed route is not shown again while its weight is >= ``exclude_weight``
   (two half-lives: the user just did it); a rejected route likewise (the user said no). After that both
   may come back (an old thumbs-down no longer hides a route forever, D7 point 5). Near-duplicates
   (same activity, start within ``dup_start_km``, length within ``dup_length_ratio``) are demoted, not
   removed: they share the rejected route's problem, but may be the right variant.

Pure functions; the pipeline (``pipeline.py``) calls them and writes the trace.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np

from .catalog import Catalog
from .constraints import haversine_km
from .query import FeedbackEvent, UserQuery

NEGATIVE_REASONS = ("too-hard", "too-easy", "too-far", "not-interested")
POSITIVE_REASON = "completed"
REASONS = (*NEGATIVE_REASONS, POSITIVE_REASON)

# Content tags: what a route is *about*. Access, shape, surface and effort tags are not taste signals
# (DATA.md section 7); "Refreshment stops available" and "Church/chapel" are proximity tags on most routes.
CONTENT_TAGS: tuple[str, ...] = (
    "Scenic",
    "Cultural/historical interest",
    "Family-friendly",
    "Castle",
    "Museum",
    "Summit route",
    "Playground",
    "Flora",
    "Lake",
    "Geological highlights",
    "Beer garden",
    "Viewpoint",
    "River",
    "Cave",
    "Ruins",
    "Nature reserve",
    "Picnic area",
    "Fauna",
    "Monastery",
    "Hut",
    "Swimming spot",
    "Alpine terrain",
    "Waterfall",
    "National park",
    "Gorge",
    "Pilgrimage route",
    "Via ferrata",
)

DIFFICULTY_MIN, DIFFICULTY_MAX, MIN_WIDTH = 0.0, 3.0, 1.0


@dataclass(frozen=True)
class FeedbackConfig:
    """Every parameter of the feedback loop. Defaults = the values chosen on dev personas (D21): the
    simulation's dev grid (``evalkit.simulate.TUNING``) found no gain from Rocchio or re-scoring over the
    route effects, and losses at larger weights, so both are off by default; the slot steps are the
    grid's light setting. ``LIGHT_LOOP`` in ``systems/pipelines.py`` keeps the full light loop runnable."""

    half_life_days: float = 60.0
    # Rocchio
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    # re-scoring weights (relevance is in [0, 1])
    eta_emb: float = 0.0
    eta_tag: float = 0.0
    nu_dup: float = 0.0
    # slot steps (fraction of the way per unit of decayed weight)
    difficulty_step: float = 0.25
    cap_step: float = 0.25
    radius_step: float = 0.25
    floor_share: float = 0.5  # cap / radius never below this share of the user's own value
    # route effects
    exclude_weight: float = 0.25  # = two half-lives
    dup_start_km: float = 1.0
    dup_length_ratio: float = 0.25

    def describe(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- signals and decay
def decay_weight(age_days: float, half_life_days: float) -> float:
    """The product's fixed true half-life, shared by positive and negative events."""
    from app.services.recommendation_service import calculate_time_decay_weight

    return float(calculate_time_decay_weight(float(age_days), half_life_days))


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _as_utc(t: datetime) -> datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def feedback_events_from_rows(
    feedback_rows: Iterable[Any] = (), souvenir_rows: Iterable[Any] = (), *, now: Optional[datetime] = None
) -> tuple[FeedbackEvent, ...]:
    """Build the events a :class:`UserQuery` carries from stored rows (for the API).

    ``feedback_rows``: objects or mappings with ``route_id``, ``reason`` and ``created_at`` (the
    ``profile_feedback`` table); unknown reasons are skipped. ``souvenir_rows``: ``route_id`` and
    ``completed_at`` (the ``souvenirs`` table) -> ``completed``. Missing timestamps count as age 0.
    ``now`` defaults to the current UTC time. Oldest first."""
    now = _as_utc(now or datetime.now(UTC))
    out: list[tuple[float, FeedbackEvent]] = []

    def age(t: Optional[datetime]) -> float:
        return max(0.0, (now - _as_utc(t)).total_seconds() / 86400.0) if t else 0.0

    for r in feedback_rows:
        reason = _get(r, "reason")
        if reason in NEGATIVE_REASONS:
            a = age(_get(r, "created_at"))
            out.append((a, FeedbackEvent(int(_get(r, "route_id")), reason, round(a, 4))))
    for r in souvenir_rows:
        a = age(_get(r, "completed_at"))
        out.append((a, FeedbackEvent(int(_get(r, "route_id")), POSITIVE_REASON, round(a, 4))))
    out.sort(key=lambda t: (-t[0], t[1].route_id, t[1].reason))
    return tuple(e for _, e in out)


@dataclass(frozen=True)
class WeightedEvent:
    route_id: int
    index: int  # catalogue index
    reason: str
    age_days: float
    weight: float


def weigh(catalog: Catalog, events: Iterable[FeedbackEvent], cfg: FeedbackConfig) -> list[WeightedEvent]:
    """Events on known routes with their decay weight, oldest first (unknown routes and reasons dropped)."""
    out = []
    for e in events:
        if e.reason not in REASONS or int(e.route_id) not in catalog:
            continue
        out.append(
            WeightedEvent(
                int(e.route_id),
                catalog.index_of(int(e.route_id)),
                e.reason,
                float(e.age_days),
                decay_weight(e.age_days, cfg.half_life_days),
            )
        )
    out.sort(key=lambda w: (-w.age_days, w.route_id))
    return out


# --------------------------------------------------------------------------- slots
@dataclass
class FeedbackState:
    """Everything the later stages need, computed once per request from the original query."""

    cfg: FeedbackConfig
    events: list[WeightedEvent]
    query: UserQuery  # adjusted, with feedback=()
    slot_changes: dict[str, Any] = field(default_factory=dict)
    tag_pref: dict[str, float] = field(default_factory=dict)  # content tag -> net preference in (-1, 1)
    excluded: dict[int, str] = field(default_factory=dict)  # route id -> reason (completed / rejected)

    @property
    def positives(self) -> list[WeightedEvent]:
        return [e for e in self.events if e.reason == POSITIVE_REASON]

    @property
    def content_negatives(self) -> list[WeightedEvent]:
        return [e for e in self.events if e.reason == "not-interested"]

    def info(self) -> dict[str, Any]:
        n = {r: sum(1 for e in self.events if e.reason == r) for r in REASONS}
        return {
            "events": {k: v for k, v in n.items() if v},
            "weight_sum": round(sum(e.weight for e in self.events), 4),
            "half_life_days": self.cfg.half_life_days,
            "slot_changes": self.slot_changes,
            "excluded": {str(k): v for k, v in sorted(self.excluded.items())},
        }


def _shift_range(rng: list[float], shift: float) -> list[float]:
    lo, hi = float(rng[0]) + shift, float(rng[1]) + shift
    width = max(float(rng[1]) - float(rng[0]), MIN_WIDTH)
    if lo < DIFFICULTY_MIN:
        lo, hi = DIFFICULTY_MIN, max(DIFFICULTY_MIN + MIN_WIDTH, min(hi, DIFFICULTY_MIN + width))
    if hi > DIFFICULTY_MAX:
        lo, hi = min(DIFFICULTY_MAX - MIN_WIDTH, max(lo, DIFFICULTY_MAX - width)), DIFFICULTY_MAX
    return [round(lo, 4), round(hi, 4)]


def _pull(value: float, target: float, frac: float, floor: float) -> float:
    """Move ``value`` down towards ``target`` by ``frac`` of the gap (never up), not below ``floor``."""
    if target >= value:
        return value
    return max(floor, value - frac * (value - target))


def adjust_query(catalog: Catalog, query: UserQuery, cfg: FeedbackConfig) -> FeedbackState:
    """Apply the query's feedback once, from the original slots. The result's query carries no feedback."""
    events = weigh(catalog, query.feedback, cfg)
    v = dict(query.user_vector or {})
    changes: dict[str, Any] = {}
    cap, radius = query.max_distance_km, query.max_travel_km
    home = query.home

    # difficulty: one net shift from all too-hard / too-easy events
    shift = cfg.difficulty_step * sum(
        (e.weight if e.reason == "too-easy" else -e.weight) for e in events if e.reason in ("too-hard", "too-easy")
    )
    if shift and v.get("difficulty_range"):
        new = _shift_range(list(v["difficulty_range"]), shift)
        changes["difficulty_range"] = {"from": list(v["difficulty_range"]), "to": new}
        v["difficulty_range"] = new

    # cap, radius, minimum distance: oldest first, each pulls by its weight
    new_cap, new_radius = cap, radius
    min_d = float(v["min_distance_km"]) if v.get("min_distance_km") is not None else None
    for e in events:
        r = catalog.routes[e.index]
        if e.reason == "too-hard" and new_cap is not None and r.length_km >= 0.6 * new_cap:
            new_cap = _pull(new_cap, 0.9 * r.length_km, cfg.cap_step * e.weight, cfg.floor_share * cap)
        elif e.reason == "too-far" and new_radius is not None and home is not None:
            d = haversine_km(home[0], home[1], r.start_lat, r.start_lon)
            new_radius = _pull(new_radius, 0.9 * float(d), cfg.radius_step * e.weight, cfg.floor_share * radius)
        elif e.reason == "too-easy" and min_d is not None and r.length_km <= max(min_d, 1.0) * 1.5:
            target = 1.1 * r.length_km
            hi = float(v.get("max_distance_km", target))
            if target > min_d:
                min_d = min(hi * 0.8, min_d + cfg.difficulty_step * e.weight * (target - min_d))
    if new_cap is not None and cap is not None and new_cap != cap:
        new_cap = round(new_cap, 3)
        changes["max_distance_km"] = {"from": cap, "to": new_cap}
    if new_radius is not None and radius is not None and new_radius != radius:
        new_radius = round(new_radius, 3)
        changes["max_travel_km"] = {"from": radius, "to": new_radius}
    if min_d is not None and v.get("min_distance_km") is not None and min_d != float(v["min_distance_km"]):
        changes["min_distance_km"] = {"from": v["min_distance_km"], "to": round(min_d, 3)}
        v["min_distance_km"] = round(min_d, 3)

    # tags: decayed net preference per content tag; vector preferred / avoided tags follow its sign
    pos: dict[str, float] = {}
    neg: dict[str, float] = {}
    for e in events:
        tgt = pos if e.reason == POSITIVE_REASON else neg if e.reason == "not-interested" else None
        if tgt is None:
            continue
        for t in catalog.routes[e.index].tags:
            if t in CONTENT_TAGS:
                tgt[t] = tgt.get(t, 0.0) + e.weight
    pref = {t: (pos.get(t, 0.0) - neg.get(t, 0.0)) / (pos.get(t, 0.0) + neg.get(t, 0.0) + 1.0) for t in {*pos, *neg}}
    if pref:
        tags = [t for t in (v.get("preferred_tags") or []) if isinstance(t, str)]
        low = {t.casefold() for t in tags}
        drop = {t.casefold() for t, p in pref.items() if p < 0}
        add = sorted(t for t, p in pref.items() if p > 0 and t.casefold() not in low)
        new_tags = [t for t in tags if t.casefold() not in drop] + [t.lower() for t in add]
        if new_tags != tags:
            changes["preferred_tags"] = {"added": [t.lower() for t in add], "removed": sorted(drop & low)}
            v["preferred_tags"] = new_tags
        avoided = sorted({*(v.get("avoided_tags") or []), *(t for t, p in pref.items() if p < 0)})
        if avoided:
            v["avoided_tags"] = avoided

    # route effects: recent completions and rejections are not shown again
    excluded: dict[int, str] = {}
    for e in events:
        if e.weight >= cfg.exclude_weight:
            excluded[e.route_id] = "completed" if e.reason == POSITIVE_REASON else f"rejected: {e.reason}"

    q2 = replace(
        query,
        user_vector=v,
        max_distance_km=new_cap if cap is not None else None,
        max_travel_km=new_radius if radius is not None else None,
        feedback=(),
    )
    return FeedbackState(cfg, events, q2, changes, {t: round(p, 6) for t, p in sorted(pref.items())}, excluded)


# --------------------------------------------------------------------------- Rocchio
def _centroid(matrix: np.ndarray, events: list[WeightedEvent]) -> Optional[np.ndarray]:
    w = np.array([e.weight for e in events], dtype=np.float64)
    if not len(events) or w.sum() <= 0:
        return None
    c = (w[:, None] * matrix[[e.index for e in events]].astype(np.float64)).sum(0) / w.sum()
    return c


def _unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


def rocchio(q: Optional[np.ndarray], matrix: np.ndarray, st: FeedbackState) -> tuple[Optional[np.ndarray], dict]:
    """``q' = alpha q + beta c_pos - gamma c_neg`` (unit length). ``q`` None (no intent) -> the feedback
    direction ``beta c_pos - gamma c_neg`` alone (None if there is no content feedback)."""
    cfg = st.cfg
    cp = _centroid(matrix, st.positives)
    cn = _centroid(matrix, st.content_negatives)
    d = np.zeros(matrix.shape[1], dtype=np.float64)
    if cp is not None:
        d += cfg.beta * cp
    if cn is not None:
        d -= cfg.gamma * cn
    info = {"alpha": cfg.alpha, "beta": cfg.beta, "gamma": cfg.gamma, "n_pos": len(st.positives),
            "n_neg": len(st.content_negatives)}
    if cp is None and cn is None:
        return (None if q is None else q), info
    if q is None:
        return _unit(d), info
    q0 = _unit(np.asarray(q, dtype=np.float64))
    q1 = _unit(cfg.alpha * q0 + d)
    info["cosine_to_original"] = round(float(q0 @ q1), 6)
    return q1.astype(np.float32), info


def feedback_direction(matrix: np.ndarray, st: FeedbackState) -> Optional[np.ndarray]:
    """``beta c_pos - gamma c_neg`` (unit), the part of Rocchio that comes from feedback only."""
    return rocchio(None, matrix, st)[0]


# --------------------------------------------------------------------------- re-scoring
def tag_affinity(tags: Iterable[str], pref: Mapping[str, float]) -> float:
    """Mean net preference over the route's content tags (0 without content tags or preferences)."""
    ts = [t for t in tags if t in CONTENT_TAGS]
    if not ts or not pref:
        return 0.0
    return float(sum(pref.get(t, 0.0) for t in ts) / len(ts))


def near_duplicate(catalog: Catalog, i: int, j: int, cfg: FeedbackConfig) -> bool:
    a, b = catalog.routes[i], catalog.routes[j]
    if i == j or a.activity != b.activity:
        return False
    if haversine_km(a.start_lat, a.start_lon, b.start_lat, b.start_lon) > cfg.dup_start_km:
        return False
    la, lb = max(a.length_km, 0.1), max(b.length_km, 0.1)
    return abs(la - lb) / max(la, lb) <= cfg.dup_length_ratio


def rescore(
    catalog: Catalog,
    cand_idx: list[int],
    scores: np.ndarray,
    st: FeedbackState,
    matrix: Optional[np.ndarray],
) -> tuple[list[int], np.ndarray, dict[int, dict[str, Any]], dict[str, Any]]:
    """Feedback stage: drop excluded routes, re-score the rest. ``cand_idx`` are catalogue indices in
    ranker order with their ranker ``scores``. Returns (kept indices in new order, new scores, per-route
    contributions keyed by catalogue index, stage info)."""
    cfg = st.cfg
    ids = [int(catalog.ids[i]) for i in cand_idx]
    keep = [p for p, rid in enumerate(ids) if rid not in st.excluded]
    removed = [rid for rid in ids if rid in st.excluded]
    if not keep:
        return [], np.array([]), {}, {"removed": removed}
    s = np.asarray(scores, dtype=np.float64)[keep]
    lo, hi = float(s.min()), float(s.max())
    rel = (s - lo) / (hi - lo) if hi > lo else np.ones(len(s))
    idx = [cand_idx[p] for p in keep]

    emb = np.zeros(len(idx))
    direction = feedback_direction(matrix, st) if matrix is not None else None
    if direction is not None and len(idx) > 1:
        cos = matrix[idx].astype(np.float64) @ direction
        span = float(cos.max() - cos.min())
        emb = (cos - cos.mean()) / span if span > 0 else np.zeros(len(idx))
    tag = np.array([tag_affinity(catalog.routes[i].tags, st.tag_pref) for i in idx])
    sources = [e for e in st.events if e.reason != "too-easy"]
    dup = np.zeros(len(idx))
    dup_of: dict[int, int] = {}
    for p, i in enumerate(idx):
        for e in sources:
            if near_duplicate(catalog, i, e.index, cfg) and e.weight > dup[p]:
                dup[p], dup_of[p] = e.weight, e.route_id
    final = rel + cfg.eta_emb * emb + cfg.eta_tag * tag - cfg.nu_dup * dup
    order = np.argsort(-final, kind="stable")
    contrib: dict[int, dict[str, Any]] = {}
    for after, o in enumerate(order, start=1):
        contrib[idx[o]] = {
            "relevance": round(float(rel[o]), 6),
            "embedding_affinity": round(float(emb[o]), 6),
            "tag_affinity": round(float(tag[o]), 6),
            "near_duplicate_of": dup_of.get(int(o)),
            "duplicate_penalty": round(float(cfg.nu_dup * dup[o]), 6),
            "score": round(float(final[o]), 6),
            "rank_before": int(o) + 1,
            "rank_after": after,
        }
    info = {"removed": removed, "eta_emb": cfg.eta_emb, "eta_tag": cfg.eta_tag, "nu_dup": cfg.nu_dup,
            "near_duplicates": len(dup_of)}
    return [idx[o] for o in order], final[order], contrib, info
