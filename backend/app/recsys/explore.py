"""Exploration for cold start (Phase 5): one reserved slot, Thompson sampling over tag clusters.

**Clusters** come from catalogue content only (:func:`tag_clusters`): every route's content tags
(``feedback.CONTENT_TAGS``) as an IDF-weighted, unit-length vector; seeded spherical k-means
(``k`` clusters, best of ``restarts``); routes without any content tag form one extra cluster
("no theme tags"). Each cluster is labelled by its centroid's three heaviest tags. Deterministic for a
given catalogue (memoised on it).

**Posteriors** are per user and rebuilt from the feedback history on every request (no stored state):
cluster c has Beta(a0 + S_c, b0 + F_c), with S_c = decayed weight of completed routes in c and F_c =
decayed weight of not-interested routes in c. too-hard / too-easy / too-far rejections are about effort
and distance, not theme, so they do not count against a cluster. Prior Beta(1, 1): nothing is known
about a new user's taste across themes, and a flat prior explores every theme with equal chance at first.

**The slot**: while the user has fewer than ``max_events`` feedback events (cold start), position
``position`` (default 10, last) of the top 10 is reserved. Thompson draws one theta per cluster from a
generator seeded by (seed, request id, number of events), so a request is reproducible; among clusters
that have an eligible route and are *not* already represented in the first ``position - 1`` routes
(all clusters if every one is), the largest draw wins. The route is that cluster's best candidate below
the reserved slot in the pipeline's own order; if the cluster has no candidate there, its eligible route
most similar to the query embedding (seeded random without one). It comes from the filtered eligible
set, so it passes every hard filter, and never a route the feedback stage excluded. The trace marks it
``"shown to learn your taste"``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from .catalog import Catalog
from .feedback import CONTENT_TAGS, POSITIVE_REASON, WeightedEvent

EXPLORE_REASON = "shown to learn your taste"


@dataclass(frozen=True)
class ExploreConfig:
    k: int = 8
    restarts: int = 5
    seed: int = 20261123
    prior_a: float = 1.0
    prior_b: float = 1.0
    position: int = 10
    max_events: int = 10

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Clusters:
    assign: np.ndarray  # catalogue index -> cluster id (k = "no theme tags")
    labels: tuple[str, ...]
    sizes: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.labels)


def _kmeans(X: np.ndarray, k: int, restarts: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
        a = np.zeros(len(X), dtype=np.int64)
        for _it in range(100):
            a_new = np.argmax(X @ C.T, axis=1)
            if _it and np.array_equal(a_new, a):
                break
            a = a_new
            for c in range(len(C)):
                m = X[a == c]
                if len(m):
                    v = m.mean(0)
                    n = np.linalg.norm(v)
                    C[c] = v / n if n > 0 else C[c]
        obj = float(np.max(X @ C.T, axis=1).sum())
        if best is None or obj > best[0] + 1e-12:
            best = (obj, C.copy(), a.copy())
    return best[1], best[2]


def tag_clusters(catalog: Catalog, cfg: ExploreConfig = ExploreConfig()) -> Clusters:
    def build() -> Clusters:
        M = np.array([[t in r.tags for t in CONTENT_TAGS] for r in catalog.routes], dtype=np.float64)
        idf = np.log(len(M) / (1.0 + M.sum(0)))
        X = M * np.maximum(idf, 0.0)
        norm = np.linalg.norm(X, axis=1)
        has = norm > 0
        X[has] /= norm[has][:, None]
        assign = np.full(len(M), cfg.k, dtype=np.int64)
        k = min(cfg.k, int(has.sum()))
        labels: list[str] = []
        if k:
            C, a = _kmeans(X[has], k, cfg.restarts, cfg.seed)
            # order clusters by size (largest first) so ids are stable and readable
            sizes = np.bincount(a, minlength=k)
            order = np.argsort(-sizes, kind="stable")
            remap = np.empty(k, dtype=np.int64)
            remap[order] = np.arange(k)
            assign[has] = remap[a]
            for c in order:
                top = np.argsort(-C[c], kind="stable")[:3]
                labels.append(" / ".join(CONTENT_TAGS[j] for j in top))
        labels += ["no theme tags"] * (cfg.k - k + 1)
        sizes_all = tuple(int((assign == c).sum()) for c in range(len(labels)))
        return Clusters(assign, tuple(labels), sizes_all)

    return catalog.memo(f"explore:clusters:{cfg.k}:{cfg.restarts}:{cfg.seed}", build)


def posteriors(clusters: Clusters, events: Sequence[WeightedEvent], cfg: ExploreConfig) -> np.ndarray:
    """(n_clusters, 2) array of Beta parameters (a, b) after the user's decayed feedback."""
    ab = np.tile([cfg.prior_a, cfg.prior_b], (clusters.n, 1)).astype(np.float64)
    for e in events:
        c = int(clusters.assign[e.index])
        if e.reason == POSITIVE_REASON:
            ab[c, 0] += e.weight
        elif e.reason == "not-interested":
            ab[c, 1] += e.weight
    return ab


def _rng(seed: int, request_id: str, n_events: int) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}|{request_id}|{n_events}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "big"))


def pick(
    catalog: Catalog,
    clusters: Clusters,
    cfg: ExploreConfig,
    *,
    request_id: str,
    events: Sequence[WeightedEvent],
    ranked_idx: Sequence[int],
    eligible_idx: np.ndarray,
    excluded_ids: set[int],
    similarity: Optional[np.ndarray] = None,
) -> Optional[dict[str, Any]]:
    """Choose the exploration route. ``ranked_idx``: catalogue indices in the pipeline's final order
    (all candidates); ``eligible_idx``: the filtered eligible set; ``similarity``: aligned with
    ``eligible_idx`` (query cosine) or None. Returns {"index", "cluster", ...} or None."""
    if len(events) >= cfg.max_events:
        return None
    head = [int(i) for i in ranked_idx[: cfg.position - 1]]
    tail = [int(i) for i in ranked_idx[cfg.position - 1 :]]
    head_set = set(head)
    elig = [int(i) for i in eligible_idx if int(catalog.ids[i]) not in excluded_ids and int(i) not in head_set]
    if not elig:
        return None
    have = {int(clusters.assign[i]) for i in head}
    avail = sorted({int(clusters.assign[i]) for i in elig})
    allowed = [c for c in avail if c not in have] or avail
    ab = posteriors(clusters, events, cfg)
    rng = _rng(cfg.seed, request_id, len(events))
    theta = rng.beta(ab[:, 0], ab[:, 1])  # one draw per cluster, always all clusters (stable stream)
    c = max(allowed, key=lambda x: (theta[x], -x))
    from_tail = [i for i in tail if int(clusters.assign[i]) == c and int(catalog.ids[i]) not in excluded_ids]
    if from_tail:
        i, source = int(from_tail[0]), "candidate"
    else:
        members = [i for i in elig if int(clusters.assign[i]) == c]
        if similarity is not None:
            pos = {int(e): p for p, e in enumerate(eligible_idx)}
            i = max(members, key=lambda m: (float(similarity[pos[m]]), -m))
            source = "eligible set, most similar to the query"
        else:
            i = int(members[int(rng.integers(len(members)))])
            source = "eligible set, random"
    return {
        "index": i,
        "cluster": c,
        "label": clusters.labels[c],
        "theta": round(float(theta[c]), 6),
        "posterior": [round(float(ab[c, 0]), 4), round(float(ab[c, 1]), 4)],
        "source": source,
        "position": cfg.position,
        "reason": EXPLORE_REASON,
    }
