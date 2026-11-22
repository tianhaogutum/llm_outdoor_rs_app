"""Evaluation metrics: pure functions, unit-tested against hand-computed cases (tests/evalkit/test_metrics.py).

Conventions (docs/EVAL.md section "Metrics"):

- ``ranking``: route ids, best first. ``qrels``: {route_id: grade 0-3} for ONE persona (all judged pairs).
- NDCG@k: gain ``2**g - 1``, discount ``log2(rank + 1)``; ideal DCG from the persona's best ``k`` judged
  grades. Unjudged retrieved routes raise (every reported top-k must be fully judged).
- MRR@k: reciprocal rank of the first route with grade >= 2 within the top k, 0 if none.
- Recall@k: share of the persona's known grade >= 2 routes found in the top k (pool-relative).
- Intra-list diversity: mean pairwise (1 - Jaccard) over the routes' catalogue tag sets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Optional

import numpy as np

RELEVANT_GRADE = 2


class UnjudgedError(KeyError):
    pass


def dcg(grades: Sequence[int]) -> float:
    return float(sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades)))


def ndcg_at_k(ranking: Sequence[int], qrels: Mapping[int, int], k: int = 10) -> Optional[float]:
    """NDCG@k, or None if the persona has no judged route with grade > 0 (undefined, excluded)."""
    top = list(ranking[:k])
    missing = [r for r in top if r not in qrels]
    if missing:
        raise UnjudgedError(f"unjudged routes in top {k}: {missing[:5]}")
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([qrels[r] for r in top]) / ideal


def mrr_at_k(ranking: Sequence[int], qrels: Mapping[int, int], k: int = 10, min_grade: int = RELEVANT_GRADE) -> float:
    for i, r in enumerate(ranking[:k]):
        if r not in qrels:
            raise UnjudgedError(f"unjudged route {r} at rank {i + 1}")
        if qrels[r] >= min_grade:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    ranking: Sequence[int], qrels: Mapping[int, int], k: int = 50, min_grade: int = RELEVANT_GRADE
) -> Optional[float]:
    """Share of known relevant (grade >= min_grade) routes in the top k; None if none is known."""
    rel = {r for r, g in qrels.items() if g >= min_grade}
    if not rel:
        return None
    return len(rel & set(ranking[:k])) / len(rel)


def violation_rate_at_k(ranking: Sequence[int], violates: Callable[[int], bool], k: int = 10) -> Optional[float]:
    top = list(ranking[:k])
    if not top:
        return None
    return sum(1 for r in top if violates(r)) / len(top)


def jaccard_distance(a: frozenset | set, b: frozenset | set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def intra_list_diversity(tag_sets: Sequence[frozenset | set]) -> Optional[float]:
    """Mean pairwise Jaccard distance; None for fewer than two items."""
    n = len(tag_sets)
    if n < 2:
        return None
    s = sum(jaccard_distance(tag_sets[i], tag_sets[j]) for i in range(n) for j in range(i + 1, n))
    return s / (n * (n - 1) / 2)


def coverage(rankings: Sequence[Sequence[int]], catalogue_size: int, k: int = 10) -> float:
    """Distinct routes in all top-k lists / catalogue size."""
    seen = {r for rk in rankings for r in rk[:k]}
    return len(seen) / catalogue_size if catalogue_size else 0.0


def percentile(xs: Sequence[float], q: float) -> Optional[float]:
    if not len(xs):
        return None
    return float(np.percentile(np.asarray(xs, dtype=float), q))


# --------------------------------------------------------------------------- agreement
def confusion_matrix(a: Sequence[int], b: Sequence[int], n_classes: int = 4) -> list[list[int]]:
    """m[i][j] = number of items with a == i and b == j."""
    m = [[0] * n_classes for _ in range(n_classes)]
    for x, y in zip(a, b, strict=True):
        m[int(x)][int(y)] += 1
    return m


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], n_classes: int = 4, weights: str = "quadratic"
) -> Optional[float]:
    """Cohen's kappa with quadratic (or linear) disagreement weights; None if undefined."""
    if len(a) == 0:
        return None
    o = np.array(confusion_matrix(a, b, n_classes), dtype=float)
    n = o.sum()
    idx = np.arange(n_classes)
    d = np.abs(idx[:, None] - idx[None, :]).astype(float)
    w = d**2 if weights == "quadratic" else d
    w /= max(w.max(), 1.0)
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    denom = (w * e).sum()
    if denom == 0:
        return None
    return float(1.0 - (w * o).sum() / denom)


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, Optional[float]]:
    """Percentile bootstrap over units (personas). Returns mean, lo, hi, n."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return {
        "mean": float(stat(x)),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(len(x)),
    }


def paired_bootstrap(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Optional[float]]:
    """Bootstrap of mean(b - a) over the units where both are defined (same resample for both systems).

    ``p_le_0``: share of bootstrap means <= 0 (a one-sided indication that b does not beat a)."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return {"diff": None, "lo": None, "hi": None, "n": 0, "p_le_0": None}
    d = np.array([y - x for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(len(d)),
        "p_le_0": float((boots <= 0).mean()),
    }
