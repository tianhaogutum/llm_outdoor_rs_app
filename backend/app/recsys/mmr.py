"""Maximal marginal relevance (Carbonell and Goldstein 1998) over the final list.

Greedy: at each step pick the candidate maximising

    lambda * rel(d) - (1 - lambda) * max_{s in selected} sim(d, s)

with ``rel`` the ranker score min-max normalised to [0, 1] over the candidate list and ``sim`` the
cosine similarity of the routes' document embeddings, clipped at 0. lambda = 1 keeps the ranker's
order. The diversity metric ILD@10 is computed on catalogue tag sets, not on embeddings, so MMR does
not optimise the reported metric directly. Ties are broken by the input order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def mmr(
    relevance: Sequence[float], sim: np.ndarray, lam: float, k: int | None = None
) -> tuple[list[int], list[dict[str, Any]]]:
    """Returns (selected positions in input order indices, per-pick details)."""
    n = len(relevance)
    if n == 0:
        return [], []
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    k = n if k is None else min(k, n)
    rel = np.asarray(relevance, dtype=np.float64)
    lo, hi = float(rel.min()), float(rel.max())
    reln = (rel - lo) / (hi - lo) if hi > lo else np.ones(n)
    S = np.clip(np.asarray(sim, dtype=np.float64), 0.0, None)
    chosen: list[int] = []
    details: list[dict[str, Any]] = []
    maxsim = np.zeros(n)
    left = np.ones(n, dtype=bool)
    for _ in range(k):
        score = lam * reln - (1 - lam) * maxsim
        score[~left] = -np.inf
        j = int(np.argmax(score))  # first maximum = earliest input position
        details.append(
            {
                "relevance": round(float(reln[j]), 6),
                "max_similarity": round(float(maxsim[j]), 6) if chosen else 0.0,
                "mmr_score": round(float(score[j]), 6),
                "penalty": round(float((1 - lam) * maxsim[j]), 6),
            }
        )
        chosen.append(j)
        left[j] = False
        maxsim = np.maximum(maxsim, S[j])
    return chosen, details
