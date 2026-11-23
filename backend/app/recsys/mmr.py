"""Maximal marginal relevance (Carbonell and Goldstein 1998) over the final list.

Greedy: at each step pick the candidate maximising

    lambda * rel(d) - (1 - lambda) * max_{s in selected} sim(d, s)

with ``rel`` the ranker score min-max normalised to [0, 1] over the candidate list and ``sim`` the
cosine similarity of the routes' document embeddings, also min-max normalised over the candidate pairs
(``normalise_sim``). Without that second normalisation lambda barely acts: route embeddings are all
alike (pairwise cosines of a candidate list sit in a narrow band), so the penalty is nearly the same
constant for every candidate (found on dev; docs/RECSYS.md section 5). lambda = 1 keeps the ranker's
order. The diversity metric ILD@10 is computed on catalogue tag sets, not on embeddings, so MMR does
not optimise the reported metric directly. Ties are broken by the input order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def mmr(
    relevance: Sequence[float], sim: np.ndarray, lam: float, k: int | None = None, normalise_sim: bool = True
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
    S = np.asarray(sim, dtype=np.float64)
    if normalise_sim and n > 1:
        off = S[~np.eye(n, dtype=bool)]
        lo_s, hi_s = float(off.min()), float(off.max())
        S = (S - lo_s) / (hi_s - lo_s) if hi_s > lo_s else np.zeros_like(S)
    S = np.clip(S, 0.0, 1.0)
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
