"""Reciprocal-rank fusion (Cormack, Clarke and Buettcher 2009).

score(d) = sum over input rankings r of 1 / (k + rank_r(d)), ranks starting at 1; a document missing
from a ranking contributes nothing for it. RRF uses ranks only, so it needs no calibration between
BM25 scores (unbounded, query-dependent) and cosine similarities (about 0.5-0.8 here); a weighted sum
of raw scores would need per-query normalisation and a tuned weight. Ties are broken by the best
single rank, then by the input order of the first ranking, then by id (deterministic).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def rrf(rankings: Mapping[str, Sequence[int]], k: int = 60, depth: int | None = None) -> list[tuple[int, float, dict]]:
    """Fuse ``{name: [item, ...]}`` (best first). Returns ``[(item, score, {name: rank}), ...]`` best first."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, items in rankings.items():
        for pos, item in enumerate(items[:depth] if depth else items, start=1):
            if name in ranks.get(item, {}):
                continue  # duplicates within one ranking count once
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos)
            ranks.setdefault(item, {})[name] = pos
    first = {item: i for i, item in enumerate(next(iter(rankings.values()), []))} if rankings else {}
    out = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], min(ranks[kv[0]].values()), first.get(kv[0], 1 << 30), kv[0]),
    )
    return [(item, s, ranks[item]) for item, s in out]
