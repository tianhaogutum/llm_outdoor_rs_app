"""BM25F over title + description + tags (numpy postings, IDF over the whole catalogue).

Per field f with weight w_f and length normalisation b_f the pseudo term frequency is

    tf'(t, d) = sum_f w_f * tf_f(t, d) / (1 - b_f + b_f * len_f(d) / avglen_f)

and score(q, d) = sum_{t in q} idf(t) * tf' * (k1 + 1) / (tf' + k1), idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)).
With all weights 1 and a single field this is classic BM25. Query terms are used once each (a term
repeated in the query is weighted by its count). Scoring touches only the postings of the query terms,
so it is O(sum of their document frequencies), about a millisecond for the 9.3k-route catalogue.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .catalog import Catalog
from .text import TEXT_VERSION, bm25_fields, query_tokens

FIELDS = ("title", "description", "tags")


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    w_title: float = 1.0
    w_description: float = 1.0
    w_tags: float = 1.0
    stem: bool = True
    glossary: bool = True
    negation: bool = True
    expand: bool = False

    def weights(self) -> dict[str, float]:
        return {"title": self.w_title, "description": self.w_description, "tags": self.w_tags}

    def describe(self) -> dict:
        return {**asdict(self), "text_version": TEXT_VERSION}

    def key(self) -> str:
        return "bm25:" + ",".join(f"{k}={v}" for k, v in sorted(asdict(self).items()))


class BM25Index:
    def __init__(self, catalog: Catalog, cfg: BM25Config) -> None:
        self.cfg = cfg
        n = len(catalog)
        self.n = n
        docs = [bm25_fields(r, glossary=cfg.glossary, do_stem=cfg.stem) for r in catalog.routes]
        w = cfg.weights()
        # pseudo-tf per term: dict term -> (doc indices, tf') ; built per field then summed
        acc: dict[str, dict[int, float]] = {}
        for f in FIELDS:
            lens = np.array([len(d[f]) for d in docs], dtype=np.float64)
            avg = float(lens.mean()) or 1.0
            if w[f] == 0:
                continue
            for i, d in enumerate(docs):
                norm = 1 - cfg.b + cfg.b * lens[i] / avg
                for t, c in Counter(d[f]).items():
                    slot = acc.setdefault(t, {})
                    slot[i] = slot.get(i, 0.0) + w[f] * c / norm
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for t, m in acc.items():
            idx = np.fromiter(m.keys(), dtype=np.int64, count=len(m))
            tf = np.fromiter(m.values(), dtype=np.float64, count=len(m))
            order = np.argsort(idx)
            self.postings[t] = (idx[order], tf[order])
            df = len(m)
            self.idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query_terms(self, intent: str) -> list[str]:
        c = self.cfg
        return query_tokens(intent, do_stem=c.stem, negation=c.negation, expand=c.expand)

    def score_all(self, terms: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Scores for every catalogue route, plus each known term's per-route contribution (dense arrays)."""
        k1 = self.cfg.k1
        total = np.zeros(self.n, dtype=np.float64)
        per_term: dict[str, np.ndarray] = {}
        for t, qtf in Counter(terms).items():
            if t not in self.postings:
                continue
            idx, tf = self.postings[t]
            contrib = np.zeros(self.n, dtype=np.float64)
            contrib[idx] = qtf * self.idf[t] * tf * (k1 + 1) / (tf + k1)
            total += contrib
            per_term[t] = contrib
        return total, per_term


def get_index(catalog: Catalog, cfg: BM25Config) -> BM25Index:
    """Index per (catalogue, config), built once per process (about 2 s for the Bavaria catalogue)."""
    return catalog.memo(cfg.key(), lambda: BM25Index(catalog, cfg))


def matched_terms(per_term: dict[str, np.ndarray], i: int, top: int = 8) -> list[dict]:
    """The query terms that matched route index ``i``, largest contribution first."""
    hits = [(t, float(v[i])) for t, v in per_term.items() if v[i] > 0]
    hits.sort(key=lambda x: (-x[1], x[0]))
    return [{"term": t, "contribution": round(v, 4)} for t, v in hits[:top]]
