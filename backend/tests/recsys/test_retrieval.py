"""Phase 3b: text processing, BM25F (hand-computed), RRF, dense matrix io and exact search."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.recsys import Catalog
from app.recsys.bm25 import BM25Config, BM25Index, matched_terms
from app.recsys.dense import DocEmbeddings, dequantise, quantise_int8
from app.recsys.fusion import rrf
from app.recsys.text import glossary_terms, query_tokens, remove_negated, split_negated, stem
from tests.recsys.factory import route_record


def _cat(texts: list[tuple[str, str, list[str]]]) -> Catalog:
    recs = []
    for i, (title, desc, tags) in enumerate(texts):
        r = route_record(i + 1, tags=tags)
        r["title"], r["description"] = title, desc
        recs.append(r)
    return Catalog.from_records(recs)


# ----------------------------------------------------------------------------- text
def test_stem_and_negation():
    assert [stem(w) for w in ("lakes", "ruins", "castles", "running", "glass", "views", "hiking")] == [
        "lake", "ruin", "castle", "run", "glass", "view", "hik",
    ]
    assert remove_negated("A lake loop, no busy roads and a beer garden") == "A lake loop, and a beer garden"
    kept, dropped = split_negated("quiet forest without steep climbs, with a hut")
    assert "steep" not in kept and "hut" in kept and "steep climbs" in dropped
    toks = query_tokens("A loop with lakes, no busy roads", do_stem=True, negation=True, expand=True)
    assert "lake" in toks and "loop" in toks and "circular" in toks and "busy" not in toks and "road" not in toks


def test_glossary_decompounds_german_titles():
    assert glossary_terms("eibsee") == ["lake"]
    assert set(glossary_terms("staffelseerundweg")) >= {"lake", "loop"}
    assert set(glossary_terms("burgruine")) >= {"castle", "ruin"}
    assert glossary_terms("innenstadt") == []  # ambiguous short morphemes only as whole tokens


# ----------------------------------------------------------------------------- BM25
def test_bm25_matches_hand_computation():
    cat = _cat([("Lake walk", "lake lake shore", []), ("Hill", "summit view", []), ("Lakes", "a forest", [])])
    cfg = BM25Config(stem=True, glossary=False, negation=False, expand=False)
    idx = BM25Index(cat, cfg)
    scores, per_term = idx.score_all(["lake"])
    # single field each with weight 1: tf' = sum_f tf_f / (1 - b + b * len_f / avg_f)
    k1, b = cfg.k1, cfg.b
    t_len = np.array([2, 1, 1.0])  # "lake walk" -> [lake, walk]; "hill"; "lakes" -> lake
    d_len = np.array([3, 2, 1.0])  # "a" is a stopword
    tf0 = 1 / (1 - b + b * t_len[0] / t_len.mean()) + 2 / (1 - b + b * d_len[0] / d_len.mean())
    tf2 = 1 / (1 - b + b * t_len[2] / t_len.mean())
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    assert scores[0] == pytest.approx(idf * tf0 * (k1 + 1) / (tf0 + k1))
    assert scores[2] == pytest.approx(idf * tf2 * (k1 + 1) / (tf2 + k1))
    assert scores[1] == 0
    assert matched_terms(per_term, 0) == [{"term": "lake", "contribution": round(scores[0], 4)}]


def test_bm25_field_weights_and_unknown_terms():
    cat = _cat([("Castle trail", "a walk", []), ("Walk", "a castle on the way", [])])
    title_heavy = BM25Index(cat, BM25Config(w_title=3.0, w_description=1.0, stem=False, glossary=False))
    desc_only = BM25Index(cat, BM25Config(w_title=0.0, w_description=1.0, stem=False, glossary=False))
    s1, _ = title_heavy.score_all(["castle"])
    s2, _ = desc_only.score_all(["castle", "unknownterm"])
    assert s1[0] > s1[1] and s2[0] == 0 and s2[1] > 0


# ----------------------------------------------------------------------------- RRF
def test_rrf_scores_and_ties():
    fused = rrf({"bm25": [1, 2, 3], "dense": [3, 1, 4]}, k=60)
    got = {i: s for i, s, _ in fused}
    assert got[1] == pytest.approx(1 / 61 + 1 / 62)
    assert got[3] == pytest.approx(1 / 63 + 1 / 61)
    assert got[2] == pytest.approx(1 / 62) and got[4] == pytest.approx(1 / 63)
    assert [i for i, _, _ in fused] == [1, 3, 2, 4]
    assert fused[0][2] == {"bm25": 1, "dense": 2}
    # exact tie on score: better single rank first, then the first list's order
    tie = rrf({"a": [10, 20], "b": [20, 10]}, k=1)
    assert [i for i, _, _ in tie] == [10, 20]
    # duplicates inside one list count once; depth cuts the input
    assert {i for i, _, _ in rrf({"a": [1, 1, 2, 3]}, k=5, depth=2)} == {1}
    with pytest.raises(ValueError):
        rrf({"a": [1]}, k=0)


# ----------------------------------------------------------------------------- dense
def test_doc_embeddings_roundtrip_and_alignment(tmp_path):
    rng = np.random.default_rng(0)
    m = rng.normal(size=(5, 16)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    d = DocEmbeddings(np.array([5, 1, 4, 2, 3]), m, {"provider": "mock", "model": "x", "dim": 16})
    for prec, tol in (("float16", 1e-3), ("int8", 2e-2), ("float32", 1e-7)):
        p = tmp_path / f"e-{prec}.npz"
        d.save(p, precision=prec)
        again = DocEmbeddings.load(p)
        assert again.meta["precision"] == prec
        assert np.allclose(again.matrix, m, atol=tol)
        assert np.allclose(np.linalg.norm(again.matrix, axis=1), 1, atol=1e-5)
    b1 = (tmp_path / "e-float16.npz").read_bytes()
    d.save(tmp_path / "again.npz", precision="float16")
    assert (tmp_path / "again.npz").read_bytes() == b1  # deterministic file
    cat = Catalog.from_records([route_record(i) for i in (1, 2, 3, 4, 5)])
    al = d.aligned_to(cat)
    assert list(al.ids) == [1, 2, 3, 4, 5] and np.allclose(al.matrix[0], m[1])
    q = m[2]
    sims = al.cosine(q, np.arange(5))
    assert int(np.argmax(sims)) == 3 and sims[3] == pytest.approx(1.0, abs=1e-6)  # route 4 is row 2
    with pytest.raises(ValueError):
        DocEmbeddings(np.array([1, 2]), m[:2], {}).aligned_to(cat)


def test_int8_quantisation_error_is_small():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(50, 64)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    q, s = quantise_int8(m)
    assert q.dtype == np.int8 and np.abs(dequantise(q, s) - m).max() < 0.02
