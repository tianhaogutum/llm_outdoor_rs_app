"""Candidate-generation experiments on the **dev** split (Phase 3b): BM25 variants, embedded document
formulations, RRF k, and the reduced precision of the committed embedding matrix.

    cd backend && python -m evalkit.cg embed   # embed the catalogue per formulation (cached); write the artefact
    cd backend && python -m evalkit.cg tune    # dev experiments -> eval/results/_cg_dev.json

Measured without new judgements, against the current qrels: Recall@50 (pool-relative) and a
*condensed* NDCG@10 (unjudged routes removed from the ranking before cutting at 10; Sakai 2007), so
variants can be compared for free. Pool bias: ``pool_intent_lexical`` is plain BM25 (the ``plain``
variant here), so the pool favours that variant's results; the comparison leans towards it, not away.
Test personas are never read here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np

from app.llm import run_sync
from app.recsys import Catalog, get_catalog
from app.recsys.bm25 import BM25Config, get_index
from app.recsys.dense import DOC_EMBEDDINGS_FILE, DocEmbeddings, build_doc_embeddings
from app.recsys.fusion import rrf
from app.recsys.text import DOC_FORMULATIONS
from evalkit import metrics as M
from evalkit import paths
from evalkit.evaluate import dumps, system_client
from evalkit.personas import Persona, load_personas
from evalkit.qrels import Qrels

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
CHOSEN_FORMULATION = "facts"  # decided by `tune` on dev (eval/results/_cg_dev.json)
OUT = "_cg_dev.json"

BM25_VARIANTS: dict[str, BM25Config] = {
    "plain": BM25Config(stem=False, glossary=False, negation=False, expand=False),
    "+stem": BM25Config(stem=True, glossary=False, negation=False, expand=False),
    "+stem+glossary": BM25Config(stem=True, glossary=True, negation=False, expand=False),
    "+stem+glossary+negation": BM25Config(stem=True, glossary=True, negation=True, expand=False),
    "+stem+glossary+negation+expand": BM25Config(stem=True, glossary=True, negation=True, expand=True),
    "+stem+negation+expand (no glossary)": BM25Config(stem=True, glossary=False, negation=True, expand=True),
    "no glossary + tags x2": BM25Config(glossary=False, expand=True, w_tags=2.0),
    "no glossary + title x0.5, tags x2": BM25Config(glossary=False, expand=True, w_title=0.5, w_tags=2.0),
    "all + tags x2": BM25Config(expand=True, w_tags=2.0),
    "all + title x0.5, tags x2": BM25Config(expand=True, w_title=0.5, w_tags=2.0),
    "all + k1 1.5, b 0.5": BM25Config(expand=True, k1=1.5, b=0.5),
    "fields: title x2": BM25Config(w_title=2.0),
    "fields: tags x2": BM25Config(w_tags=2.0),
    "fields: title x0.5, tags x2": BM25Config(w_title=0.5, w_tags=2.0),
    "fields: description x0 (title+tags only)": BM25Config(w_description=0.0),
}


def condensed_ndcg(ranking: list[int], qrels: dict[int, int], k: int = 10) -> float | None:
    judged = [r for r in ranking if r in qrels][:k]
    return M.ndcg_at_k(judged, qrels, k)


def _score(rankings: dict[str, list[int]], qp: dict[str, dict[int, int]]) -> dict[str, Any]:
    rec = [M.recall_at_k(r, qp.get(pid, {}), 50) for pid, r in rankings.items()]
    nd = [condensed_ndcg(r, qp.get(pid, {})) for pid, r in rankings.items()]
    j10 = [np.mean([x in qp.get(pid, {}) for x in r[:10]]) for pid, r in rankings.items()]
    return {
        "recall_at_50": M.bootstrap_ci(rec, n_boot=1000, seed=20261122),
        "condensed_ndcg_at_10": M.bootstrap_ci(nd, n_boot=1000, seed=20261122),
        "judged_at_10": float(np.mean(j10)),
        "n": len(rankings),
    }


def _paired(a: dict[str, list[int]], b: dict[str, list[int]], qp) -> dict[str, Any]:
    pids = sorted(a)
    ra = [M.recall_at_k(a[p], qp.get(p, {}), 50) for p in pids]
    rb = [M.recall_at_k(b[p], qp.get(p, {}), 50) for p in pids]
    return M.paired_bootstrap(ra, rb, n_boot=1000, seed=20261122)


def _eligible(catalog: Catalog, p: Persona) -> np.ndarray:
    from app.recsys.filters import apply_filters

    return apply_filters(catalog, p.constraints()).indices


def bm25_rankings(catalog: Catalog, personas: list[Persona], cfg: BM25Config, elig: dict[str, np.ndarray]):
    idx = get_index(catalog, cfg)
    out = {}
    for p in personas:
        e = elig[p.persona_id]
        sc, _ = idx.score_all(idx.query_terms(p.intent or ""))
        order = e[np.argsort(-sc[e], kind="stable")]
        out[p.persona_id] = [int(catalog.ids[i]) for i in order[:100]]
    return out


def query_vectors(personas: list[Persona]) -> dict[str, np.ndarray]:
    """One embed call per intent (as the pipeline makes it), so cached latency is per request."""
    client = system_client()

    async def go():
        import asyncio

        res = await asyncio.gather(
            *(
                client.embed([p.intent], task="query", model=_model(client), dim=EMBED_DIM, tag="query-embed")
                for p in personas
            )
        )
        return {p.persona_id: np.asarray(r.vectors[0]) for p, r in zip(personas, res, strict=True)}

    return run_sync(go())


def _model(client) -> str | None:
    return None if client.config.embed_provider == "mock" else EMBED_MODEL


def doc_embeddings(catalog: Catalog, formulation: str) -> DocEmbeddings:
    client = system_client()
    return build_doc_embeddings(
        catalog, client, formulation=formulation, model=_model(client), dim=EMBED_DIM, tag="doc-embed"
    )


def dense_rankings(catalog, personas, docs: DocEmbeddings, qv, elig):
    out = {}
    for p in personas:
        e = elig[p.persona_id]
        s = docs.matrix[e] @ qv[p.persona_id]
        out[p.persona_id] = [int(catalog.ids[i]) for i in e[np.argsort(-s, kind="stable")[:100]]]
    return out


# --------------------------------------------------------------------------- commands
def cmd_embed() -> None:
    catalog = get_catalog()
    reports = {}
    for f in DOC_FORMULATIONS:
        t0 = time.time()
        d = doc_embeddings(catalog, f)
        reports[f] = {"seconds": round(time.time() - t0, 1), **d.meta}
        if f == CHOSEN_FORMULATION:
            d.save(DOC_EMBEDDINGS_FILE, precision=COMMITTED_PRECISION)
    print(json.dumps(reports, indent=1))


COMMITTED_PRECISION = "float16"  # int8 is half the size but reorders more near-ties (see quantisation_check)


def _reduced(m: np.ndarray, precision: str) -> np.ndarray:
    from app.recsys.dense import dequantise, quantise_int8

    if precision == "int8":
        return dequantise(*quantise_int8(m))
    r = m.astype(np.float16).astype(np.float32)
    return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)


def quantisation_check(catalog, personas, docs32: DocEmbeddings, qv, elig) -> dict[str, Any]:
    """Rankings from the float32 matrix vs reduced-precision copies, on dev queries over the eligible sets."""
    out: dict[str, Any] = {
        "committed_precision": COMMITTED_PRECISION,
        "committed_file_bytes": DOC_EMBEDDINGS_FILE.stat().st_size,
        "float32_bytes": int(docs32.matrix.nbytes),
        "n_queries": len(personas),
    }
    for prec in ("float16", "int8"):
        red = _reduced(docs32.matrix, prec)
        over10, same10, same50, maxdiff = [], [], [], []
        for p in personas:
            e = elig[p.persona_id]
            a = docs32.matrix[e] @ qv[p.persona_id]
            b = red[e] @ qv[p.persona_id]
            ra, rb = np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")
            over10.append(len(set(ra[:10]) & set(rb[:10])) / min(10, len(e)))
            same10.append(bool(np.array_equal(ra[:10], rb[:10])))
            same50.append(bool(np.array_equal(ra[:50], rb[:50])))
            maxdiff.append(float(np.max(np.abs(a - b))))
        out[prec] = {
            "top10_overlap_mean": float(np.mean(over10)),
            "top10_identical_order_share": float(np.mean(same10)),
            "top50_identical_order_share": float(np.mean(same50)),
            "max_abs_cosine_diff": float(np.max(maxdiff)),
            "median_max_abs_cosine_diff": float(np.median(maxdiff)),
        }
    e = elig[personas[0].persona_id]
    sims = docs32.matrix[e] @ qv[personas[0].persona_id]
    out["example_cosine_range"] = [float(sims.min()), float(np.median(sims)), float(sims.max())]
    return out


def exact_search_timing(docs: DocEmbeddings, qv: dict[str, np.ndarray]) -> dict[str, Any]:
    """Time exact cosine over the full catalogue (worst case: no filter) and project it linearly."""
    q = np.stack(list(qv.values())[:50])
    M_ = docs.matrix
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for v in q:
            s = M_ @ v
            np.argpartition(-s, 100)[:100]
        reps.append((time.perf_counter() - t0) * 1000 / len(q))
    ms = float(np.median(reps))
    return {
        "n_docs": int(M_.shape[0]),
        "dim": int(M_.shape[1]),
        "ms_per_query_full_catalogue": ms,
        "ns_per_doc": ms * 1e6 / M_.shape[0],
        "machine_note": "local laptop, numpy matmul, single query",
    }


def cmd_tune() -> None:
    catalog = get_catalog()
    personas = [p for p in load_personas() if p.split == "dev"]
    qrels = Qrels.load()
    qp = qrels.by_persona()
    elig = {p.persona_id: _eligible(catalog, p) for p in personas}
    out: dict[str, Any] = {
        "schema": "evalkit.cg_dev/v1",
        "split": "dev",
        "n_personas": len(personas),
        "qrels_sha256": qrels.sha256(),
        "protocol": "Recall@50 (pool-relative) and condensed NDCG@10 against the current qrels; dev only",
    }
    # BM25 variants
    bm = {}
    base = None
    for name, cfg in BM25_VARIANTS.items():
        r = bm25_rankings(catalog, personas, cfg, elig)
        bm[name] = {"config": cfg.describe(), **_score(r, qp)}
        if name == "plain":
            base = r
        else:
            bm[name]["recall_vs_plain"] = _paired(base, r, qp)
        print(
            f"[cg] bm25 {name}: R@50 {bm[name]['recall_at_50']['mean']:.3f} "
            f"cNDCG {bm[name]['condensed_ndcg_at_10']['mean']:.3f}",
            file=sys.stderr,
        )
    out["bm25"] = bm
    # dense formulations
    qv = query_vectors(personas)
    dense, drank = {}, {}
    mats = {}
    for f in DOC_FORMULATIONS:
        d = doc_embeddings(catalog, f)
        mats[f] = d
        r = dense_rankings(catalog, personas, d, qv, elig)
        drank[f] = r
        dense[f] = {"meta": {k: d.meta[k] for k in ("model", "dim", "formulation", "task")}, **_score(r, qp)}
        print(
            f"[cg] dense {f}: R@50 {dense[f]['recall_at_50']['mean']:.3f} "
            f"cNDCG {dense[f]['condensed_ndcg_at_10']['mean']:.3f}",
            file=sys.stderr,
        )
    fs = list(DOC_FORMULATIONS)
    dense["paired_recall"] = {f"{fs[1]} - {fs[0]}": _paired(drank[fs[0]], drank[fs[1]], qp)}
    out["dense"] = dense
    # RRF k (chosen BM25 config, chosen formulation)
    from app.recsys.systems.pipelines import BM25_DEFAULT

    br = bm25_rankings(catalog, personas, BM25_DEFAULT, elig)
    fr = {}
    for k in (10, 30, 60, 100):
        fused = {
            pid: [i for i, _, _ in rrf({"bm25": br[pid], "dense": drank[CHOSEN_FORMULATION][pid]}, k=k)][:50]
            for pid in br
        }
        fr[str(k)] = _score(fused, qp)
    out["rrf_k"] = fr
    out["chosen"] = {"bm25": BM25_DEFAULT.describe(), "formulation": CHOSEN_FORMULATION}
    if DOC_EMBEDDINGS_FILE.exists():
        out["quantisation"] = quantisation_check(catalog, personas, mats[CHOSEN_FORMULATION], qv, elig)
        out["exact_search"] = exact_search_timing(DocEmbeddings.load(DOC_EMBEDDINGS_FILE).aligned_to(catalog), qv)
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS_DIR / OUT).write_text(dumps(out), encoding="utf-8")
    print(f"wrote {paths.RESULTS_DIR / OUT}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["embed", "tune"])
    args = ap.parse_args(argv)
    {"embed": cmd_embed, "tune": cmd_tune}[args.cmd]()


if __name__ == "__main__":
    main()
