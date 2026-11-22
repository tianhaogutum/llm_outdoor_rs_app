"""Ranker-API pilot on **dev** personas: which OpenRouter reranker, and which document text.

    cd backend && python -m evalkit.rank_pilot      # -> eval/results/_rank_pilot.json

Every candidate model re-ranks the *same* candidate lists (the ``v4_rrf`` fused top 50 of 40 seeded dev
personas), so ordering quality can be compared without new judgements: condensed NDCG@10 over the
judged candidates (unjudged ones removed, Sakai 2007), identical judged subsets for every model. Cost
is OpenRouter's reported charge per call (``usage.cost``), latency the measured request time.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import numpy as np

from app.llm import run_sync
from app.recsys import get_catalog
from app.recsys.text import ranker_text
from evalkit import metrics as M
from evalkit import paths
from evalkit.cg import condensed_ndcg
from evalkit.evaluate import dumps, make_system, system_client
from evalkit.personas import load_personas
from evalkit.qrels import Qrels

N_PERSONAS = 40
SEED = 20261122
MODELS = [
    "cohere/rerank-v3.5",
    "cohere/rerank-4-fast",
    "cohere/rerank-4-pro",
    "cohere/rerank-5-fast",
    "cohere/rerank-5-pro",
    "mongodb/rerank-3-lite",
    "mongodb/rerank-3",
    "qwen/qwen3-reranker-8b",
]
DOC_FORMULATIONS = ("facts_desc", "desc")
OUT = "_rank_pilot.json"


def compute() -> dict[str, Any]:
    catalog = get_catalog()
    dev = [p for p in load_personas() if p.split == "dev"]
    rng = np.random.default_rng(SEED)
    personas = [dev[i] for i in sorted(rng.choice(len(dev), N_PERSONAS, replace=False))]
    qp = Qrels.load().by_persona()
    v4 = make_system("v4_rrf", catalog)
    cands = {p.persona_id: v4.recommend(p.to_query(), 50).route_ids for p in personas}
    client = system_client()

    def evaluate(model: str, doc: str) -> dict[str, Any]:
        async def go():
            async def one(p):
                docs = [ranker_text(catalog.get(r), doc) for r in cands[p.persona_id]]
                try:
                    res = await client.rank(p.intent, docs, model=model, tag=f"rank-pilot:{model}:{doc}")
                except Exception as e:  # a model may be unavailable; record it
                    return p, None, f"{type(e).__name__}: {str(e)[:200]}"
                return p, res, None

            return await asyncio.gather(*(one(p) for p in personas))

        rows = run_sync(go())
        errs = [e for _, r, e in rows if r is None]
        if errs:
            return {"model": model, "doc": doc, "error": errs[0], "n_errors": len(errs)}
        nd, base, lat, cost, reported = [], [], [], [], set()
        for p, res, _ in rows:
            ids = cands[p.persona_id]
            order = [ids[i] for i in np.argsort(-np.asarray(res.scores), kind="stable")]
            g = qp.get(p.persona_id, {})
            nd.append(condensed_ndcg(order, g))
            base.append(condensed_ndcg(ids, g))
            lat.append(res.nominal_latency_ms)
            cost.append(res.nominal_cost_usd)
            reported.add(res.model)
        return {
            "model": model,
            "doc": doc,
            "reported_models": sorted(reported),
            "condensed_ndcg_at_10": M.bootstrap_ci(nd, n_boot=1000, seed=SEED),
            "vs_rrf_order": M.paired_bootstrap(base, nd, n_boot=1000, seed=SEED),
            "latency_ms": {"p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95)},
            "cost_usd_per_call": float(np.mean(cost)),
        }

    results = []
    for m in MODELS:
        r = evaluate(m, DOC_FORMULATIONS[0])
        results.append(r)
        print(
            f"[rank-pilot] {m}: {r.get('condensed_ndcg_at_10', {}).get('mean')} {r.get('error', '')}", file=sys.stderr
        )
    ok = [r for r in results if "error" not in r]
    best = max(ok, key=lambda r: r["condensed_ndcg_at_10"]["mean"])
    doc_cmp = [best] + [evaluate(best["model"], d) for d in DOC_FORMULATIONS[1:]]
    base_nd = [condensed_ndcg(cands[p.persona_id], qp.get(p.persona_id, {})) for p in personas]
    out = {
        "schema": "evalkit.rank_pilot/v1",
        "split": "dev",
        "n_personas": N_PERSONAS,
        "personas": [p.persona_id for p in personas],
        "candidates": "v4_rrf fused top 50",
        "rrf_order": M.bootstrap_ci(base_nd, n_boot=1000, seed=SEED),
        "models": results,
        "doc_formulations": doc_cmp,
        "best_by_condensed_ndcg": best["model"],
    }
    return out


def replay(check: bool) -> tuple[list[str], list[str]]:
    """For freeze/replay-check: re-issue the pilot's calls; in check mode compare with the committed file."""
    f = paths.RESULTS_DIR / OUT
    if not f.exists():
        return [], []
    client = system_client()
    client.reset_touched()
    out = compute()
    keys = client.touched_keys()
    probs = []
    if check and json.loads(dumps(out)) != json.loads(f.read_text(encoding="utf-8")):
        probs.append("ranker pilot summary differs")
    return keys, probs


def main() -> None:
    out = compute()
    results = out["models"]
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS_DIR / OUT).write_text(dumps(out), encoding="utf-8")
    print(
        json.dumps(
            {
                r["model"]: (
                    r.get("condensed_ndcg_at_10", {}).get("mean"),
                    r.get("cost_usd_per_call"),
                    r.get("latency_ms"),
                )
                for r in results
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
