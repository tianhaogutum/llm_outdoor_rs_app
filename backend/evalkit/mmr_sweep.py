"""MMR lambda sweep on the **dev** split (Phase 3d): NDCG@10 vs intra-list diversity.

    cd backend && python -m evalkit.mmr_sweep --estimate   # holes and judge cost, writes nothing
    cd backend && python -m evalkit.mmr_sweep              # judge dev holes, -> eval/results/_mmr_sweep.json
                                                           #    and docs/mmr_tradeoff.png

Each lambda is the base system (``v6_ltr_noapi``, the default ranker) with ``mmr_lambda`` set;
lambda = 1.0 is the base itself.
Every top 10 is judged (holes go to the production judge), so NDCG@10 is exact, not condensed.
Test personas are never run here.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.recsys import get_catalog
from evalkit import metrics as M
from evalkit import paths
from evalkit.evaluate import JUDGE_COST_PER_PAIR_USD, dumps, run_system
from evalkit.judge import rule_grade
from evalkit.personas import load_personas
from evalkit.pool import POOL_DEPTH, pool_pairs
from evalkit.qrels import Qrels, fill_holes

BASE = "v6_ltr_noapi"  # the default ranker (docs/RECSYS.md: no ranker API in the default)
LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3)
OUT = "_mmr_sweep.json"
FIGURE = paths.DOCS_DIR / "mmr_tradeoff.png"


def _runs(catalog, personas) -> dict[float, dict[str, Any]]:
    out = {}
    for lam in LAMBDAS:
        cfg = {} if lam == 1.0 else {"mmr_lambda": lam}
        out[lam] = run_system(BASE, catalog, personas, depth=10, **cfg)[0]
    return out


def compute(judge: bool = True) -> dict[str, Any]:
    catalog = get_catalog()
    personas = [p for p in load_personas() if p.split == "dev"]
    pm = {p.persona_id: p for p in personas}
    runs = _runs(catalog, personas)
    qrels = Qrels.load()
    pool = pool_pairs({str(lam): {pid: r["ids"] for pid, r in run.items()} for lam, run in runs.items()}, POOL_DEPTH)
    holes = qrels.holes(pool)
    if holes:
        if not judge:
            raise SystemExit(f"{len(holes)} unjudged pairs")
        rep = fill_holes(qrels, holes, pm, catalog)
        qrels.save()
        print(f"[mmr] judged holes {rep}", file=sys.stderr)
    qp = qrels.by_persona()
    rows = []
    base_nd = None
    for lam in LAMBDAS:
        run = runs[lam]
        nd = [M.ndcg_at_k(run[p.persona_id]["ids"], qp[p.persona_id], 10) for p in personas]
        ild = [
            M.intra_list_diversity([frozenset(catalog.get(r).tags) for r in run[p.persona_id]["ids"][:10]])
            for p in personas
        ]
        row = {
            "lambda": lam,
            "ndcg_at_10": M.bootstrap_ci(nd, n_boot=1000, seed=20261122),
            "ild_at_10": M.bootstrap_ci(ild, n_boot=1000, seed=20261122),
            "coverage_at_10": M.coverage([run[p.persona_id]["ids"] for p in personas], len(catalog), 10),
        }
        if base_nd is None:
            base_nd = nd
        else:
            row["ndcg_vs_lambda_1"] = M.paired_bootstrap(base_nd, nd, n_boot=1000, seed=20261122)
        rows.append(row)
    return {
        "schema": "evalkit.mmr_sweep/v1",
        "split": "dev",
        "base": BASE,
        "n_personas": len(personas),
        "similarity": "document-embedding cosine",
        "diversity_metric": "ILD@10 = mean pairwise tag-Jaccard distance",
        "rows": rows,
    }


def estimate() -> dict[str, Any]:
    catalog = get_catalog()
    personas = [p for p in load_personas() if p.split == "dev"]
    pm = {p.persona_id: p for p in personas}
    runs = _runs(catalog, personas)
    pool = pool_pairs({str(lam): {pid: r["ids"] for pid, r in run.items()} for lam, run in runs.items()}, POOL_DEPTH)
    holes = Qrels.load().holes(pool)
    llm = [h for h in holes if rule_grade(pm[h[0]], catalog, h[1]) is None]
    return {"holes": len(holes), "llm_pairs": len(llm), "est_judge_usd": round(len(llm) * JUDGE_COST_PER_PAIR_USD, 2)}


def figure(doc: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = doc["rows"]
    x = [r["ild_at_10"]["mean"] for r in rows]
    y = [r["ndcg_at_10"]["mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.errorbar(
        x,
        y,
        yerr=[
            [r["ndcg_at_10"]["mean"] - r["ndcg_at_10"]["lo"] for r in rows],
            [r["ndcg_at_10"]["hi"] - r["ndcg_at_10"]["mean"] for r in rows],
        ],
        fmt="o-",
        color="#2a6f97",
        capsize=3,
    )
    for r, xi, yi in zip(rows, x, y, strict=True):
        ax.annotate(f"λ={r['lambda']}", (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("ILD@10 (tag-Jaccard diversity), dev")
    ax.set_ylabel("NDCG@10, dev (95% CI)")
    ax.set_title(f"MMR over {doc['base']}: relevance vs diversity ({doc['n_personas']} dev personas)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Simulated users, LLM-judged relevance (judge not yet validated)", fontsize=6, color="#555")
    fig.tight_layout()
    fig.savefig(FIGURE, metadata={"Software": None})
    plt.close(fig)


def replay(check: bool) -> tuple[list[str], list[str]]:
    f = paths.RESULTS_DIR / OUT
    if not f.exists():
        return [], []
    from evalkit.evaluate import system_client

    client = system_client()
    client.reset_touched()
    doc = compute(judge=False)
    probs = []
    if check and json.loads(dumps(doc)) != json.loads(f.read_text(encoding="utf-8")):
        probs.append("MMR sweep differs")
    return client.touched_keys(), probs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)
    if args.estimate:
        print(json.dumps(estimate(), indent=1))
        return
    doc = compute()
    (paths.RESULTS_DIR / OUT).write_text(dumps(doc), encoding="utf-8")
    figure(doc)
    for r in doc["rows"]:
        print(f"lambda {r['lambda']}: NDCG {r['ndcg_at_10']['mean']:.3f} ILD {r['ild_at_10']['mean']:.3f}")


if __name__ == "__main__":
    main()
