"""Run systems, pool and judge holes, compute metrics, write ``eval/results/<system>.json``.

    cd backend && python -m evalkit.evaluate --system v0_rules     # make eval SYSTEM=v0_rules
    cd backend && python -m evalkit.evaluate --all                 # make eval-all (from stored runs)

A results file is byte-identical on rerun except for its ``volatile`` block (locally measured timings
and this process's LLM ledger). Run files: ``eval/runs/<system>.jsonl`` (deterministic) and
``eval/runs/<system>.timings.json`` (local timings).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from typing import Any, Optional

import numpy as np

from app.recsys import Catalog, build_system, get_catalog, get_system_spec, list_systems
from evalkit import metrics as M
from evalkit import paths
from evalkit.personas import Persona, file_sha256, load_personas, persona_stats
from evalkit.pool import POOL_DEPTH, POOL_ONLY, contributor_runs, pool_pairs
from evalkit.qrels import Qrels, fill_holes

RESULTS_SCHEMA = "evalkit.results/v1"
RUN_DEPTH = 50
K = 10
N_BOOT = 1000
BOOT_SEED = 20261122
SPLITS = ("test", "dev", "all")
NOTE = (
    "Simulated users (LLM-written intents over sampled latent profiles); relevance judged by an LLM; "
    "the judge is {validated}."
)


def _round(x: Any, nd: int = 6) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


def dumps(doc: dict[str, Any]) -> str:
    return json.dumps(_round(doc), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- runs
def run_system(
    name: str, catalog: Catalog, personas: list[Persona], depth: int = RUN_DEPTH, **config: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Returns (run, timings, system config). The system only ever sees ``persona.to_query()``."""
    system = build_system(name, catalog, **config)
    run, timings = {}, {}
    for p in personas:
        q = p.to_query()
        t0 = time.perf_counter()
        rec = system.recommend(q, depth)
        wall = (time.perf_counter() - t0) * 1000
        run[p.persona_id] = {
            "ids": [int(r) for r in rec.route_ids],
            "scores": [round(float(s), 6) for s in rec.scores],
            "trace": [t.summary() for t in rec.trace],
            "upstream_latency_ms": round(rec.upstream_latency_ms, 3),
            "upstream_cost_usd": round(rec.upstream_cost_usd, 9),
        }
        timings[p.persona_id] = {"local_ms": round(max(wall, rec.local_ms), 3)}
    return run, timings, system.config()


def save_run(name: str, run: dict[str, Any], timings: dict[str, Any]) -> None:
    paths.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (paths.RUNS_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for pid in sorted(run):
            fh.write(json.dumps({"persona_id": pid, **run[pid]}, sort_keys=True) + "\n")
    (paths.RUNS_DIR / f"{name}.timings.json").write_text(
        json.dumps(
            {
                "note": "locally measured, varies run to run",
                "machine": platform.machine(),
                "python": platform.python_version(),
                "per_persona": timings,
            },
            sort_keys=True,
            indent=0,
        )
        + "\n",
        encoding="utf-8",
    )


def load_run(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run = {}
    with (paths.RUNS_DIR / f"{name}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            run[d.pop("persona_id")] = d
    tp = paths.RUNS_DIR / f"{name}.timings.json"
    timings = json.loads(tp.read_text(encoding="utf-8"))["per_persona"] if tp.exists() else {}
    return run, timings


def stored_systems() -> list[str]:
    return sorted(p.stem for p in paths.RUNS_DIR.glob("*.jsonl") if p.stem in list_systems())


# --------------------------------------------------------------------------- pooling
def judge_pool(
    catalog: Catalog,
    personas: list[Persona],
    system_runs: dict[str, dict[str, Any]],
    qrels: Qrels,
    *,
    judge: bool = True,
) -> dict[str, Any]:
    """Pool = top-10 of the given systems plus the pool-only contributors; grade the holes."""
    runs = {n: {pid: r["ids"] for pid, r in run.items()} for n, run in system_runs.items()}
    runs.update(contributor_runs(catalog, personas, POOL_DEPTH))
    pairs = pool_pairs(runs, POOL_DEPTH)
    holes = qrels.holes(pairs)
    if not holes:
        return {"holes": 0}
    if not judge:
        return {"holes": len(holes), "judged": False}
    rep = fill_holes(qrels, holes, {p.persona_id: p for p in personas}, catalog)
    qrels.save()
    return rep


# --------------------------------------------------------------------------- metrics
def per_persona_metrics(catalog: Catalog, personas: list[Persona], run: dict[str, Any], qrels: Qrels) -> dict[str, Any]:
    qp = qrels.by_persona()
    tags = {}
    out = {}
    for p in personas:
        r = run[p.persona_id]
        ids = r["ids"]
        g = qp.get(p.persona_id, {})
        hc = p.constraints()
        judged = sum(1 for x in ids[:K] if x in g) / max(len(ids[:K]), 1)
        row: dict[str, Any] = {"judged_at_10": judged}
        if judged == 1.0:
            row["ndcg_at_10"] = M.ndcg_at_k(ids, g, K)
            row["mrr_at_10"] = M.mrr_at_k(ids, g, K)
        else:
            row["ndcg_at_10"] = row["mrr_at_10"] = None
        row["recall_at_50"] = M.recall_at_k(ids, g, 50)
        row["violation_at_10"] = M.violation_rate_at_k(ids, lambda x, hc=hc: bool(hc.violations(catalog.get(x))), K)
        for x in ids[:K]:
            if x not in tags:
                tags[x] = frozenset(catalog.get(x).tags)
        row["ild_at_10"] = M.intra_list_diversity([tags[x] for x in ids[:K]])
        row["cost_usd"] = r["upstream_cost_usd"]
        row["upstream_latency_ms"] = r["upstream_latency_ms"]
        out[p.persona_id] = row
    return out


METRIC_KEYS = ("ndcg_at_10", "mrr_at_10", "recall_at_50", "violation_at_10", "ild_at_10", "cost_usd")


def split_metrics(catalog: Catalog, personas: list[Persona], run: dict[str, Any], pp: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in METRIC_KEYS:
        out[key] = M.bootstrap_ci([pp[p.persona_id][key] for p in personas], n_boot=N_BOOT, seed=BOOT_SEED)
    # coverage@10 is a statistic of the whole set of lists, not a per-persona mean. A with-replacement
    # bootstrap would bias it downwards (duplicated personas add no new routes), so no CI is given.
    # It also grows with the number of personas, so compare systems only on the same split.
    lists = [run[p.persona_id]["ids"] for p in personas]
    out["coverage_at_10"] = {
        "mean": M.coverage(lists, len(catalog), K),
        "lo": None,
        "hi": None,
        "n": len(lists),
        "distinct_routes": len({r for rk in lists for r in rk[:K]}),
    }
    lat = [pp[p.persona_id]["upstream_latency_ms"] for p in personas]
    out["upstream_latency_ms"] = {"p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95)}
    out["judged_share_at_10"] = float(np.mean([pp[p.persona_id]["judged_at_10"] for p in personas]))
    out["ndcg_undefined"] = sum(1 for p in personas if pp[p.persona_id]["ndcg_at_10"] is None)
    out["n_personas"] = len(personas)
    return out


def owner_validation() -> dict[str, Any]:
    kp = paths.RESULTS_DIR / "_kappa.json"
    if kp.exists():
        k = json.loads(kp.read_text(encoding="utf-8"))
        if k.get("owner", {}).get("status") == "ok":
            return {"validated": True, "owner_qwk": k["owner"]["qwk"], "n": k["owner"]["n"]}
    return {"validated": False, "owner_qwk": None}


def compute_results(
    name: str,
    catalog: Catalog,
    personas: list[Persona],
    run: dict[str, Any],
    timings: dict[str, Any],
    qrels: Qrels,
    config: dict[str, Any],
) -> tuple[dict, dict]:
    from evalkit.intents import writer_config
    from evalkit.judge import RUBRIC_VERSION, production_judge

    pp = per_persona_metrics(catalog, personas, run, qrels)
    by_split = {s: [p for p in personas if s == "all" or p.split == s] for s in SPLITS}
    mets = {s: split_metrics(catalog, ps, run, pp) for s, ps in by_split.items()}
    jcfg = production_judge()
    st = qrels.stats(personas)
    val = owner_validation()
    doc = {
        "schema": RESULTS_SCHEMA,
        "system": name,
        "description": get_system_spec(name).description,
        "config": config,
        "note": NOTE.format(
            validated=(f"validated against the owner's labels (QWK {val['owner_qwk']:.2f})")
            if val["validated"]
            else "not yet validated against the owner's labels"
        ),
        "protocol": {
            "run_depth": RUN_DEPTH,
            "pool_depth": POOL_DEPTH,
            "k": K,
            "relevant_grade": M.RELEVANT_GRADE,
            "gain": "2^g - 1",
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "ci": 0.95},
            "pool_only_contributors": list(POOL_ONLY),
        },
        "data": {
            "catalogue_sha256": catalog.sha256,
            "catalogue_routes": len(catalog),
            "personas_sha256": file_sha256(paths.PERSONAS_FILE),
            "n_personas": len(personas),
            "split_sizes": {s: len(ps) for s, ps in by_split.items()},
            "qrels_sha256": qrels.sha256(),
            "qrels_pairs": st["pairs"],
            "qrels_by_source": st["by_source"],
        },
        "judge": {"config": jcfg.describe(), "models": st["models"], "rubric": RUBRIC_VERSION, "owner_validation": val},
        "models": {"intent_writer": writer_config(), "system": config.get("models", [])},
        "metrics": mets,
        "per_persona": pp,
    }
    loc = {
        s: [timings.get(p.persona_id, {}).get("local_ms", 0.0) + pp[p.persona_id]["upstream_latency_ms"] for p in ps]
        for s, ps in by_split.items()
    }
    volatile = {
        "note": "locally measured or run-specific; excluded from the determinism check",
        "latency_ms_nominal": {
            s: {
                "p50": M.percentile(v, 50),
                "p95": M.percentile(v, 95),
                "definition": "local compute + recorded upstream latency",
            }
            for s, v in loc.items()
        },
        "machine": platform.machine(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return doc, volatile


def write_results(name: str, doc: dict, volatile: dict, ledger: Optional[dict] = None) -> None:
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full = {**doc, "volatile": {**volatile, "llm_ledger": ledger}}
    (paths.RESULTS_DIR / f"{name}.json").write_text(dumps(full), encoding="utf-8")


def write_meta(catalog: Catalog, personas: list[Persona], qrels: Qrels) -> None:
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS_DIR / "_qrels.json").write_text(
        dumps({"schema": "evalkit.qrels_stats/v1", **qrels.stats(personas), "qrels_sha256": qrels.sha256()}),
        encoding="utf-8",
    )
    ps = persona_stats(personas)
    ps["personas_sha256"] = file_sha256(paths.PERSONAS_FILE)
    (paths.RESULTS_DIR / "_personas.json").write_text(
        dumps({"schema": "evalkit.persona_stats/v1", **ps}), encoding="utf-8"
    )


def _ledger() -> Optional[dict]:
    from evalkit.judge import _judge_clients

    reps = {n: c.usage_report() for n, c in _judge_clients.items()}
    return reps or None


# --------------------------------------------------------------------------- commands
def evaluate(
    name: str, *, judge: bool = True, catalog: Optional[Catalog] = None, personas: Optional[list[Persona]] = None
) -> dict:
    catalog = catalog or get_catalog()
    personas = personas or load_personas()
    run, timings, config = run_system(name, catalog, personas)
    save_run(name, run, timings)
    qrels = Qrels.load()
    rep = judge_pool(catalog, personas, {name: run}, qrels, judge=judge)
    print(f"[eval] {name}: pool holes {rep}", file=sys.stderr)
    doc, vol = compute_results(name, catalog, personas, run, timings, qrels, config)
    write_results(name, doc, vol, _ledger())
    write_meta(catalog, personas, qrels)
    return doc


def evaluate_all(
    *, judge: bool = True, catalog: Optional[Catalog] = None, personas: Optional[list[Persona]] = None
) -> list[str]:
    """Recompute every stored run's results against the current qrels (judging holes of all runs first)."""
    catalog = catalog or get_catalog()
    personas = personas or load_personas()
    names = stored_systems()
    loaded = {n: load_run(n) for n in names}
    qrels = Qrels.load()
    rep = judge_pool(catalog, personas, {n: r for n, (r, _) in loaded.items()}, qrels, judge=judge)
    print(f"[eval-all] pool holes {rep}", file=sys.stderr)
    for n in names:
        run, timings = loaded[n]
        config = build_system(n, catalog).config()
        doc, vol = compute_results(n, catalog, personas, run, timings, qrels, config)
        write_results(n, doc, vol, _ledger())
    write_meta(catalog, personas, qrels)
    return names


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--system")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--no-judge", action="store_true", help="fail instead of judging holes")
    args = ap.parse_args(argv)
    if args.all:
        names = evaluate_all(judge=not args.no_judge)
        print("recomputed:", ", ".join(names))
        return
    doc = evaluate(args.system, judge=not args.no_judge)
    m = doc["metrics"]["test"]
    print(
        f"{args.system} test: NDCG@10 {m['ndcg_at_10']['mean']:.3f} [{m['ndcg_at_10']['lo']:.3f}, "
        f"{m['ndcg_at_10']['hi']:.3f}]  violations@10 {m['violation_at_10']['mean']:.3f}  "
        f"judged@10 {m['judged_share_at_10']:.3f}"
    )


if __name__ == "__main__":
    main()
