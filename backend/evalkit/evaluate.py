"""Run systems, pool and judge holes, compute metrics, write ``eval/results/<system>.json``.

    cd backend && python -m evalkit.evaluate --system v0_rules     # make eval SYSTEM=v0_rules
    cd backend && python -m evalkit.evaluate --system recsys_serve --crossfit   # make eval ... CROSSFIT=1
    cd backend && python -m evalkit.evaluate --system recsys_serve --holdout    # make eval ... HOLDOUT=1
    cd backend && python -m evalkit.evaluate --all                 # make eval-all (from stored runs)

A results file is byte-identical on rerun except for its ``volatile`` block (locally measured timings
and this process's LLM ledger). Run files: ``eval/runs/<system>.jsonl`` (deterministic) and
``eval/runs/<system>.timings.json`` (local timings).

Three benches (:class:`Bench`): ``main`` (the 300 personas, test / dev / all), ``crossfit`` (the dev personas,
each ranked by a copy of the system's learned ranker that never saw its labels; ``evalkit.crossfit``,
docs/EVAL.md section 13) and ``holdout`` (150 personas generated for the iteration-3 decision, with their own
qrels file; docs/DECISIONS.md D28). Each has its own runs and results directory.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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


# --------------------------------------------------------------------------- benches
@dataclass(frozen=True)
class Bench:
    """A set of personas with its own runs and results (paths resolved lazily, so tests can redirect them).

    - ``main``: ``eval/personas/personas.jsonl`` (dev 200 / test 100), ``eval/qrels/qrels.jsonl``.
    - ``crossfit``: the main dev personas, each ranked by a learned-ranker copy that never saw its labels
      (``evalkit.crossfit``); judgements go to the main qrels (dev pairs only).
    - ``holdout``: ``eval/personas/holdout.jsonl`` and its own ``eval/qrels/holdout_qrels.jsonl``; used only for
      the iteration-3 decision (D28) and afterwards for reporting.
    """

    name: str

    @property
    def splits(self) -> tuple[str, ...]:
        return {"main": SPLITS, "crossfit": ("dev",), "holdout": ("holdout",)}[self.name]

    @property
    def personas_file(self) -> Path:
        return paths.PERSONA_DIR / "holdout.jsonl" if self.name == "holdout" else paths.PERSONAS_FILE

    @property
    def qrels_file(self) -> Path:
        return paths.QRELS_DIR / "holdout_qrels.jsonl" if self.name == "holdout" else paths.QRELS_FILE

    @property
    def runs_dir(self) -> Path:
        return paths.RUNS_DIR if self.name == "main" else paths.RUNS_DIR / self.name

    @property
    def results_dir(self) -> Path:
        return paths.RESULTS_DIR if self.name == "main" else paths.RESULTS_DIR / self.name

    def personas(self) -> list[Persona]:
        if self.name == "holdout":
            return load_personas(self.personas_file)
        ps = load_personas()
        return [p for p in ps if p.split == "dev"] if self.name == "crossfit" else ps

    def qrels(self) -> Qrels:
        return Qrels.load(self.qrels_file)


MAIN, CROSSFIT, HOLDOUT = Bench("main"), Bench("crossfit"), Bench("holdout")


# --------------------------------------------------------------------------- runs
_system_client = None


def system_client():
    """LLM client for the systems' own upstream calls (query embeddings, ranker API): embeddings on Vertex,
    ranking on OpenRouter, or everything on the mock provider when LLM_PROVIDER=mock (tests, CI)."""
    global _system_client
    if _system_client is None:
        from app.llm import LLMConfig
        from evalkit.llm import make_client

        if LLMConfig.from_settings().llm_provider == "mock":
            _system_client = make_client("systems")
        else:
            _system_client = make_client("systems", embed_provider="vertex", rank_provider="openrouter")
    return _system_client


_chat_client = None


def chat_client():
    """The chat extractor's client for systems that run the chat (``uses_chat``): pinned to the chosen chat
    model (D18: Claude Haiku 5.5 via OpenRouter, reasoning off) whatever credentials are configured, so a
    replay without credentials asks the frozen cache for the same calls; mock under LLM_PROVIDER=mock."""
    global _chat_client
    if _chat_client is None:
        from app.chat.llm import CHAT_DEFAULTS, build_chat_client
        from app.llm import LLMConfig
        from evalkit import llm as L

        cfg = LLMConfig.from_settings()
        cm = CHAT_DEFAULTS["mock" if cfg.llm_provider == "mock" else "openrouter"]
        frozen = cfg.frozen_dirs
        import os

        if "LLM_CACHE_FROZEN_DIRS" not in os.environ and paths.FROZEN_DIR.exists() and paths.FROZEN_DIR not in frozen:
            frozen = (*frozen, paths.FROZEN_DIR)
        _chat_client = build_chat_client(cm, config=cfg.with_(frozen_dirs=frozen), budget_usd=1.0)
        L._clients.append(("chat", _chat_client))
    return _chat_client


def make_system(name: str, catalog: Catalog, **config: Any):
    spec = get_system_spec(name)
    if spec.uses_llm and "llm" not in config:
        config["llm"] = system_client()
    if spec.uses_chat and "chat_llm" not in config:
        config["chat_llm"] = chat_client()
    return build_system(name, catalog, **config)


def run_system(
    name: str, catalog: Catalog, personas: list[Persona], depth: int = RUN_DEPTH, *, concurrency: int = 8, **config: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Returns (run, timings, system config). The system only ever sees ``persona.to_query()``.

    Systems with an async entry point (``arecommend``) run ``concurrency`` personas at a time; their local
    time is the sum of the stages' own compute (``Recommendation.local_ms``), which excludes waiting for
    upstream calls (their recorded latency is added separately as nominal upstream latency)."""
    system = make_system(name, catalog, **config)
    run, timings = {}, {}

    def record(p: Persona, rec, local_ms: float) -> None:
        run[p.persona_id] = {
            "ids": [int(r) for r in rec.route_ids],
            "scores": [round(float(s), 6) for s in rec.scores],
            "trace": [t.summary() for t in rec.trace],
            "upstream_latency_ms": round(rec.upstream_latency_ms, 3),
            "upstream_cost_usd": round(rec.upstream_cost_usd, 9),
        }
        timings[p.persona_id] = {"local_ms": round(local_ms, 3)}

    if hasattr(system, "arecommend"):
        import asyncio

        from app.llm import run_sync

        async def go():
            sem = asyncio.Semaphore(concurrency)

            async def one(p: Persona):
                async with sem:
                    return p, await system.arecommend(p.to_query(), depth)

            return await asyncio.gather(*(one(p) for p in personas))

        for p, rec in run_sync(go()):
            record(p, rec, rec.local_ms)
    else:
        for p in personas:
            q = p.to_query()
            t0 = time.perf_counter()
            rec = system.recommend(q, depth)
            wall = (time.perf_counter() - t0) * 1000
            record(p, rec, max(wall, rec.local_ms))
    return run, timings, system.config()


def save_run(name: str, run: dict[str, Any], timings: dict[str, Any], bench: Bench = MAIN) -> None:
    bench.runs_dir.mkdir(parents=True, exist_ok=True)
    with (bench.runs_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for pid in sorted(run):
            fh.write(json.dumps({"persona_id": pid, **run[pid]}, sort_keys=True) + "\n")
    (bench.runs_dir / f"{name}.timings.json").write_text(
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


def load_run(name: str, bench: Bench = MAIN) -> tuple[dict[str, Any], dict[str, Any]]:
    run = {}
    with (bench.runs_dir / f"{name}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            run[d.pop("persona_id")] = d
    tp = bench.runs_dir / f"{name}.timings.json"
    timings = json.loads(tp.read_text(encoding="utf-8"))["per_persona"] if tp.exists() else {}
    return run, timings


def stored_systems(bench: Bench = MAIN) -> list[str]:
    return sorted(p.stem for p in bench.runs_dir.glob("*.jsonl") if p.stem in list_systems())


def bench_run(
    name: str, catalog: Catalog, personas: list[Persona], bench: Bench = MAIN
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """``run_system`` on a bench: the cross-fitted bench ranks each dev persona with the fold copy of the
    system's learned ranker that was trained without it (``evalkit.crossfit``)."""
    if bench.name == "crossfit":
        from evalkit import crossfit

        return crossfit.run(name, catalog, personas)
    return run_system(name, catalog, personas)


def bench_config(name: str, catalog: Catalog, bench: Bench = MAIN) -> dict[str, Any]:
    if bench.name == "crossfit":
        from evalkit import crossfit

        return crossfit.config(name, catalog)
    return make_system(name, catalog).config()


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


BENCH_LABEL = {
    "main": None,
    "crossfit": "dev, cross-fitted ranker: each dev persona ranked by a copy of the learned ranker trained without "
    "it (docs/EVAL.md section 13); not in-sample",
    "holdout": "holdout: 150 personas generated after the iteration-3 pre-registration (docs/DECISIONS.md D28), "
    "used for that decision and for reporting only",
}


def compute_results(
    name: str,
    catalog: Catalog,
    personas: list[Persona],
    run: dict[str, Any],
    timings: dict[str, Any],
    qrels: Qrels,
    config: dict[str, Any],
    bench: Bench = MAIN,
) -> tuple[dict, dict]:
    from evalkit.intents import writer_config
    from evalkit.judge import RUBRIC_VERSION, production_judge

    pp = per_persona_metrics(catalog, personas, run, qrels)
    by_split = {s: [p for p in personas if s == "all" or p.split == s] for s in bench.splits}
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
            "personas_sha256": file_sha256(bench.personas_file),
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
    if bench.name != "main":
        doc["bench"] = {"name": bench.name, "label": BENCH_LABEL[bench.name], "qrels_file": bench.qrels_file.name}
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


def write_results(name: str, doc: dict, volatile: dict, ledger: Optional[dict] = None, bench: Bench = MAIN) -> None:
    bench.results_dir.mkdir(parents=True, exist_ok=True)
    full = {**doc, "volatile": {**volatile, "llm_ledger": ledger}}
    (bench.results_dir / f"{name}.json").write_text(dumps(full), encoding="utf-8")


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
    write_reference(catalog, personas, qrels)


def write_reference(catalog: Catalog, personas: list[Persona], qrels: Qrels) -> None:
    """Scores of the pool-only latent matcher. NOT a system: it reads the simulated user's latent profile, so
    it indicates headroom only and never appears in the ablation (``_reference.json``)."""
    from evalkit.pool import pool_latent

    run = {
        p.persona_id: {"ids": pool_latent(catalog, p, RUN_DEPTH), "upstream_cost_usd": 0.0, "upstream_latency_ms": 0.0}
        for p in personas
    }
    pp = per_persona_metrics(catalog, personas, run, qrels)
    by_split = {s: [p for p in personas if s == "all" or p.split == s] for s in SPLITS}
    doc = {
        "schema": "evalkit.reference/v1",
        "name": "pool_latent",
        "note": "not a system: reads the simulated user's latent profile (pool-only contributor)",
        "qrels_sha256": qrels.sha256(),
        "metrics": {s: split_metrics(catalog, ps, run, pp) for s, ps in by_split.items()},
        "per_persona": pp,
    }
    (paths.RESULTS_DIR / "_reference.json").write_text(dumps(doc), encoding="utf-8")


def _ledger() -> Optional[dict]:
    from evalkit.judge import _judge_clients

    reps = {n: c.usage_report() for n, c in _judge_clients.items()}
    if _system_client is not None:
        reps["systems"] = _system_client.usage_report()
    if _chat_client is not None:
        reps["chat"] = _chat_client.usage_report()
    return reps or None


# --------------------------------------------------------------------------- commands
def evaluate(
    name: str,
    *,
    judge: bool = True,
    catalog: Optional[Catalog] = None,
    personas: Optional[list[Persona]] = None,
    bench: Bench = MAIN,
) -> dict:
    catalog = catalog or get_catalog()
    personas = personas or bench.personas()
    run, timings, config = bench_run(name, catalog, personas, bench)
    save_run(name, run, timings, bench)
    qrels = bench.qrels()
    rep = judge_pool(catalog, personas, {name: run}, qrels, judge=judge)
    print(f"[eval:{bench.name}] {name}: pool holes {rep}", file=sys.stderr)
    doc, vol = compute_results(name, catalog, personas, run, timings, qrels, config, bench)
    write_results(name, doc, vol, _ledger(), bench)
    if bench.name == "main":
        write_meta(catalog, personas, qrels)
    elif bench.name == "holdout":
        from evalkit import holdout

        holdout.write_decision()
    return doc


# real cost per LLM-judged pair of the production judge as measured on the v1_filters holes (2.93 USD / 2206)
JUDGE_COST_PER_PAIR_USD = 0.00133


def estimate_holes(
    names: list[str],
    *,
    catalog: Optional[Catalog] = None,
    personas: Optional[list[Persona]] = None,
    bench: Bench = MAIN,
) -> dict[str, Any]:
    """Pool holes the given systems would add (rule-graded vs LLM-judged) and the judge's nominal cost.
    Runs the systems (upstream calls go through the cache) but writes nothing."""
    from evalkit.judge import rule_grade

    catalog = catalog or get_catalog()
    personas = personas or bench.personas()
    pm = {p.persona_id: p for p in personas}
    runs = {}
    for n in names:
        if (bench.runs_dir / f"{n}.jsonl").exists() and n not in names[-1:]:
            runs[n] = load_run(n, bench)[0]
        else:
            runs[n] = bench_run(n, catalog, personas, bench)[0]
    q = bench.qrels()
    pruns = {n: {pid: r["ids"] for pid, r in run.items()} for n, run in runs.items()}
    pruns.update(contributor_runs(catalog, personas, POOL_DEPTH))  # judge_pool pools them too
    pool = pool_pairs(pruns, POOL_DEPTH)
    holes = q.holes(pool)
    llm = [h for h in holes if rule_grade(pm[h[0]], catalog, h[1]) is None]
    by_split = {s: sum(1 for h in llm if pm[h[0]].split == s) for s in ("dev", "test", "holdout")}
    return {
        "bench": bench.name,
        "systems": names,
        "holes": len(holes),
        "rule_graded": len(holes) - len(llm),
        "llm_pairs": len(llm),
        "llm_pairs_by_split": by_split,
        "est_judge_usd": round(len(llm) * JUDGE_COST_PER_PAIR_USD, 2),
    }


def evaluate_all(
    *, judge: bool = True, catalog: Optional[Catalog] = None, personas: Optional[list[Persona]] = None
) -> list[str]:
    """Recompute every stored run's results against the current qrels (judging holes of all runs first): the
    main runs and the cross-fitted dev runs share the main qrels and are pooled together; the holdout runs are
    pooled and rescored against the holdout qrels, and the holdout decision file is rewritten."""
    catalog = catalog or get_catalog()
    personas = personas or load_personas()
    names = stored_systems()
    loaded = {n: load_run(n) for n in names}
    cf_names = stored_systems(CROSSFIT)
    cf_loaded = {n: load_run(n, CROSSFIT) for n in cf_names}
    qrels = Qrels.load()
    pool_runs = {n: r for n, (r, _) in loaded.items()}
    pool_runs.update({f"{n}@crossfit": r for n, (r, _) in cf_loaded.items()})
    rep = judge_pool(catalog, personas, pool_runs, qrels, judge=judge)
    print(f"[eval-all] pool holes {rep}", file=sys.stderr)
    for n in names:
        run, timings = loaded[n]
        config = make_system(n, catalog).config()
        doc, vol = compute_results(n, catalog, personas, run, timings, qrels, config)
        write_results(n, doc, vol, _ledger())
    write_meta(catalog, personas, qrels)
    dev = [p for p in personas if p.split == "dev"]
    for n in cf_names:
        run, timings = cf_loaded[n]
        doc, vol = compute_results(n, catalog, dev, run, timings, qrels, bench_config(n, catalog, CROSSFIT), CROSSFIT)
        write_results(n, doc, vol, _ledger(), CROSSFIT)
    ho_names = stored_systems(HOLDOUT)
    if ho_names:
        hp = HOLDOUT.personas()
        ho_loaded = {n: load_run(n, HOLDOUT) for n in ho_names}
        hq = HOLDOUT.qrels()
        rep = judge_pool(catalog, hp, {n: r for n, (r, _) in ho_loaded.items()}, hq, judge=judge)
        print(f"[eval-all] holdout pool holes {rep}", file=sys.stderr)
        for n in ho_names:
            run, timings = ho_loaded[n]
            doc, vol = compute_results(n, catalog, hp, run, timings, hq, make_system(n, catalog).config(), HOLDOUT)
            write_results(n, doc, vol, _ledger(), HOLDOUT)
        from evalkit import holdout

        holdout.write_decision()
    return names


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--system")
    g.add_argument("--all", action="store_true")
    g.add_argument("--estimate", nargs="+", metavar="SYSTEM", help="count pool holes and judge cost; writes nothing")
    b = ap.add_mutually_exclusive_group()
    b.add_argument(
        "--crossfit", action="store_true", help="dev personas, each ranked by a ranker copy trained without it"
    )
    b.add_argument("--holdout", action="store_true", help="the holdout personas (docs/DECISIONS.md D28)")
    ap.add_argument("--no-judge", action="store_true", help="fail instead of judging holes")
    args = ap.parse_args(argv)
    bench = CROSSFIT if args.crossfit else HOLDOUT if args.holdout else MAIN
    if args.estimate:
        print(json.dumps(estimate_holes(args.estimate, bench=bench), indent=1))
        return
    if args.all:
        names = evaluate_all(judge=not args.no_judge)
        print("recomputed:", ", ".join(names))
        return
    doc = evaluate(args.system, judge=not args.no_judge, bench=bench)
    split = bench.splits[0]
    m = doc["metrics"][split]
    print(
        f"{args.system} {bench.name}/{split}: NDCG@10 {m['ndcg_at_10']['mean']:.3f} [{m['ndcg_at_10']['lo']:.3f}, "
        f"{m['ndcg_at_10']['hi']:.3f}]  violations@10 {m['violation_at_10']['mean']:.3f}  "
        f"judged@10 {m['judged_share_at_10']:.3f}"
    )


if __name__ == "__main__":
    main()
