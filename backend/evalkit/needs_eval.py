"""Iteration 2: the stated-need stage measured on the development split only (``eval/results/_needs_dev.json``).

    cd backend && python -m evalkit.needs_eval extractor     # extractor precision / recall on dev (no LLM call)
    cd backend && python -m evalkit.needs_eval dev           # variants vs recsys_serve on dev (judges dev holes)

- ``extractor``: :func:`app.recsys.needs.extract_needs` on every dev intent, scored against the simulated
  user's latent profile where a latent field corresponds to the need (stroller party, train travel, MTB
  taste, loop / one-way shape, multi-day trips, the comfortable distance band). The latent profile is used
  here only to *measure* the extractor offline; no system ever reads it.
- ``dev``: each variant and ``recsys_serve`` on the 200 dev personas, holes of the dev pool judged (dev
  personas only: no test persona is run or judged), paired bootstrap of the per-persona differences, the
  pre-stated acceptance rule evaluated (docs/DECISIONS.md D27). Test is run once, afterwards, by
  ``make eval SYSTEM=<variant>`` (both splits); this file never contains a test number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

import numpy as np

from app.recsys import get_catalog
from app.recsys.needs import NEED_CODES, extract_needs
from evalkit import paths
from evalkit.evaluate import (
    BOOT_SEED,
    N_BOOT,
    dumps,
    judge_pool,
    load_run,
    per_persona_metrics,
    run_system,
)
from evalkit.metrics import bootstrap_ci, paired_bootstrap
from evalkit.personas import Persona, load_personas
from evalkit.qrels import Qrels

OUT = "_needs_dev.json"
SCHEMA = "evalkit.needs_dev/v1"
BASELINE = "recsys_serve"
VARIANTS = ("recsys_serve_needs", "recsys_serve_needs_promote", "recsys_serve_needs_demote")
# The acceptance rule, fixed before any dev result of a variant was computed (docs/DECISIONS.md D27).
RULE = {
    "metric": "ndcg_at_10",
    "split": "dev",
    "baseline": BASELINE,
    "accept_if": [
        "paired bootstrap 95% CI of mean NDCG@10(variant) - NDCG@10(recsys_serve) on dev has lower bound > 0",
        "mean Violations@10 on dev not above recsys_serve's",
        "the stage's own compute adds < 20 ms at p95 (local)",
    ],
    "choose": "among accepted variants, the larger mean dev NDCG@10 difference; none accepted -> stage off",
    "extractor_gate": "a need with a latent counterpart is enabled only with dev precision >= 0.8",
}


def _latent_truth(p: Persona) -> dict[str, Any]:
    L = p.latent
    return {
        "stroller": bool(L["party"].get("stroller")),
        "no_steps": bool(L["party"].get("stroller")),  # closest counterpart: a stroller party cannot take steps
        "train": L["transport"] == "train",
        "public_transport": L["transport"] == "train",
        "mtb": L["style"] == "mtb",
        "loop": L["shape"] == "loop",
        "one_way": L["shape"] == "linear",
        "days": bool(L.get("multi_day")),
    }


def _band_share(lo: float, hi: float, wlo, whi) -> float:
    """Share of the latent comfortable band [lo, hi] inside the stated window."""
    a = max(lo, wlo if wlo is not None else -np.inf)
    b = min(hi, whi if whi is not None else np.inf)
    return max(0.0, b - a) / max(hi - lo, 1e-9)


def extractor_report(personas: list[Persona]) -> dict[str, Any]:
    dev = [p for p in personas if p.split == "dev"]
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    fired: Counter = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    days_exact = [0, 0]
    length_shares: dict[str, list[float]] = {"length": [], "max_length": []}
    for p in dev:
        sn = extract_needs(p.intent, p.latent["category"])
        codes = set(sn.codes())
        truth = _latent_truth(p)
        for c in codes:
            fired[c] += 1
            n = next(n for n in sn.needs if n.code == c)
            if c in truth:
                if truth[c]:
                    tp[c] += 1
                else:
                    fp[c] += 1
                    examples.setdefault(c, []).append({"persona": p.persona_id, "phrase": n.phrase, "intent": p.intent})
            if c == "days" and truth["days"]:
                days_exact[0] += 1
                days_exact[1] += int(n.p("days") == p.latent.get("days"))
            if c in length_shares:
                lo, hi = p.latent["distance_band_km"]
                length_shares[c].append(_band_share(lo, hi, n.p("lo"), n.p("hi")))
        for c, v in truth.items():
            if c == "mtb" and p.latent["category"] != "cycling":
                continue
            if v and c not in codes:
                fn[c] += 1
    per_need = {}
    for c in NEED_CODES:
        row: dict[str, Any] = {"fired": fired[c]}
        if c in ("stroller", "no_steps", "train", "public_transport", "mtb", "loop", "one_way", "days"):
            row.update(
                tp=tp[c],
                fp=fp[c],
                fn=fn[c],
                precision=(tp[c] / (tp[c] + fp[c])) if tp[c] + fp[c] else None,
                recall=(tp[c] / (tp[c] + fn[c])) if tp[c] + fn[c] else None,
                false_positives=examples.get(c, [])[:5],
            )
        if c in length_shares and length_shares[c]:
            s = np.array(length_shares[c])
            row.update(band_share_mean=float(s.mean()), band_mostly_inside=float((s >= 0.5).mean()))
        per_need[c] = row
    per_need["days"]["days_exact"] = days_exact[1] / days_exact[0] if days_exact[0] else None
    with_need = sum(1 for p in dev if extract_needs(p.intent, p.latent["category"]))
    return {
        "n_dev": len(dev),
        "dev_personas_with_a_need": with_need,
        "per_need": per_need,
        "truth": {
            "stroller / no_steps": "party pushes a stroller",
            "train / public_transport": "latent transport is train (the judge treats it as a hard wish)",
            "mtb": "latent style is mountain biking (cycling personas only)",
            "loop / one_way": "latent shape loop / linear",
            "days": "latent multi-day trip; days_exact = share with the right day count",
            "length / max_length": "share of the latent comfortable band inside the stated window",
        },
    }


def needs_met_in_top10(run: dict[str, Any], personas: list[Persona]) -> dict[str, Any]:
    """For each need: over the dev personas who stated it, the mean share of the top 10 that meets it strictly
    and at any level (catalogue facts only; no judge). How often a system ignores a stated, verifiable need."""
    from app.recsys.needs import first_level, need_activity

    catalog = get_catalog()
    acc: dict[str, list[tuple[float, float]]] = {}
    for p in personas:
        if p.split != "dev":
            continue
        sn = extract_needs(p.intent, p.latent["category"])
        act = need_activity(p.latent["category"], sn)
        ids = run[p.persona_id]["ids"][:10]
        for n in sn.needs:
            lv = [first_level(catalog, catalog.index_of(r), n, act) for r in ids]
            acc.setdefault(n.code, []).append(
                (sum(x == 0 for x in lv) / max(len(ids), 1), sum(x is not None for x in lv) / max(len(ids), 1))
            )
    return {
        c: {
            "personas": len(v),
            "strict": float(np.mean([a for a, _ in v])),
            "any_level": float(np.mean([b for _, b in v])),
        }
        for c, v in sorted(acc.items())
    }


# --------------------------------------------------------------------------------------- dev comparison
def _compare(runs: dict[str, dict[str, Any]], baseline: str, dev: list[Persona], qrels: Qrels) -> dict[str, Any]:
    """Metrics of every run on ``dev`` and paired differences of each against ``baseline`` (overall, where a need
    was stated, per need), the largest per-persona gains and losses, and how often the top 10 meets the needs."""
    catalog = get_catalog()
    pp = {n: per_persona_metrics(catalog, dev, run, qrels) for n, run in runs.items()}
    pids = [p.persona_id for p in dev]
    stated = {p.persona_id: extract_needs(p.intent, p.latent["category"]).codes() for p in dev}
    out: dict[str, Any] = {"systems": {}, "paired_vs_baseline": {}}

    def pb(a: str, b: str, key: str, among: list[str]) -> dict[str, Any]:
        return paired_bootstrap(
            [pp[a][pid][key] for pid in among], [pp[b][pid][key] for pid in among], n_boot=N_BOOT, seed=BOOT_SEED
        )

    for n in runs:
        m = {
            k: bootstrap_ci([pp[n][pid][k] for pid in pids], n_boot=N_BOOT, seed=BOOT_SEED)
            for k in ("ndcg_at_10", "mrr_at_10", "recall_at_50", "violation_at_10", "ild_at_10")
        }
        m["judged_share_at_10"] = float(np.mean([pp[n][pid]["judged_at_10"] for pid in pids]))
        out["systems"][n] = m
    for v in runs:
        if v == baseline:
            continue
        d = {k: pb(baseline, v, k, pids) for k in ("ndcg_at_10", "mrr_at_10", "recall_at_50", "violation_at_10")}
        d["ndcg_at_10_personas_with_needs"] = pb(baseline, v, "ndcg_at_10", [pid for pid in pids if stated[pid]])
        d["ndcg_at_10_by_need"] = {
            c: pb(baseline, v, "ndcg_at_10", [pid for pid in pids if c in stated[pid]])
            for c in sorted({c for cs in stated.values() for c in cs})
        }
        diffs = [
            (pp[v][pid]["ndcg_at_10"] - pp[baseline][pid]["ndcg_at_10"], pid)
            for pid in pids
            if pp[v][pid]["ndcg_at_10"] is not None and pp[baseline][pid]["ndcg_at_10"] is not None
        ]
        d["better_by_0_1"] = sum(1 for x, _ in diffs if x > 0.1)
        d["worse_by_0_1"] = sum(1 for x, _ in diffs if x < -0.1)
        d["unchanged"] = sum(1 for x, _ in diffs if abs(x) < 1e-9)
        d["largest_losses"] = [{"persona": pid, "diff": x, "needs": stated[pid]} for x, pid in sorted(diffs)[:8]]
        d["largest_gains"] = [
            {"persona": pid, "diff": x, "needs": stated[pid]} for x, pid in sorted(diffs, reverse=True)[:8]
        ]
        d["personas_where_stage_ran"] = sum(
            1 for pid in pids if any(s["stage"] == "needs" for s in runs[v][pid]["trace"])
        )
        d["top10_meets_needs"] = needs_met_in_top10(runs[v], dev)
        ndcg = d["ndcg_at_10"]
        viol_ok = out["systems"][v]["violation_at_10"]["mean"] <= out["systems"][baseline]["violation_at_10"]["mean"]
        d["rule"] = {"ndcg_ci_lower_above_0": bool(ndcg["lo"] is not None and ndcg["lo"] > 0), "violations_ok": viol_ok}
        out["paired_vs_baseline"][v] = d
    out["per_persona_ndcg_at_10"] = {n: {pid: pp[n][pid]["ndcg_at_10"] for pid in pids} for n in runs}
    return out


def dev_comparison(variants: tuple[str, ...] = VARIANTS, *, judge: bool = True) -> dict[str, Any]:
    """The pre-registered comparison: each variant vs the stored ``recsys_serve`` run, dev personas only."""
    catalog = get_catalog()
    dev = [p for p in load_personas() if p.split == "dev"]
    base_run, _ = load_run(BASELINE)
    runs = {BASELINE: {p.persona_id: base_run[p.persona_id] for p in dev}}
    for v in variants:
        runs[v] = run_system(v, catalog, dev)[0]
    qrels = Qrels.load()
    rep = judge_pool(catalog, dev, {v: runs[v] for v in variants}, qrels, judge=judge)
    print(f"[needs dev] pool holes {rep}", file=sys.stderr)
    return {"holes": rep, **_compare(runs, BASELINE, dev, qrels)}


# --------------------------------------------------------------------------------------- cross-fitted check
CF_SUFFIX = "@crossfit"


def crossfit_models() -> tuple[list[tuple[set[str], Any]], dict[str, Any]]:
    """Five copies of the served ranker (no ranker-API feature, no ``length_cap_ratio``), each trained like
    ``v1_noapi_nocap`` (same pinned dev labels, parameters and number of rounds) on four fifths of the dev
    personas: the persona folds of the ranker's own grouped CV. Since iteration 3 this is the harness feature
    ``evalkit.crossfit`` (``make eval SYSTEM=... CROSSFIT=1``); this post-hoc check uses the same fold models."""
    from evalkit import crossfit

    models = crossfit.fold_models("v1_noapi_nocap")
    meta = {
        "folds": len(models),
        "rounds": models[0][1].meta["rounds"],
        "fold_sizes": [len(f) for f, _ in models],
        "lightgbm": models[0][1].meta["lightgbm"],
    }
    return models, meta


def crossfit_comparison(variants: tuple[str, ...] = VARIANTS, *, judge: bool = True) -> dict[str, Any]:
    """Post-hoc (not part of the pre-registered rule): ``recsys_serve`` and each variant on dev with the
    cross-fitted ranker, holes of these lists judged (dev only)."""
    catalog = get_catalog()
    dev = [p for p in load_personas() if p.split == "dev"]
    models, meta = crossfit_models()

    def fold_of(pid: str) -> int:
        return next((k for k, (f, _) in enumerate(models) if pid in f), 0)

    runs: dict[str, dict[str, Any]] = {}
    for name in (BASELINE, *variants):
        run: dict[str, Any] = {}
        for k, (_, m) in enumerate(models):
            ps = [p for p in dev if fold_of(p.persona_id) == k]
            run.update(run_system(name, catalog, ps, ltr_model=m)[0])
        runs[name + CF_SUFFIX] = run
    qrels = Qrels.load()
    rep = judge_pool(catalog, dev, runs, qrels, judge=judge)
    print(f"[needs crossfit] pool holes {rep}", file=sys.stderr)
    if not judge and rep.get("holes"):
        return {"holes": rep, "models": meta}
    return {"holes": rep, "models": meta, **_compare(runs, BASELINE + CF_SUFFIX, dev, qrels)}


def _stage_ms(variants: tuple[str, ...]) -> dict[str, Any]:
    """p50 / p95 of the needs stage's own local compute over the dev personas (a fresh in-process run)."""
    from app.llm import run_sync
    from evalkit.evaluate import make_system

    catalog = get_catalog()
    dev = [p for p in load_personas() if p.split == "dev"]
    res = {}
    for v in variants:
        pipe = make_system(v, catalog)
        ms = []
        for p in dev:
            rec = run_sync(pipe.arecommend(p.to_query(), 50))
            ms += [t.local_ms for t in rec.trace if t.stage == "needs"]
        res[v] = {
            "n": len(ms),
            "p50": float(np.percentile(ms, 50)) if ms else None,
            "p95": float(np.percentile(ms, 95)) if ms else None,
        }
    return res


def _load() -> dict[str, Any]:
    p = paths.RESULTS_DIR / OUT
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"schema": SCHEMA, "rule": RULE}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=("extractor", "dev", "crossfit", "latency"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args(argv)
    doc = _load()
    doc["rule"] = RULE
    if args.what == "extractor":
        ps = load_personas()
        doc["extractor"] = extractor_report(ps)
        doc["served_top10_meets_needs"] = needs_met_in_top10(load_run(BASELINE)[0], ps)
    elif args.what == "dev":
        doc["dev"] = dev_comparison(tuple(args.variants), judge=not args.no_judge)
    elif args.what == "crossfit":
        res = crossfit_comparison(tuple(args.variants), judge=not args.no_judge)
        if args.no_judge:
            print(json.dumps(res["holes"]))
            return
        doc["dev_crossfit"] = res
    else:
        doc["stage_ms"] = _stage_ms(tuple(args.variants))
    (paths.RESULTS_DIR / OUT).write_text(dumps(doc), encoding="utf-8")
    print(f"wrote {paths.RESULTS_DIR / OUT}")


if __name__ == "__main__":
    main()
