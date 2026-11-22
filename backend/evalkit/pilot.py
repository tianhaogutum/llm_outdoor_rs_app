"""Judge pilot: pick the production judge on agreement, cost and lifecycle risk (``make judge-pilot``).

- 20 dev personas stratified over activity style and party; 4 pilot pairs each (80 pairs) spanning the
  grade range: one of the latent matcher's top 3, one of its ranks 4-15, one from v0_rules' top 50 that
  passes the hard constraints (else a mid-ranked feasible route), one random feasible route.
- Candidates grade the 4 pilot pairs inside a batch of 8 (4 filler routes of the same persona, seeded
  order); ``-b4`` candidates grade the 4 pilot pairs alone. The reference (a clearly stronger model from
  another vendor, reasoning on) grades the 4 pilot pairs alone.
- Batch-order sensitivity: selected candidates grade the same pilot pairs again with different fillers
  and a different order.
- Output: ``eval/pilot/pilot_records.jsonl`` and ``eval/results/_judge_pilot.json``.

Cheap on reruns: everything is cached.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import numpy as np

from app.llm import run_sync
from app.recsys import build_system, get_catalog
from evalkit import paths
from evalkit.judge import JUDGES, _seed, client_for, judge_batches_async
from evalkit.metrics import confusion_matrix, weighted_kappa
from evalkit.personas import load_personas
from evalkit.pool import feasible_indices, latent_utility

PILOT_SEED = 7
N_PERSONAS = 20
CANDIDATES = ["g25f-t0-b8", "g25f-t1k-b8", "g35f-low-b8", "g35f-min-b8", "g25f-t0-b4"]
REFERENCE = "ref-opus55"
SENSITIVITY = ["g25f-t0-b8", "g35f-low-b8", "g35f-min-b8"]
STRATA = {"hiking": 12, "touring": 5, "mtb": 3}


def select(catalog, personas) -> list[dict[str, Any]]:
    rng = np.random.default_rng(PILOT_SEED)
    dev = [p for p in personas if p.split == "dev"]
    chosen = []
    for style, n in STRATA.items():
        group = [p for p in dev if p.latent["style"] == style]
        # spread over parties: round-robin over party kinds in a seeded order
        by_party: dict[str, list] = {}
        for p in group:
            by_party.setdefault(p.latent["party"]["kind"], []).append(p)
        kinds = sorted(by_party)
        rng.shuffle(kinds)
        for k in kinds:
            rng.shuffle(by_party[k])
        while len([c for c in chosen if c.latent["style"] == style]) < n:
            for k in kinds:
                if by_party[k] and len([c for c in chosen if c.latent["style"] == style]) < n:
                    chosen.append(by_party[k].pop())
    v0 = build_system("v0_rules", catalog)
    out = []
    for p in sorted(chosen, key=lambda p: p.persona_id):
        idx = feasible_indices(catalog, p)
        ranked = [
            int(catalog.ids[i])
            for i in sorted(idx, key=lambda i: (-latent_utility(p, catalog.routes[i]), int(catalog.ids[i])))
        ]
        feas = set(ranked)
        picks: list[int] = []

        def take(cands, picks=picks):
            cands = [c for c in cands if c not in picks]
            if cands:
                picks.append(int(cands[int(rng.integers(len(cands)))]))

        take(ranked[:3])
        take(ranked[3:15])
        v0_top = [r for r in v0.recommend(p.to_query(), 50).route_ids if r in feas]
        take(v0_top or ranked[15:100])
        take(ranked[15:])
        rest = [r for r in ranked if r not in picks]
        fill = [int(rest[i]) for i in rng.permutation(len(rest))[:8]]
        out.append({"persona": p, "pilot": picks, "fill1": fill[:4], "fill2": fill[4:8]})
    return out


def _ordered(ids: list[int], seed: int, pid: str) -> list[int]:
    rng = np.random.default_rng(_seed("pilot-order", seed, pid))
    return [ids[i] for i in rng.permutation(len(ids))]


def run(args) -> dict[str, Any]:
    catalog = get_catalog()
    personas = load_personas()
    sel = select(catalog, personas)
    pilot_pairs = [(s["persona"].persona_id, r) for s in sel for r in s["pilot"]]
    configs = [REFERENCE] + CANDIDATES
    variants = {name: ("fill1", 1) for name in configs}
    runs: dict[str, dict[tuple[str, int], int]] = {}
    meta: dict[str, dict[str, Any]] = {}
    records = []
    jobs = [(name, name, *variants[name]) for name in configs] + [(f"{n}#order2", n, "fill2", 2) for n in SENSITIVITY]
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only.split(",")]
    for label, cfg_name, fill_key, seed in jobs:
        cfg = JUDGES[cfg_name]
        batches = []
        for s in sel:
            ids = list(s["pilot"]) + (list(s[fill_key]) if cfg.batch_size >= 8 else [])
            batches.append((s["persona"], _ordered(ids, seed, s["persona"].persona_id)))
        if args.limit:
            batches = batches[: args.limit]
        client = client_for(cfg)
        out = run_sync(judge_batches_async(cfg, client, catalog, batches, tag=f"pilot:{label}"))
        grades = {(r["persona_id"], r["route_id"]): r["grade"] for r in out.records}
        runs[label] = {k: v for k, v in grades.items() if k in set(pilot_pairs)}
        n_calls = max(out.calls, 1)
        meta[label] = {
            "config": cfg.describe(),
            "calls": out.calls,
            "failures": out.failures,
            "nominal_cost_usd": round(out.nominal_cost_usd, 6),
            "nominal_cost_per_pair_usd": round(out.nominal_cost_usd / max(len(out.records), 1), 7),
            "nominal_latency_ms_per_call": round(out.nominal_latency_ms / n_calls, 1),
            "models": sorted({r["model"] for r in out.records}),
        }
        for r in out.records:
            records.append(
                {
                    "run": label,
                    **{k: r[k] for k in ("persona_id", "route_id", "grade", "reason", "model")},
                    "pilot_pair": (r["persona_id"], r["route_id"]) in set(pilot_pairs),
                }
            )
        print(
            f"[pilot] {label}: {len(runs[label])} pilot grades, {out.calls} calls, "
            f"nominal {out.nominal_cost_usd:.4f} USD"
        )
    return {"selection": sel, "pairs": pilot_pairs, "runs": runs, "meta": meta, "records": records}


def summarise(res: dict[str, Any]) -> dict[str, Any]:
    ref = res["runs"].get(REFERENCE, {})
    table = []
    for label, grades in res["runs"].items():
        if label == REFERENCE:
            continue
        keys = [k for k in res["pairs"] if k in grades and k in ref]
        a = [ref[k] for k in keys]
        b = [grades[k] for k in keys]
        row = {
            "run": label,
            "n": len(keys),
            "qwk_vs_reference": weighted_kappa(a, b),
            "exact_agreement": float(np.mean([x == y for x, y in zip(a, b, strict=True)])) if keys else None,
            "within_one": float(np.mean([abs(x - y) <= 1 for x, y in zip(a, b, strict=True)])) if keys else None,
            "mean_grade_minus_reference": float(np.mean(np.array(b) - np.array(a))) if keys else None,
            "confusion_reference_rows": confusion_matrix(a, b),
            "grade_distribution": dict(sorted(Counter(b).items())),
            **res["meta"][label],
        }
        table.append(row)
    sens = []
    for n in SENSITIVITY:
        a_run, b_run = res["runs"].get(n, {}), res["runs"].get(f"{n}#order2", {})
        keys = [k for k in res["pairs"] if k in a_run and k in b_run]
        if keys:
            a = [a_run[k] for k in keys]
            b = [b_run[k] for k in keys]
            sens.append(
                {
                    "config": n,
                    "n": len(keys),
                    "qwk_between_orders": weighted_kappa(a, b),
                    "exact_agreement": float(np.mean([x == y for x, y in zip(a, b, strict=True)])),
                    "confusion": confusion_matrix(a, b),
                }
            )
    return {
        "schema": "evalkit.judge_pilot/v1",
        "reference": REFERENCE,
        "reference_meta": res["meta"].get(REFERENCE),
        "reference_grade_distribution": dict(sorted(Counter(ref.values()).items())),
        "n_pairs": len(res["pairs"]),
        "n_personas": len(res["selection"]),
        "strata": {k: sum(1 for s in res["selection"] if s["persona"].latent["style"] == k) for k in STRATA},
        "candidates": table,
        "batch_order_sensitivity": sens,
    }


def main(argv=None) -> None:
    from evalkit.llm import print_usage  # noqa: F401  (spend log hook)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated run labels (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N personas (smoke tests, not written)")
    args = ap.parse_args(argv)
    res = run(args)
    summary = summarise(res)
    if args.limit or args.only:
        print(json.dumps(summary, indent=1)[:4000])
        return
    paths.PILOT_DIR.mkdir(parents=True, exist_ok=True)
    with (paths.PILOT_DIR / "pilot_records.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(res["records"], key=lambda r: (r["run"], r["persona_id"], r["route_id"])):
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS_DIR / "_judge_pilot.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in summary["candidates"]:
        print(
            f"{row['run']:22s} n={row['n']:3d} qwk={row['qwk_vs_reference']:.3f} exact={row['exact_agreement']:.2f} "
            f"bias={row['mean_grade_minus_reference']:+.2f} cost/pair={row['nominal_cost_per_pair_usd']:.5f}"
        )
    for s in summary["batch_order_sensitivity"]:
        print(f"order sensitivity {s['config']}: qwk={s['qwk_between_orders']:.3f} exact={s['exact_agreement']:.2f}")


if __name__ == "__main__":
    main()
