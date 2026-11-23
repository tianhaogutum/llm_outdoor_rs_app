"""Warm end-to-end latency of the served default, the Phase 3 default and its ranker-API variant (live calls,
cache off).

    cd backend && python -m evalkit.latency                         # make latency -> eval/results/_latency.json
    cd backend && python -m evalkit.latency --systems recsys_serve  # re-measure some systems, keep the others

Why a separate benchmark: the evaluation's nominal latency adds the latency *recorded* when each
upstream call was first made. Those recordings include cold starts (SDK client creation, auth, TLS)
for the first few of hundreds of concurrent calls, which is not what a running API process sees. Here
one process with one warm client sends requests **sequentially** (one request in flight, no queueing),
the LLM cache disabled, after ``WARMUP`` discarded requests, and measures wall time of ``arecommend``.
Dev personas only. Machine-dependent: the file records the machine; it is not part of the replay check.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np

from app.llm import run_sync
from app.recsys import build_system, get_catalog
from evalkit import paths
from evalkit.evaluate import dumps
from evalkit.llm import make_client
from evalkit.personas import load_personas

SYSTEMS = ("recsys_serve", "recsys_v1", "v7_mmr")
N = 60
WARMUP = 5
OUT = "_latency.json"
BUDGET_MS = {"with_rank_api": 1500.0, "without_rank_api": 500.0}


def main(argv=None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    args = ap.parse_args(argv)
    catalog = get_catalog()
    personas = [p for p in load_personas() if p.split == "dev"][: N + WARMUP]
    client = make_client(
        "latency", cache_enabled=False, embed_provider="vertex", rank_provider="openrouter", budget_usd=0.5
    )
    out: dict[str, Any] = {
        "schema": "evalkit.latency/v1",
        "note": "sequential warm requests, LLM cache off, wall time of Pipeline.arecommend; machine-dependent",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "n": N,
        "warmup": WARMUP,
        "budget_ms_p95": BUDGET_MS,
        "systems": {},
    }
    prev = paths.RESULTS_DIR / OUT
    if set(args.systems) != set(SYSTEMS) and prev.exists():  # keep the systems not re-measured now
        old = json.loads(prev.read_text(encoding="utf-8"))
        out["systems"] = {n: v for n, v in old["systems"].items() if n not in args.systems}
        out["real_usd_earlier_runs"] = old.get("real_usd", 0.0) + old.get("real_usd_earlier_runs", 0.0)
    for name in args.systems:
        sysm = build_system(name, catalog, llm=client)

        async def go(sysm=sysm):
            rows = []
            for i, p in enumerate(personas):
                t0 = time.perf_counter()
                rec = await sysm.arecommend(p.to_query(), 10)
                wall = (time.perf_counter() - t0) * 1000
                if i >= WARMUP:
                    st = {t.stage: (t.local_ms, t.upstream_latency_ms) for t in rec.trace}
                    rows.append(
                        {"wall_ms": wall, "local_ms": rec.local_ms, "stages": st, "cost_usd": rec.upstream_cost_usd}
                    )
            return rows

        rows = run_sync(go())
        wall = [r["wall_ms"] for r in rows]
        stages = sorted({s for r in rows for s in r["stages"]})
        api = sysm.uses_rank_api
        p95 = float(np.percentile(wall, 95))
        budget = BUDGET_MS["with_rank_api" if api else "without_rank_api"]
        out["systems"][name] = {
            "uses_rank_api": api,
            "wall_ms": {"p50": float(np.percentile(wall, 50)), "p95": p95, "max": float(np.max(wall))},
            "local_compute_ms": {
                "p50": float(np.percentile([r["local_ms"] for r in rows], 50)),
                "p95": float(np.percentile([r["local_ms"] for r in rows], 95)),
            },
            "per_stage_ms_p50": {
                s: {
                    "local": float(np.median([r["stages"][s][0] for r in rows if s in r["stages"]])),
                    "upstream": float(np.median([r["stages"][s][1] for r in rows if s in r["stages"]])),
                }
                for s in stages
            },
            "upstream_ms_p95": {
                s: float(np.percentile([r["stages"][s][1] for r in rows if s in r["stages"]], 95))
                for s in stages
                if any(r["stages"][s][1] > 0 for r in rows if s in r["stages"])
            },
            "cost_usd_per_request": float(np.mean([r["cost_usd"] for r in rows])),
            "budget_ms_p95": budget,
            "within_budget": p95 <= budget,
        }
        print(name, out["systems"][name]["wall_ms"], out["systems"][name]["upstream_ms_p95"])
    out["systems"] = {n: out["systems"][n] for n in SYSTEMS if n in out["systems"]}
    out["real_usd"] = client.usage_report()["real_usd"]
    (paths.RESULTS_DIR / OUT).write_text(dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
