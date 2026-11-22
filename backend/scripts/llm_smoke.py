#!/usr/bin/env python3
"""
Live smoke test of the LLM client for the configured providers.

One generate, one structured generate, one embed (query + document) and one
rank. Prints provider, model id, latency, cost and cache status; never prints
credentials or the project id. A second run should be all cache hits.

Usage (from the repo root or backend/):
    backend/.venv/bin/python backend/scripts/llm_smoke.py              # providers from settings
    backend/.venv/bin/python backend/scripts/llm_smoke.py --narrative  # + one live genai_service call
    LLM_PROVIDER=mock backend/.venv/bin/python backend/scripts/llm_smoke.py   # offline

Spend is capped at --budget USD (default 0.10) regardless of LLM_BUDGET_USD_PER_RUN.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.llm import LLMClient, LLMConfig, LLMError, set_client  # noqa: E402


class RouteFacts(BaseModel):
    """Structured-output probe."""

    mountain: str
    country: str
    height_m: int = Field(ge=0, le=9000)
    difficulty: Literal["easy", "moderate", "hard"]


def line(cap: str, res, extra: str = "") -> None:
    cached = "HIT " if getattr(res, "cached", False) else "MISS"
    print(
        f"{cap:<10} {res.provider:<10} {res.model:<34} cache={cached} "
        f"latency={res.latency_ms:8.1f}ms (nominal {res.nominal_latency_ms:8.1f}ms) "
        f"cost=${res.cost_usd:.6f} (nominal ${res.nominal_cost_usd:.6f}) {extra}"
    )


async def main(args: argparse.Namespace) -> int:
    cfg = LLMConfig.from_settings()
    cfg = cfg.with_(budget_usd=min(cfg.budget_usd, args.budget))
    client = LLMClient(cfg)
    set_client(client)
    print(
        f"providers: generate={cfg.llm_provider} embed={cfg.embed_provider} rank={cfg.rank_provider} | "
        f"budget=${cfg.budget_usd:.2f} | cache={'on' if cfg.cache_enabled else 'off'}"
    )
    try:
        g = await client.generate(
            "In one sentence: what is a via ferrata?", max_tokens=80, temperature=0.0, tag="smoke.generate"
        )
        line("generate", g, f"tokens={g.usage.input_tokens}/{g.usage.output_tokens}/{g.usage.thinking_tokens}")
        print(f"           -> {g.text[:120]!r}")

        s = await client.generate(
            "Give facts about Germany's highest mountain.",
            system="Answer with accurate facts.",
            schema=RouteFacts,
            max_tokens=200,
            temperature=0.0,
            tag="smoke.structured",
        )
        line("structured", s, f"attempts={s.schema_attempts}")
        print(f"           -> {s.parsed!r}")

        q = await client.embed(["easy lake walk for a family with kids"], task="query", tag="smoke.embed")
        d = await client.embed(
            [
                "A flat 5 km loop around a quiet lake, with a playground and a cafe at the start.",
                "A steep 1,400 m ascent on a secured via ferrata; helmet and harness required.",
            ],
            task="document",
            tag="smoke.embed",
        )
        sims = d.vectors @ q.vectors[0]
        line("embed.q", q, f"shape={q.vectors.shape} norm={np.linalg.norm(q.vectors[0]):.3f}")
        line("embed.d", d, f"shape={d.vectors.shape} cos(q,d)={[round(float(x), 3) for x in sims]}")

        docs = [
            "Steep alpine scramble with exposed ridges.",
            "Flat stroller-friendly loop around the lake with playground.",
            "City museum tour through the old town.",
        ]
        r = await client.rank("easy lake walk for a family with kids", docs, top_n=3, tag="smoke.rank")
        line("rank", r, f"scores={[round(x, 3) for x in r.scores]} order={r.order}")

        if args.narrative:
            from app.services.genai_service import generate_post_run_summary

            text = await generate_post_run_summary("Herzogstand Loop", 9.3, 2, 3, 4)
            n = client.usage_report()["by_tag"]["narrative.post_run"]["generate"]
            print(
                f"narrative  {cfg.llm_provider:<10} {cfg.default_model('generate'):<34} cache=OFF  "
                f"latency={n['real_latency_ms']:8.1f}ms cost=${n['real_usd']:.6f}"
            )
            print(f"           -> {text[:300]!r}")
    except LLMError as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 1
    finally:
        await client.aclose()

    rep = client.usage_report()
    print(
        f"total: real=${rep['real_usd']:.6f} nominal=${rep['nominal_usd']:.6f} "
        f"remaining=${rep['budget_remaining_usd']:.4f}"
    )
    print("models_used:", json.dumps(rep["models_used"]))
    if args.report:
        print(json.dumps(rep, indent=2))
    client.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--narrative", action="store_true", help="also call one genai_service narrative live")
    ap.add_argument("--budget", type=float, default=0.10, help="spend cap in USD for this run (default 0.10)")
    ap.add_argument("--report", action="store_true", help="print the full usage_report() JSON")
    sys.exit(asyncio.run(main(ap.parse_args())))
