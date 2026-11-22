"""LLM client construction for the harness, plus a local spend log across processes.

Every call goes through ``app.llm`` (cache, budget, ledger). ``make_client`` builds a client from
settings with overrides; unless ``LLM_CACHE_FROZEN_DIRS`` is set explicitly, the committed frozen
layer ``eval/cache/`` is added so a fresh clone replays recorded calls without credentials.
Each process appends its usage report to ``.cache/llm/spend_log.jsonl`` (git-ignored) so the total
real spend over many runs can be summed with ``python -m evalkit.llm``.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from app.llm import LLMClient, LLMConfig
from evalkit import paths

_clients: list[tuple[str, LLMClient]] = []


def _spend_log() -> Path:
    from app.settings import get_settings

    return Path(get_settings().resolved_llm_cache_dir) / "spend_log.jsonl"


def make_client(label: str = "default", **overrides: Any) -> LLMClient:
    """Client from settings plus overrides. Use one client per (provider, model) so the spend log can
    attribute cost (pass the model as a config override, e.g. ``openrouter_llm_model=...``)."""
    cfg = LLMConfig.from_settings()
    if (
        "LLM_CACHE_FROZEN_DIRS" not in os.environ
        and paths.FROZEN_DIR.exists()
        and paths.FROZEN_DIR not in cfg.frozen_dirs
    ):
        overrides.setdefault("frozen_dirs", (*cfg.frozen_dirs, paths.FROZEN_DIR))
    client = LLMClient(cfg.with_(**overrides))
    _clients.append((label, client))
    return client


def print_usage(client: LLMClient, file=sys.stderr) -> None:
    r = client.usage_report()
    tags = (
        {
            f"{cap}:{tag}": (s["calls"], s["cache_hits"], s["provider_requests"], s["real_usd"])
            for (tag, caps) in r["by_tag"].items()
            for cap, s in caps.items()
        }
        if isinstance(r["by_tag"], dict)
        else {}
    )
    print(
        f"[llm] real spend {r['real_usd']:.4f} USD (nominal {r['nominal_usd']:.4f}); "
        f"by tag (calls, cache hits, requests, usd): {tags}",
        file=file,
    )


def _log_spend() -> None:
    rows = []
    for label, c in _clients:
        r = c.usage_report()
        if r["real_usd"] <= 0 and not any(s.get("provider_requests") for s in r["totals"].values()):
            continue
        rows.append(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "argv": sys.argv[:3],
                "label": label,
                "real_usd": r["real_usd"],
                "totals": r["totals"],
                "models_used": r["models_used"],
                "providers": r["providers"],
            }
        )
    if not rows:
        return
    try:
        p = _spend_log()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        pass


atexit.register(_log_spend)


def spend_summary() -> dict[str, Any]:
    p = _spend_log()
    out: dict[str, Any] = {"total_usd": 0.0, "by_provider": {}, "by_model": {}}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out["total_usd"] += row["real_usd"]
        for cap, s in row["totals"].items():
            prov = row["providers"][cap]["provider"]
            model = row["providers"][cap]["model"]
            out["by_provider"][prov] = round(out["by_provider"].get(prov, 0.0) + s["real_usd"], 6)
            key = f"{prov}:{model}"
            out["by_model"][key] = round(out["by_model"].get(key, 0.0) + s["real_usd"], 6)
    out["total_usd"] = round(out["total_usd"], 6)
    return out


if __name__ == "__main__":
    print(json.dumps(spend_summary(), indent=1))
