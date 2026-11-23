"""Scripted chat evaluation: run every case through the real turn logic for one or more candidate models.

    python -m evalkit.chat.run --candidate g35f-min     # one candidate (live where the cache misses)
    python -m evalkit.chat.run --all                    # every candidate, then the comparison table
    python -m evalkit.chat.run --all --cache-only       # replay: no provider is ever called (make chat-eval)
    python -m evalkit.chat.run --all --freeze           # also export the touched cache to eval/chat/cache/

Every LLM call goes through ``app.llm`` (cache, budget, ledger). Recorded calls replay from the
committed frozen cache ``eval/chat/cache/<candidate>/``, so a rerun is deterministic and free.
Results: ``eval/results/chat/<candidate>.json`` and ``eval/results/_chat_eval.json`` (the summary that
``make report`` renders). Everything except each file's ``volatile`` block is byte-identical on a replay.

The cases (``eval/chat/utterances.jsonl``) and their expected slots were written by the same agent
that wrote the extractor: see docs/CHAT.md section 8 for what that does to the numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.chat import ChatSession, ChatSlots, ChatState, IndexPlaceResolver  # noqa: E402
from app.chat.engine import default_extractor  # noqa: E402
from app.chat.extract import PROMPT_VERSION, SYSTEM_PROMPT, _shingles, _words  # noqa: E402
from app.chat.llm import ChatModel, build_chat_client  # noqa: E402
from app.llm import CacheMiss, LLMConfig, LLMError, SchemaValidationError  # noqa: E402
from evalkit import paths  # noqa: E402
from evalkit.chat.scoring import SCALAR_SLOTS, TAG_SLOTS, aggregate, percentile, score_case  # noqa: E402

DATA = paths.EVAL_DIR / "chat" / "utterances.jsonl"
RESULTS = paths.RESULTS_DIR / "chat"
SUMMARY = paths.RESULTS_DIR / "_chat_eval.json"
FROZEN = paths.EVAL_DIR / "chat" / "cache"
CATALOG = BACKEND / "data" / "catalog"
PROTOCOL_VERSION = "chat-eval-v1"


@dataclass(frozen=True)
class Candidate:
    name: str
    provider: str
    model: str
    location: Optional[str] = None
    thinking: Any = 0
    note: str = ""

    def chat_model(self) -> ChatModel:
        return ChatModel(self.provider, self.model, self.location, self.thinking)


CANDIDATES: dict[str, Candidate] = {
    "rules": Candidate("rules", "mock", "mock-llm-1", thinking=None,
                       note="keyword extractor (the offline mock), not an LLM: a floor"),
    "g25f": Candidate("g25f", "vertex", "gemini-2.5-flash", thinking=0,
                      note="the app's default LLM, europe-west4, past its retirement date"),
    "g35f-min": Candidate("g35f-min", "vertex", "gemini-3.5-flash", "global", "minimal",
                          note="GA, global endpoint only for this project"),
    "g35fl-min": Candidate("g35fl-min", "vertex", "gemini-3.5-flash-lite", "global", "minimal",
                           note="cheaper Flash-Lite, global endpoint"),
    "haiku55": Candidate("haiku55", "openrouter", "anthropic/claude-haiku-5.5", thinking=0,
                         note="second vendor via OpenRouter"),
}


def load_cases(path: Path = DATA) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def place_key(ref: Any) -> Optional[str]:
    if ref is None:
        return None
    return f"{ref.name}|{ref.admin_district or ''}"


def make_client(cand: Candidate, *, cache_only: bool):
    cfg = LLMConfig.from_settings()
    frozen = tuple(p for p in sorted(FROZEN.glob("*")) if p.is_dir()) if FROZEN.exists() else ()
    overrides: dict[str, Any] = {"frozen_dirs": (*cfg.frozen_dirs, *frozen), "cache_only": cache_only}
    if cand.provider == "mock":  # offline and free: never cached, never blocked by replay mode
        overrides.update(cache_enabled=False, cache_only=False)
    budget = float(os.environ.get("CHAT_EVAL_BUDGET_USD", "1.0"))
    return build_chat_client(cand.chat_model(), config=cfg, budget_usd=budget, **overrides)


async def _start_slots(start: dict[str, Any], resolver: IndexPlaceResolver) -> ChatSlots:
    data = dict(start)
    place = data.pop("place", None)
    slots = ChatSlots(**data)
    if place:
        slots.place = await resolver(place)
        assert slots.place is not None, f"start place {place!r} does not resolve"
    return slots


async def run_case(case: dict[str, Any], client, resolver: IndexPlaceResolver) -> dict[str, Any]:
    start = await _start_slots(case.get("start", {}), resolver)
    session = ChatSession(ChatState(), start.model_copy(deep=True), {}, resolver=resolver,
                          extractor=default_extractor(client))
    calls, last_status, last_question = [], None, None
    for message in case["turns"]:
        call: dict[str, Any] = {"message": message}
        try:
            res = await session.send(message)
        except SchemaValidationError as e:
            call.update(error="schema_failure", attempts=e.attempts)
            calls.append(call)
            continue
        except CacheMiss:
            # only failed calls are missing from a complete frozen cache (valid outputs are always stored)
            call.update(error="cache_miss")
            calls.append(call)
            continue
        except LLMError as e:
            call.update(error=type(e).__name__)
            calls.append(call)
            continue
        ext = res.extraction
        call.update(
            status=res.status,
            question=res.question.slot if res.question else None,
            raw=ext.raw,
            update=ext.update.model_dump(exclude_defaults=True),
            dropped=[d.model_dump() for d in res.dropped],
            changes=[c.model_dump() for c in res.changes],
            schema_attempts=ext.schema_attempts,
            model=ext.model,
            nominal_cost_usd=round(ext.nominal_cost_usd, 8),
            # the mock's latency is wall time of a regex pass: not a recorded figure, so not reported
            nominal_latency_ms=0.0 if ext.provider == "mock" else round(ext.nominal_latency_ms, 1),
            _real_cost_usd=ext.cost_usd,
            _cached=ext.cached,
        )
        calls.append(call)
        last_status, last_question = res.status, (res.question.slot if res.question else None)
    final = session.slots
    sources = session.state.sources
    changes: dict[str, Any] = {}
    for slot in SCALAR_SLOTS:
        new, old = getattr(final, slot), getattr(start, slot)
        if new != old and sources.get(slot) != "default" and new is not None:
            changes[slot] = place_key(new) if slot == "place" else new
    for slot in TAG_SLOTS:
        added = [t for t in getattr(final, slot) if t not in getattr(start, slot)]
        if added:
            changes[slot] = added
    intent = final.intent
    leak = bool(intent and _shingles(_words(intent), 6) & _shingles(_words(SYSTEM_PROMPT), 6))
    return {
        "id": case["id"], "group": case["group"], "lang": case["lang"], "turns": case["turns"],
        "calls": calls, "changes": changes, "clarify": last_question if last_status == "clarify" else None,
        "intent": intent, "intent_prompt_overlap_6w": leak,
        "final_slots": final.model_dump(), "defaults_applied": list(session.state.defaults_applied),
    }


async def normalise_case(case: dict[str, Any], resolver: IndexPlaceResolver) -> dict[str, Any]:
    """Expected places as resolver keys, so typo'd and exact names compare on the same locality."""
    c = json.loads(json.dumps(case))
    for block in ("expect", "optional"):
        if c.get(block) and c[block].get("place"):
            ref = await resolver(c[block]["place"])
            assert ref is not None, f"{case['id']}: expected place {c[block]['place']!r} does not resolve"
            c[block]["place"] = place_key(ref)
    return c


async def run_candidate(cand: Candidate, cases: list[dict[str, Any]], resolver: IndexPlaceResolver, *,
                        cache_only: bool, freeze: bool, concurrency: int = 6) -> dict[str, Any]:
    client = make_client(cand, cache_only=cache_only)
    sem = asyncio.Semaphore(concurrency)

    async def one(case):
        async with sem:
            return await run_case(case, client, resolver)

    t0 = time.perf_counter()
    outcomes = await asyncio.gather(*(one(c) for c in cases))
    wall = time.perf_counter() - t0
    norm = [await normalise_case(c, resolver) for c in cases]
    scored = [score_case(c, o) for c, o in zip(norm, outcomes, strict=True)]
    metrics = aggregate(scored, [c["group"] for c in cases])

    calls = [k for o in outcomes for k in o["calls"]]
    ok = [k for k in calls if "error" not in k]
    attempts = sum(k.get("schema_attempts", 0) for k in ok) + sum(k.get("attempts", 1) for k in calls
                                                                    if k.get("error") == "schema_failure")
    violations = sum(k["schema_attempts"] - 1 for k in ok) + sum(k.get("attempts", 1) for k in calls
                                                                 if k.get("error") == "schema_failure")
    failures = sum(1 for k in calls if k.get("error") in ("schema_failure", "cache_miss"))
    drops = [d for k in ok for d in k["dropped"]]
    lat = [k["nominal_latency_ms"] for k in ok]
    injection = [o for o in outcomes if o["group"] == "injection"]
    inj_scored = [s for s, o in zip(scored, outcomes, strict=True) if o["group"] == "injection"]
    for o, s in zip(outcomes, scored, strict=True):
        o["score"] = {"errors": s["errors"], "clarify": s["clarify"], "intent_hit": s["intent_hit"]}
    metrics["schema"] = {
        "calls": len(calls), "provider_attempts": attempts, "violations": violations,
        "violation_rate": round(violations / attempts, 4) if attempts else None,
        "failed_calls": failures, "failure_rate": round(failures / len(calls), 4) if calls else None,
        "other_errors": sum(1 for k in calls if k.get("error") not in (None, "schema_failure", "cache_miss")),
    }
    by_reason: dict[str, int] = {}
    for d in drops:
        key = f"{d['field']}: {d['reason']}"
        by_reason[key] = by_reason.get(key, 0) + 1
    metrics["validator_drops"] = {"total": len(drops), "per_call": round(len(drops) / len(ok), 4) if ok else None,
                                  "by_reason": dict(sorted(by_reason.items()))}
    metrics["injection"] = {
        "cases": len(injection),
        "cases_with_unexpected_change": sum(1 for s in inj_scored if any(e["kind"] in ("fp", "wrong")
                                                                         for e in s["errors"])),
        "intent_prompt_overlap": sum(o["intent_prompt_overlap_6w"] for o in injection),
        "out_of_range_values_stored": 0,  # by construction (SlotUpdate limits); asserted in tests
    }
    metrics["latency_ms"] = {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "basis": "nominal"}
    nominal = sum(k["nominal_cost_usd"] for k in ok)
    metrics["cost_usd"] = {"total": round(nominal, 6), "per_call": round(nominal / len(ok), 8) if ok else None,
                           "basis": "nominal (what the calls cost when recorded)"}
    report = client.usage_report()
    models_used = sorted({k["model"] for k in ok if k.get("model")})
    for o in outcomes:  # volatile per-call fields out of the deterministic part
        for k in o["calls"]:
            k.pop("_real_cost_usd", None)
            k.pop("_cached", None)
    result = {
        "protocol": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "candidate": asdict(cand),
        "models_used": models_used,
        "cases_file": "eval/chat/utterances.jsonl",
        "metrics": metrics,
        "cases": outcomes,
        "volatile": {
            "real_usd": report["real_usd"], "wall_s": round(wall, 1),
            "cache_hits": sum(s.get("cache_hits", 0) for s in report["totals"].values()),
            "provider_requests": sum(s.get("provider_requests", 0) for s in report["totals"].values()),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    if freeze and cand.provider != "mock":
        dest = FROZEN / cand.name
        if dest.exists():
            for f in dest.iterdir():
                f.unlink()
        client.export_frozen(dest)
    await client.aclose()
    client.close()
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def summary_row(r: dict[str, Any]) -> dict[str, Any]:
    m = r["metrics"]
    return {
        "candidate": r["candidate"]["name"], "provider": r["candidate"]["provider"],
        "model": r["candidate"]["model"], "location": r["candidate"]["location"],
        "thinking": r["candidate"]["thinking"],
        "overall": m["overall"], "per_type": m["per_type"],
        "clarify_decision_accuracy": m["clarify"]["decision_accuracy"],
        "clarify_slot_accuracy": m["clarify"]["slot_accuracy"],
        "intent_mention_hit_rate": m["intent_mentions"]["hit_rate"],
        "schema_violation_rate": m["schema"]["violation_rate"], "schema_failed_calls": m["schema"]["failed_calls"],
        "validator_drops_per_call": m["validator_drops"]["per_call"],
        "injection_cases_with_unexpected_change": m["injection"]["cases_with_unexpected_change"],
        "latency_p50_ms": m["latency_ms"]["p50"], "latency_p95_ms": m["latency_ms"]["p95"],
        "cost_usd_per_call": m["cost_usd"]["per_call"], "cost_usd_total": m["cost_usd"]["total"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", choices=sorted(CANDIDATES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cache-only", action="store_true", help="replay only; a cache miss is a failed call")
    ap.add_argument("--freeze", action="store_true", help="export the touched cache to eval/chat/cache/<name>/")
    ap.add_argument("--cases", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    names = sorted(CANDIDATES) if args.all else (args.candidate or [])
    if not names:
        ap.error("give --candidate NAME or --all")
    cases = load_cases(args.cases)
    resolver = IndexPlaceResolver.from_catalog(CATALOG)
    rows = []
    for name in names:
        res = asyncio.run(run_candidate(CANDIDATES[name], cases, resolver, cache_only=args.cache_only,
                                        freeze=args.freeze))
        write_json(args.out / f"{name}.json", res)
        rows.append(summary_row(res))
        m = res["metrics"]
        print(f"[chat-eval] {name}: P={m['overall']['precision']} R={m['overall']['recall']} "
              f"F1={m['overall']['f1']} clarify={m['clarify']['decision_accuracy']} "
              f"schema_viol={m['schema']['violation_rate']} real_usd={res['volatile']['real_usd']:.4f} "
              f"hits={res['volatile']['cache_hits']} requests={res['volatile']['provider_requests']}",
              file=sys.stderr)
    if args.all:
        summary = SUMMARY if args.out == RESULTS else args.out / "summary.json"
        write_json(summary, {"protocol": PROTOCOL_VERSION, "cases": len(cases), "candidates": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
