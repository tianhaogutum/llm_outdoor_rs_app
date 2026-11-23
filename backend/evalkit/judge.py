"""LLM relevance judge (grades 0-3, rubric in ``eval/qrels/rubric_v1.md``).

- Pairs that violate a hard constraint are graded 0 by rule and never reach the LLM (:func:`rule_grade`).
- The judge sees the persona's full latent profile and intent and a factual route card; persona and
  catalogue text go through ``wrap_untrusted``.
- Several routes of the same persona are graded per call, in a seeded order; structured output,
  temperature 0. Every call goes through ``app.llm`` (cache, budget, ledger).
- Card version (``evalkit.cards``, D29): new judgements use ``card/v2`` and record ``"card": "card/v2"``; records
  without the key were graded with ``card/v1``. A pair re-graded because of the v1 rounding keeps its earlier
  judgement under ``"previous"``; :func:`replay_records` yields both, so a replay re-issues and checks both.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from app.llm import UNTRUSTED_DATA_RULE, LLMClient, LLMConfig, SchemaValidationError, wrap_untrusted
from app.recsys import Catalog
from evalkit import paths
from evalkit.cards import CARD_V1, CARD_VERSION, persona_card, route_card
from evalkit.personas import Persona

RUBRIC_VERSION = "rubric/v1"
RUBRIC_FILE = paths.QRELS_DIR / "rubric_v1.md"
PROMPT_VERSION = "judge/v1"


def rubric_text() -> str:
    return RUBRIC_FILE.read_text(encoding="utf-8")


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    provider: str  # vertex | openrouter
    model: str
    location: Optional[str] = None  # vertex only; None = configured region
    thinking: Union[int, str, None] = 0
    batch_size: int = 8
    max_tokens: int = 2500

    def client_overrides(self) -> dict[str, Any]:
        if self.provider == "vertex":
            o: dict[str, Any] = {"llm_provider": "vertex", "vertex_llm_model": self.model}
            if self.location:
                o["vertex_llm_location"] = self.location
            return o
        return {"llm_provider": "openrouter", "openrouter_llm_model": self.model}

    def describe(self) -> dict[str, Any]:
        return asdict(self)


JUDGES: dict[str, JudgeConfig] = {
    # pilot candidates
    "g25f-t0-b8": JudgeConfig("g25f-t0-b8", "vertex", "gemini-2.5-flash", thinking=0, batch_size=8),
    "g25f-t0-b4": JudgeConfig("g25f-t0-b4", "vertex", "gemini-2.5-flash", thinking=0, batch_size=4),
    "g25f-t1k-b8": JudgeConfig(
        "g25f-t1k-b8", "vertex", "gemini-2.5-flash", thinking=1024, batch_size=8, max_tokens=4000
    ),
    "g35f-low-b8": JudgeConfig(
        "g35f-low-b8", "vertex", "gemini-3.5-flash", location="global", thinking="low", batch_size=8, max_tokens=6000
    ),
    "g35f-min-b8": JudgeConfig(
        "g35f-min-b8",
        "vertex",
        "gemini-3.5-flash",
        location="global",
        thinking="minimal",
        batch_size=8,
        max_tokens=4000,
    ),
    # reference for the pilot (stronger model, other vendor, reasoning on)
    "ref-opus55": JudgeConfig(
        "ref-opus55", "openrouter", "anthropic/claude-opus-5.5", thinking="medium", batch_size=4, max_tokens=12000
    ),
    # interim cross-vendor check on the owner's 150 pairs (not a human check)
    "second-sonnet55": JudgeConfig(
        "second-sonnet55", "openrouter", "anthropic/claude-sonnet-5.5", thinking="low", batch_size=8, max_tokens=8000
    ),
}

# Chosen by the pilot (docs/DECISIONS.md D9, eval/results/_judge_pilot.json).
PRODUCTION_JUDGE = "g35f-min-b8"


def production_judge() -> JudgeConfig:
    return JUDGES[PRODUCTION_JUDGE]


_judge_clients: dict[str, LLMClient] = {}


def client_for(cfg: JudgeConfig, **overrides: Any) -> LLMClient:
    """One client per judge config; stays on the mock provider when LLM_PROVIDER=mock (tests, CI)."""
    from evalkit.llm import make_client

    if cfg.name in _judge_clients and not overrides:
        return _judge_clients[cfg.name]
    if LLMConfig.from_settings().llm_provider == "mock":
        c = make_client(f"judge:{cfg.name}", **overrides)
    else:
        c = make_client(f"judge:{cfg.name}", **cfg.client_overrides(), **overrides)
    if not overrides:
        _judge_clients[cfg.name] = c
    return c


# --------------------------------------------------------------------------- rules
def rule_grade(p: Persona, catalog: Catalog, route_id: int) -> Optional[dict[str, Any]]:
    """Grade 0 by rule if the pair violates a hard constraint, else None (goes to the LLM)."""
    v = p.constraints().violations(catalog.get(route_id))
    if not v:
        return None
    return {
        "persona_id": p.persona_id,
        "route_id": int(route_id),
        "grade": 0,
        "source": "rule",
        "rule": "+".join(v),
        "reason": f"violates hard constraint: {', '.join(v)}",
        "judge": None,
        "model": None,
        "rubric": RUBRIC_VERSION,
    }


# --------------------------------------------------------------------------- batching
def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def make_batches(persona_id: str, route_ids: list[int], batch_size: int, seed: int = 0) -> list[list[int]]:
    """Seeded shuffle of one persona's routes, split into balanced batches of at most ``batch_size``."""
    ids = sorted(set(int(r) for r in route_ids))
    if not ids:
        return []
    rng = np.random.default_rng(_seed(seed, persona_id, ",".join(map(str, ids))))
    ids = [ids[i] for i in rng.permutation(len(ids))]
    n_b = math.ceil(len(ids) / batch_size)
    size = math.ceil(len(ids) / n_b)
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def batch_id(cfg_name: str, persona_id: str, route_ids: list[int]) -> str:
    h = hashlib.sha256(f"{cfg_name}|{persona_id}|{','.join(map(str, route_ids))}".encode()).hexdigest()
    return h[:12]


# --------------------------------------------------------------------------- prompt
class RouteGrade(BaseModel):
    route: str = Field(description="the route label, e.g. R1")
    reason: str = Field(description="one sentence, at most 25 words, naming the deciding facts", max_length=400)
    grade: int = Field(ge=0, le=3)


class Grades(BaseModel):
    grades: list[RouteGrade]


def system_prompt() -> str:
    return (
        "You are a careful relevance assessor for an outdoor route recommender. Apply the rubric exactly.\n\n"
        + rubric_text()
        + "\n\n"
        + UNTRUSTED_DATA_RULE
    )


def judge_prompt(
    p: Persona, catalog: Catalog, route_ids: list[int], card: str = CARD_VERSION
) -> tuple[str, list[str]]:
    home = tuple(p.profile["home"])
    labels = [f"R{i + 1}" for i in range(len(route_ids))]
    parts = [
        "PERSON (profile):",
        wrap_untrusted(persona_card(p, include_intent=False, card=card), "person-profile"),
        "",
        "What the person typed into the app:",
        wrap_untrusted(p.intent or "(nothing)", "person-request"),
        "",
        f"ROUTES ({len(route_ids)}):",
    ]
    for lab, rid in zip(labels, route_ids, strict=True):
        parts.append(f"{lab}:")
        parts.append(wrap_untrusted(route_card(catalog.get(rid), home, card=card), f"route-{lab}"))
    parts.append("")
    parts.append(
        f"Grade every route ({', '.join(labels)}) for this person. Reply with JSON "
        '{"grades": [{"route": "R1", "reason": "...", "grade": 0-3}, ...]} with one entry per route, in order.'
    )
    return "\n".join(parts), labels


# --------------------------------------------------------------------------- judging
@dataclass
class JudgeOutcome:
    records: list[dict[str, Any]]
    calls: int = 0
    failures: int = 0
    nominal_cost_usd: float = 0.0
    nominal_latency_ms: float = 0.0


def card_of(rec: dict[str, Any]) -> str:
    """The card version a recorded LLM judgement was made with (records before D29 carry no key: v1)."""
    return rec.get("card", CARD_V1)


async def _judge_one_batch(
    cfg: JudgeConfig,
    client: LLMClient,
    p: Persona,
    catalog: Catalog,
    route_ids: list[int],
    tag: str,
    out: JudgeOutcome,
    card: str = CARD_VERSION,
) -> list[dict[str, Any]]:
    prompt, labels = judge_prompt(p, catalog, route_ids, card)
    try:
        res = await client.generate(
            prompt,
            system=system_prompt(),
            schema=Grades,
            temperature=0.0,
            max_tokens=cfg.max_tokens,
            thinking=cfg.thinking,
            tag=tag,
        )
    except SchemaValidationError:
        out.failures += 1
        if len(route_ids) == 1:
            raise
        half = len(route_ids) // 2
        a = await _judge_one_batch(cfg, client, p, catalog, route_ids[:half], tag, out, card)
        b = await _judge_one_batch(cfg, client, p, catalog, route_ids[half:], tag, out, card)
        return a + b
    out.calls += 1
    out.nominal_cost_usd += res.nominal_cost_usd
    out.nominal_latency_ms += res.nominal_latency_ms
    by_label = {}
    for g in res.parsed.grades:
        key = g.route.strip().upper()
        if key in labels and key not in by_label:
            by_label[key] = g
    bid = batch_id(cfg.name, p.persona_id, route_ids)
    if card != CARD_V1:  # v1 batch ids stay as recorded; a v2 batch of the same routes is a different call
        bid = batch_id(f"{cfg.name}|{card}", p.persona_id, route_ids)
    recs, missing = [], []
    for pos, (lab, rid) in enumerate(zip(labels, route_ids, strict=True)):
        g = by_label.get(lab)
        if g is None:
            missing.append(rid)
            continue
        recs.append(
            {
                "persona_id": p.persona_id,
                "route_id": int(rid),
                "grade": int(g.grade),
                "source": "llm",
                "reason": " ".join(g.reason.split()),
                "judge": cfg.name,
                "model": res.model,
                "rubric": RUBRIC_VERSION,
                "prompt": PROMPT_VERSION,
                "batch_id": bid,
                "batch_pos": pos,
                "batch_size": len(route_ids),
                "batch_ids": list(map(int, route_ids)),
                **({"card": card} if card != CARD_V1 else {}),
            }
        )
    if missing:  # the model skipped labels: grade those one at a time
        out.failures += 1
        if len(route_ids) == 1:
            raise SchemaValidationError(
                f"judge {cfg.name} returned no grade for route {route_ids[0]}", attempts=1, last_text=""
            )
        for rid in missing:
            recs += await _judge_one_batch(cfg, client, p, catalog, [rid], tag, out, card)
    return recs


async def judge_batches_async(
    cfg: JudgeConfig,
    client: LLMClient,
    catalog: Catalog,
    batches: list[tuple[Persona, list[int]] | tuple[Persona, list[int], str]],
    tag: str = "judge",
    card: str = CARD_VERSION,
) -> JudgeOutcome:
    """Judge (persona, route ids[, card]) batches; a batch without its own card version uses ``card``."""
    out = JudgeOutcome(records=[])
    # de-duplicate identical batches (the client does not de-duplicate in-flight calls)
    uniq: dict[tuple[str, tuple[int, ...], str], tuple[Persona, list[int], str]] = {}
    for b in batches:
        p, ids = b[0], b[1]
        c = b[2] if len(b) > 2 else card
        uniq.setdefault((p.persona_id, tuple(ids), c), (p, ids, c))
    results = await asyncio.gather(
        *(_judge_one_batch(cfg, client, p, catalog, ids, tag, out, c) for p, ids, c in uniq.values())
    )
    for r in results:
        out.records.extend(r)
    return out


def judge_pairs(
    cfg: JudgeConfig,
    catalog: Catalog,
    personas: dict[str, Persona],
    pairs: list[tuple[str, int]],
    *,
    client: Optional[LLMClient] = None,
    seed: int = 0,
    tag: str = "judge",
    card: str = CARD_VERSION,
) -> JudgeOutcome:
    """Grade (persona_id, route_id) pairs with ``cfg`` (batched per persona, seeded order)."""
    from app.llm import run_sync

    client = client or client_for(cfg)
    by_p: dict[str, list[int]] = {}
    for pid, rid in pairs:
        by_p.setdefault(pid, []).append(int(rid))
    batches = [(personas[pid], b) for pid in sorted(by_p) for b in make_batches(pid, by_p[pid], cfg.batch_size, seed)]
    return run_sync(judge_batches_async(cfg, client, catalog, batches, tag=tag, card=card))


def replay_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every recorded LLM judgement in ``records``: the current ones and, for pairs re-graded with a newer card,
    the superseded judgement kept under ``"previous"`` (with its persona and route filled in)."""
    out = []
    for r in records:
        if r.get("source") != "llm":
            continue
        out.append(r)
        prev = r.get("previous")
        if prev and prev.get("source") == "llm":
            out.append({**prev, "persona_id": r["persona_id"], "route_id": r["route_id"]})
    return out


def replay_problems(
    catalog: Catalog,
    personas: dict[str, Persona],
    records: list[dict[str, Any]],
    label: str,
) -> tuple[list[str], list[str]]:
    """Replay every recorded batch (current and superseded) and compare the grades. Returns (touched keys,
    problems)."""
    keys: list[str] = []
    problems: list[str] = []
    recs = replay_records(records)
    for name in sorted({r["judge"] for r in recs}):
        cfg = JUDGES[name]
        c = client_for(cfg)
        out = replay_batches(cfg, catalog, personas, recs, client=c)
        keys += c.touched_keys()
        got = {(r["persona_id"], r["route_id"], card_of(r)): r["grade"] for r in out.records}
        for r in recs:
            if r["judge"] != name:
                continue
            k = (r["persona_id"], r["route_id"], card_of(r))
            if got.get(k) != r["grade"]:
                problems.append(f"judge replay differs for {k} ({label}): {got.get(k)} vs {r['grade']}")
    return keys, problems


def replay_batches(
    cfg: JudgeConfig,
    catalog: Catalog,
    personas: dict[str, Persona],
    records: list[dict[str, Any]],
    *,
    client: Optional[LLMClient] = None,
    tag: str = "judge",
) -> JudgeOutcome:
    """Re-issue exactly the recorded batches (from ``batch_ids``, each with its card version) - used to verify a
    frozen replay. Superseded judgements are replayed only if ``records`` contains them (:func:`replay_records`)."""
    from app.llm import run_sync

    client = client or client_for(cfg)
    seen: dict[tuple[str, tuple[int, ...], str], None] = {}
    for r in records:
        if r.get("source") == "llm" and r.get("judge") == cfg.name:
            seen.setdefault((r["persona_id"], tuple(r["batch_ids"]), card_of(r)), None)
    batches = [(personas[pid], list(ids), card) for pid, ids, card in seen]
    return run_sync(judge_batches_async(cfg, client, catalog, batches, tag=tag))
