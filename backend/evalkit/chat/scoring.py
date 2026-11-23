"""Matching rules and metrics for the scripted chat evaluation (fixed before the first run; docs/CHAT.md §8).

What is compared: for every case, the slots *after the last turn* against the slots the case started
from. Only changes made by the chat count; values the turn policy filled in as documented defaults
(``source == "default"``: the 50 km radius, the hiking fallback) are excluded on both sides.

Per slot type:

- enums (``activity``, ``difficulty``, ``distance_band``): exact match;
- ``place``: the same catalogue locality (name and district) after both the expected name and the
  extracted name went through the same place resolver;
- numbers, absolute-or-relative tolerance, whichever is larger: ``duration_hours`` ±25% or ±0.5 h,
  ``distance_cap_km`` ±10% or ±1 km, ``travel_radius_km`` ±25% or ±5 km;
- tags (``tags_preferred``, ``tags_avoided``): set precision / recall over the tags that changed.

Counting for a scalar slot with expected change ``e`` (None = should stay unchanged), predicted change
``p`` and an optional acceptable value ``o`` (an answer the case author considers defensible): p == e ->
TP; p is None, e set -> FN; p set, e None -> FP unless p matches o; both set but different -> FN, plus
FP unless p matches o. Tags: TP = |P ∩ E|, FP = |P − E − O|, FN = |E − P|. Precision, recall and F1 are
micro-averaged per slot type and overall.

Clarifying question: expected ``clarify`` (None or the slot) against the status after the last turn.
*Decision accuracy* scores asked vs not asked; *slot accuracy* also requires the right slot.
Intent: a case's ``intent_mentions`` groups must all occur (case-insensitive substring, any
alternative) in the final intent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Optional

ENUM_SLOTS = ("activity", "difficulty", "distance_band")
NUMBER_SLOTS = {"duration_hours": (0.25, 0.5), "distance_cap_km": (0.10, 1.0), "travel_radius_km": (0.25, 5.0)}
PLACE_SLOT = "place"
TAG_SLOTS = ("tags_preferred", "tags_avoided")
SCALAR_SLOTS = (*ENUM_SLOTS, *NUMBER_SLOTS, PLACE_SLOT)
ALL_SLOTS = (*SCALAR_SLOTS, *TAG_SLOTS)
SLOT_TYPE = {**{s: "enum" for s in ENUM_SLOTS}, **{s: "number" for s in NUMBER_SLOTS}, PLACE_SLOT: "place",
             **{s: "tags" for s in TAG_SLOTS}}


def value_matches(slot: str, predicted: Any, expected: Any) -> bool:
    if predicted is None or expected is None:
        return predicted is None and expected is None
    if slot in NUMBER_SLOTS:
        rel, absolute = NUMBER_SLOTS[slot]
        return abs(float(predicted) - float(expected)) <= max(absolute, rel * abs(float(expected)))
    return predicted == expected


def score_case(case: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Counts for one case.

    ``case``: the scripted case (``expect``, ``optional``, ``clarify``, ``intent_mentions``) with place
    names already normalised to the resolver's ``"name|district"`` keys (see ``run.py``).
    ``outcome``: ``changes`` (slot -> predicted new value, chat-sourced only, places as keys),
    ``clarify`` (None or slot) and ``intent``.
    """
    expect, optional = case.get("expect", {}), case.get("optional", {}) or {}
    changes = outcome.get("changes", {})
    counts: dict[str, dict[str, int]] = {s: {"tp": 0, "fp": 0, "fn": 0} for s in ALL_SLOTS}
    errors: list[dict[str, Any]] = []
    for slot in SCALAR_SLOTS:
        e, p, o = expect.get(slot), changes.get(slot), optional.get(slot)
        c = counts[slot]
        if p is None and e is None:
            continue
        if e is None:
            if not value_matches(slot, p, o):
                c["fp"] += 1
                errors.append({"slot": slot, "kind": "fp", "predicted": p, "expected": None})
            continue
        if p is None:
            c["fn"] += 1
            errors.append({"slot": slot, "kind": "fn", "predicted": None, "expected": e})
            continue
        if value_matches(slot, p, e):
            c["tp"] += 1
            continue
        c["fn"] += 1
        if not value_matches(slot, p, o):
            c["fp"] += 1
        errors.append({"slot": slot, "kind": "wrong", "predicted": p, "expected": e})
    for slot in TAG_SLOTS:
        exp_set, pred = set(expect.get(slot, [])), set(changes.get(slot, []))
        opt_set = set(optional.get(slot, []))
        c = counts[slot]
        c["tp"] += len(pred & exp_set)
        fp, fn = sorted(pred - exp_set - opt_set), sorted(exp_set - pred)
        c["fp"] += len(fp)
        c["fn"] += len(fn)
        if fp:
            errors.append({"slot": slot, "kind": "fp", "predicted": fp, "expected": sorted(exp_set)})
        if fn:
            errors.append({"slot": slot, "kind": "fn", "predicted": sorted(pred), "expected": fn})

    exp_clarify, got_clarify = case.get("clarify"), outcome.get("clarify")
    intent = (outcome.get("intent") or "").casefold()
    mentions = case.get("intent_mentions") or []
    intent_hit = all(any(alt.casefold() in intent for alt in group) for group in mentions) if mentions else None
    return {
        "counts": counts,
        "errors": errors,
        "clarify": {"expected": exp_clarify, "predicted": got_clarify,
                    "decision_ok": (exp_clarify is None) == (got_clarify is None),
                    "slot_ok": exp_clarify == got_clarify},
        "intent_hit": intent_hit,
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, Optional[float]]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": _r(p), "recall": _r(r), "f1": _r(f)}


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 4)


def aggregate(scored: Iterable[Mapping[str, Any]], groups: Optional[Iterable[str]] = None) -> dict[str, Any]:
    scored = list(scored)
    groups = list(groups) if groups is not None else [None] * len(scored)
    per_slot: dict[str, dict[str, int]] = {s: {"tp": 0, "fp": 0, "fn": 0} for s in ALL_SLOTS}
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total = {"tp": 0, "fp": 0, "fn": 0}
    dec_ok = slot_ok = 0
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}  # positive = a question was asked
    intent_n = intent_hits = 0
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "tp": 0, "fp": 0, "fn": 0, "clarify_ok": 0})
    for sc, group in zip(scored, groups, strict=True):
        for slot, c in sc["counts"].items():
            for k in ("tp", "fp", "fn"):
                per_slot[slot][k] += c[k]
                per_type[SLOT_TYPE[slot]][k] += c[k]
                total[k] += c[k]
                if group is not None:
                    by_group[group][k] += c[k]
        cl = sc["clarify"]
        dec_ok += cl["decision_ok"]
        slot_ok += cl["slot_ok"]
        asked, should = cl["predicted"] is not None, cl["expected"] is not None
        confusion["tp" if asked and should else "fp" if asked else "fn" if should else "tn"] += 1
        if sc["intent_hit"] is not None:
            intent_n += 1
            intent_hits += bool(sc["intent_hit"])
        if group is not None:
            by_group[group]["cases"] += 1
            by_group[group]["clarify_ok"] += cl["slot_ok"]
    n = len(scored)
    return {
        "cases": n,
        "per_slot": {s: _prf(**c) for s, c in per_slot.items()},
        "per_type": {t: _prf(**c) for t, c in sorted(per_type.items())},
        "overall": _prf(**total),
        "clarify": {"decision_accuracy": _r(dec_ok / n) if n else None,
                    "slot_accuracy": _r(slot_ok / n) if n else None, "confusion": confusion},
        "intent_mentions": {"cases": intent_n, "hit_rate": _r(intent_hits / intent_n) if intent_n else None},
        "by_group": {g: {**v, **{k: x for k, x in _prf(v["tp"], v["fp"], v["fn"]).items()
                                if k in ("precision", "recall")}} for g, v in sorted(by_group.items())},
    }


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)
