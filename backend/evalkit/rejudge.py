"""Re-judge the pairs the v1 judge card could have misled (docs/DECISIONS.md D29).

    cd backend && python -m evalkit.rejudge estimate   # affected pairs per file and the judging cost; writes nothing
    cd backend && python -m evalkit.rejudge run        # re-judge them with card/v2, then write the summary
    cd backend && python -m evalkit.rejudge control    # the control: unaffected pairs re-judged the same way
    cd backend && python -m evalkit.rejudge summary    # recompute eval/results/_rejudge.json from the qrels files

``card/v1`` (``evalkit.cards``) printed the persona's travel radius, comfortable distance band and distance cap
rounded to whole km and the distance from home to whole km. A judged pair is *affected* when a comparison the
judge makes on the printed numbers comes out differently from the same comparison on the exact numbers:

- ``cap``: the printed route length is above the printed cap, although the route is within the exact cap (the
  rule lets it through, so the judge saw it and could call it "too long");
- ``band``: the route's position relative to the comfortable band (below / inside / above) differs between the
  printed and the exact numbers (the rubric's grade 3 needs a length inside or very close to the band);
- ``radius``: the printed distance from home is above the printed radius although the start is within it.

Ascent (whole metres against a tolerance in tens of metres) and days (integers) were printed exactly. Only
LLM-graded records with the v1 card are candidates; rule-graded pairs never reached the judge.

Every affected pair of every qrels file (pooled, simulation, holdout) is re-judged with ``card/v2``, batched per
persona and file over the affected pairs only (production judge, seed 0, batches of at most 8). The new record
replaces the old one and keeps it under ``"previous"`` (and keeps the ``sample`` marker of the learned ranker's
random sample); ``make replay-check`` replays both. The learned ranker's pinned training labels
(``ltr_train_v1.jsonl``) and the experiments' pinned label files are not rewritten: they record what was used.
The summary counts how many of their labels the re-judging changed.

Control. Re-judging in new, smaller batches changes some grades by itself (the judge is not perfectly stable
across batch compositions, D9). ``control`` re-judges a seeded random sample of *unaffected* LLM-graded pairs of
the pooled qrels, drawn from the personas that have affected pairs, exactly the same way (card/v2, batched per
persona over the sampled pairs). Its change rate is what re-judging alone does; the difference to the affected
pairs' change rate is what the card fix does. The control's judgements go to ``rejudge_control.jsonl`` and never
replace a grade.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from app.recsys import Catalog, RouteRecord, get_catalog, haversine_km
from evalkit import paths
from evalkit.cards import CARD_V1, CARD_V2, km
from evalkit.personas import Persona, load_personas
from evalkit.qrels import Qrels

OUT = "_rejudge.json"
TAG = "rejudge:card-v2"
CONTROL_FILE = paths.QRELS_DIR / "rejudge_control.jsonl"
CONTROL_N = 200
CONTROL_SEED = 20261129
REASONS = ("cap", "band", "radius")
# pinned grade snapshots taken from the pooled qrels (persona_id, route_id, grade)
PINNED = {
    "ltr_train_v1.jsonl": "learned ranker's training labels (D15)",
    "mmr_sweep_labels.jsonl": "MMR sweep labels (D16)",
    "rank_pilot_labels.jsonl": "ranker-API pilot labels (D14)",
}


def qrels_files() -> dict[str, Any]:
    from evalkit.evaluate import HOLDOUT
    from evalkit.simulate import sim_qrels_file

    return {"main": paths.QRELS_FILE, "simulation": sim_qrels_file(), "holdout": HOLDOUT.qrels_file}


def all_personas() -> dict[str, Persona]:
    from evalkit.evaluate import HOLDOUT

    ps = {p.persona_id: p for p in load_personas()}
    if HOLDOUT.personas_file.exists():
        ps.update({p.persona_id: p for p in load_personas(HOLDOUT.personas_file)})
    return ps


def _pos(v: float, lo: float, hi: float) -> int:
    return -1 if v < lo else (1 if v > hi else 0)


def _shown_len(r: RouteRecord) -> float:
    return float(f"{r.length_km:.1f}")  # the route card prints the length to 0.1 km in both versions


def affected(p: Persona, r: RouteRecord) -> list[str]:
    """Why the v1 card could have misled the judge on this pair (empty: it could not)."""
    x = p.latent
    v1 = lambda v: float(km(v, CARD_V1))  # noqa: E731
    out = []
    shown = _shown_len(r)
    if r.length_km <= x["max_distance_km"] and shown > v1(x["max_distance_km"]):
        out.append("cap")
    lo, hi = x["distance_band_km"]
    if _pos(shown, v1(lo), v1(hi)) != _pos(r.length_km, lo, hi):
        out.append("band")
    d = haversine_km(p.profile["home"][0], p.profile["home"][1], r.start_lat, r.start_lon)
    if d <= x["max_travel_km"] and float(f"{d:.0f}") > v1(x["max_travel_km"]):
        out.append("radius")
    return out


def expected_direction(p: Persona, r: RouteRecord, reasons: list[str]) -> int:
    """+1 if the exact numbers make the route look better than v1 did, -1 if worse, 0 if unclear."""
    if "cap" in reasons or "radius" in reasons:
        return 1
    x = p.latent
    lo, hi = x["distance_band_km"]
    was = _pos(_shown_len(r), float(km(lo, CARD_V1)), float(km(hi, CARD_V1)))
    now = _pos(r.length_km, lo, hi)
    if now == 0 and was != 0:
        return 1
    if was == 0 and now != 0:
        return -1
    return 0


def v2_residual(p: Persona, r: RouteRecord) -> bool:
    """With the v2 card, does the printed band position still differ from the exact one (route within 0.05 km of
    an edge)? Counted, not re-judged."""
    lo, hi = p.latent["distance_band_km"]
    return _pos(_shown_len(r), float(km(lo, CARD_V2)), float(km(hi, CARD_V2))) != _pos(r.length_km, lo, hi)


def candidates(q: Qrels, ps: dict[str, Persona], catalog: Catalog) -> dict[tuple[str, int], list[str]]:
    """Affected pairs still carrying a v1 LLM judgement."""
    from evalkit.judge import card_of

    out = {}
    for (pid, rid), rec in sorted(q.records.items()):
        if rec["source"] != "llm" or card_of(rec) != CARD_V1:
            continue
        why = affected(ps[pid], catalog.get(rid))
        if why:
            out[(pid, rid)] = why
    return out


def estimate() -> dict[str, Any]:
    from evalkit.judge import make_batches, production_judge

    catalog, ps = get_catalog(), all_personas()
    bs = production_judge().batch_size
    rep: dict[str, Any] = {}
    for name, f in qrels_files().items():
        c = candidates(Qrels.load(f), ps, catalog)
        per_p: dict[str, list[int]] = {}
        for pid, rid in c:
            per_p.setdefault(pid, []).append(rid)
        calls = sum(len(make_batches(pid, ids, bs)) for pid, ids in per_p.items())
        rep[name] = {
            "pairs": len(c),
            "personas": len(per_p),
            "calls": calls,
            "by_reason": dict(sorted(Counter(r for why in c.values() for r in why).items())),
            # same per-call / per-pair rates as the simulation's estimate (evalkit.simulate)
            "est_usd": round(calls * 0.0025 + len(c) * 0.0009, 2),
        }
    rep["total_est_usd"] = round(sum(v["est_usd"] for v in rep.values()), 2)
    return rep


def run() -> dict[str, Any]:
    from evalkit.judge import judge_pairs, production_judge

    catalog, ps = get_catalog(), all_personas()
    cfg = production_judge()
    report: dict[str, Any] = {}
    for name, f in qrels_files().items():
        q = Qrels.load(f)
        c = candidates(q, ps, catalog)
        if not c:
            report[name] = {"pairs": 0}
            continue
        out = judge_pairs(cfg, catalog, ps, list(c), tag=TAG, card=CARD_V2)
        for new in out.records:
            k = (new["persona_id"], int(new["route_id"]))
            old = q.records[k]
            prev = {kk: v for kk, v in old.items() if kk not in ("persona_id", "route_id")}
            rec = {**new, "previous": prev}
            if "sample" in old:
                rec["sample"] = old["sample"]
            q.records[k] = rec
        q.save(f)
        if name == "simulation":
            f.with_suffix(".txt").unlink(missing_ok=True)  # as evalkit.simulate: the TREC copy is only for the pool
        report[name] = {"pairs": len(c), "calls": out.calls, "nominal_cost_usd": round(out.nominal_cost_usd, 4)}
    return report


def control_pairs(q: Qrels, ps: dict[str, Persona], catalog: Catalog) -> list[tuple[str, int]]:
    """Seeded sample of unaffected v1 LLM pairs of the pooled qrels, from personas with re-judged pairs."""
    import numpy as np

    from evalkit.judge import card_of

    rej = {pid for (pid, _), r in q.records.items() if r.get("previous")}
    pool = [
        k
        for k, r in sorted(q.records.items())
        if k[0] in rej and r["source"] == "llm" and card_of(r) == CARD_V1 and not affected(ps[k[0]], catalog.get(k[1]))
    ]
    idx = np.random.default_rng(CONTROL_SEED).choice(len(pool), size=min(CONTROL_N, len(pool)), replace=False)
    return sorted(pool[i] for i in idx)


def control() -> dict[str, Any]:
    from evalkit.judge import judge_pairs, production_judge

    catalog, ps = get_catalog(), all_personas()
    pairs = control_pairs(Qrels.load(paths.QRELS_FILE), ps, catalog)
    out = judge_pairs(production_judge(), catalog, ps, pairs, tag="rejudge:control", card=CARD_V2)
    Qrels(out.records).save(CONTROL_FILE)
    CONTROL_FILE.with_suffix(".txt").unlink(missing_ok=True)
    return {"pairs": len(pairs), "calls": out.calls, "nominal_cost_usd": round(out.nominal_cost_usd, 4)}


def _transitions(rows: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "changed": sum(1 for a, b in rows if a != b),
        "up": sum(1 for a, b in rows if b > a),
        "down": sum(1 for a, b in rows if b < a),
        "mean_old": round(sum(a for a, _ in rows) / len(rows), 4) if rows else None,
        "mean_new": round(sum(b for _, b in rows) / len(rows), 4) if rows else None,
        "matrix_old_to_new": {f"{a}->{b}": n for (a, b), n in sorted(Counter(rows).items())},
    }


def summary() -> dict[str, Any]:
    """Deterministic from the qrels files (old grades are kept under ``previous``)."""
    from evalkit.judge import card_of

    catalog, ps = get_catalog(), all_personas()
    doc: dict[str, Any] = {
        "schema": "evalkit.rejudge/v1",
        "decision": "D29",
        "card_old": CARD_V1,
        "card_new": CARD_V2,
        "rule": "affected = LLM-graded with card/v1 and the printed cap, band position or radius disagrees with the "
        "exact one (evalkit.rejudge.affected)",
        "files": {},
    }
    new_main: dict[tuple[str, int], int] = {}
    for name, f in qrels_files().items():
        q = Qrels.load(f)
        rows: dict[str, list[tuple[int, int]]] = {"all": [], **{r: [] for r in REASONS}}
        direction: Counter = Counter()
        llm = left = resid = 0
        for (pid, rid), rec in sorted(q.records.items()):
            prev = rec.get("previous")
            # only the lines that existed at the re-judging (v1, or re-judged): judgements added later with card/v2
            # do not change this summary
            if rec["source"] != "llm" or (card_of(rec) != CARD_V1 and not prev):
                continue
            llm += 1
            p, r = ps[pid], catalog.get(rid)
            if card_of(rec) == CARD_V1 and affected(p, r):
                left += 1  # affected but not re-judged (must be 0 after ``run``)
            if not prev:
                continue
            if v2_residual(p, r):
                resid += 1
            why = affected(p, r)
            pair = (int(prev["grade"]), int(rec["grade"]))
            rows["all"].append(pair)
            for w in why:
                rows[w].append(pair)
            exp = expected_direction(p, r, why)
            got = (pair[1] > pair[0]) - (pair[1] < pair[0])
            sign = "+" if exp > 0 else "-" if exp < 0 else "?"
            direction[f"expected {sign}: " + ("up" if got > 0 else "down" if got < 0 else "same")] += 1
            if name == "main":
                new_main[(pid, rid)] = pair[1]
        doc["files"][name] = {
            "file": f.name,
            "llm_pairs": llm,
            "affected_not_rejudged": left,
            "rejudged_v2_band_residual_within_0.05km": resid,
            "rejudged": {k: _transitions(v) for k, v in rows.items() if v or k == "all"},
            "direction_vs_expected": dict(sorted(direction.items())),
        }
    main_q = Qrels.load(qrels_files()["main"])
    pinned = {}
    for fname, label in PINNED.items():
        f = paths.QRELS_DIR / fname
        if not f.exists():
            continue
        labels = {
            (d["persona_id"], int(d["route_id"])): int(d["grade"])
            for d in (json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        }
        hit = [(k, g) for k, g in labels.items() if k in new_main]
        pinned[fname] = {
            "label": label,
            "labels": len(labels),
            "rejudged_pairs": len(hit),
            "label_equals_old_grade": sum(
                1 for k, g in hit if g == int(main_q.records[k]["previous"]["grade"])
            ),
            "changed_by_rejudge": sum(1 for k, g in hit if g != new_main[k]),
            "changed_share": round(sum(1 for k, g in hit if g != new_main[k]) / len(labels), 5) if labels else None,
            "transitions": _transitions([(g, new_main[k]) for k, g in hit]),
        }
    doc["pinned_labels"] = pinned
    if CONTROL_FILE.exists():
        ctrl = Qrels.load(CONTROL_FILE)
        rows = [(int(main_q.records[k]["grade"]), int(r["grade"])) for k, r in sorted(ctrl.records.items())]
        doc["control"] = {
            "file": CONTROL_FILE.name,
            "sample": f"{CONTROL_N} unaffected LLM pairs of the pooled qrels from personas with re-judged pairs, "
            f"seed {CONTROL_SEED}; re-judged with card/v2 the same way; grades not used",
            **_transitions(rows),
        }
    tot = [doc["files"][n]["rejudged"]["all"] for n in doc["files"]]
    doc["total"] = {
        "rejudged": sum(t["pairs"] for t in tot),
        "changed": sum(t["changed"] for t in tot),
        "up": sum(t["up"] for t in tot),
        "down": sum(t["down"] for t in tot),
    }
    doc["est_cost_rule"] = "calls x 0.0025 + pairs x 0.0009 USD (as evalkit.simulate estimate)"
    return doc


def write_summary() -> dict[str, Any]:
    from evalkit.evaluate import dumps

    doc = summary()
    (paths.RESULTS_DIR / OUT).write_text(dumps(doc), encoding="utf-8")
    return doc


def replay(check: bool) -> tuple[list[str], list[str]]:
    """For ``make replay-check``: the summary is a function of the qrels files (their judge batches are replayed by
    ``evalkit.freeze`` and ``evalkit.holdout``); recompute it and compare."""
    keys: list[str] = []
    probs: list[str] = []
    if CONTROL_FILE.exists():
        from evalkit.judge import replay_problems

        keys, probs = replay_problems(
            get_catalog(), all_personas(), list(Qrels.load(CONTROL_FILE).records.values()), CONTROL_FILE.name
        )
    f = paths.RESULTS_DIR / OUT
    if not check or not f.exists():
        return keys, probs
    from evalkit.evaluate import dumps

    if dumps(summary()) != f.read_text(encoding="utf-8"):
        probs.append(f"{OUT} differs")
    return keys, probs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("estimate", "run", "control", "summary"))
    args = ap.parse_args(argv)
    if args.cmd == "estimate":
        print(json.dumps(estimate(), indent=1))
        return
    if args.cmd == "run":
        print(json.dumps(run(), indent=1))
    if args.cmd == "control":
        print(json.dumps(control(), indent=1))
    doc = write_summary()
    print(json.dumps({"total": doc["total"], **{n: d["rejudged"]["all"] for n, d in doc["files"].items()}}, indent=1))
    if "control" in doc:
        print("control:", json.dumps({k: v for k, v in doc["control"].items() if k != "sample"}))
    print(f"wrote {paths.RESULTS_DIR / OUT}")


if __name__ == "__main__":
    main()
