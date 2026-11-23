"""Failure analysis of the served default on the test split (``make failures``).

    cd backend && python -m evalkit.failures     # -> eval/results/_failures.json, then make report

Takes the ``N_WORST`` test personas with the lowest NDCG@10 of ``recsys_serve`` (``eval/results``), replays
each request through the pipeline (query embeddings from the frozen cache; ``LLM_CACHE_ONLY``, so no provider
call and no credentials) and checks that the replay returns exactly the stored ranking. For each persona it
records what the system saw, what it returned (grades and the judge's reasons) and where the pipeline lost
the best judged routes, stage by stage:

- ``filter``: the route violates one of the request's hard constraints (it cannot be relevant then: such
  pairs are graded 0 by rule, so this never happens for a judged-relevant route unless the request's slots
  differ from the persona's own limits);
- ``needs``: it passes the hard filter but the stated-need stage (iteration 2, systems with ``needs``
  configured) left it out of the eligible set;
- ``retrieval``: eligible, but not in the fused top 50 (BM25 and dense ranks among the eligible set say which
  generator missed it);
- ``ranker``: a candidate the learned ranker put below rank 10;
- ``mmr``: in the ranker's top 10, pushed below 10 by MMR;
- ``exploration``: in MMR's top 10, pushed to rank 11 by the exploration pick;
- ``shown``: in the returned top 10.

The diagnosis of each failure is written by hand in docs/FAILURES.md; ``make report`` renders the data
blocks there from this file.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Optional

from app.llm import run_sync
from app.recsys import get_catalog
from app.recsys.constraints import haversine_km
from app.recsys.systems.pipelines import SERVED_SYSTEM
from evalkit import paths
from evalkit.evaluate import dumps, load_run, make_system, stored_systems
from evalkit.llm import make_client
from evalkit.personas import load_personas
from evalkit.pool import POOL_DEPTH, contributor_runs
from evalkit.qrels import Qrels

SYSTEM = SERVED_SYSTEM
N_WORST = 10
N_BEST = 5  # best judged routes traced per persona (highest grade first)
OUT = "_failures.json"
SCHEMA = "evalkit.failures/v1"
# other systems' NDCG@10 on the same request, for context
COMPARE = ("recsys_v1", "recsys_v1_nocap", "v4_rrf", "v2_bm25", "v3_dense", "v1_filters")
STAGES = ("filter", "needs", "retrieval", "ranker", "mmr", "exploration", "shown")
LATENT_KEYS = (
    "party",
    "fitness",
    "distance_band_km",
    "max_distance_km",
    "max_ascent_m",
    "technical_max",
    "desires",
    "dislikes",
    "shape",
    "surface",
    "transport",
    "multi_day",
    "days",
)


def worst_personas(doc: dict[str, Any], split_of: dict[str, str], n: int = N_WORST, split: str = "test") -> list[str]:
    """The ``n`` personas of ``split`` with the lowest NDCG@10 (ties by persona id); undefined NDCG is skipped."""
    rows = [
        (v["ndcg_at_10"], pid)
        for pid, v in doc["per_persona"].items()
        if split_of[pid] == split and v["ndcg_at_10"] is not None
    ]
    return [pid for _, pid in sorted(rows)[:n]]


def _int(x: float) -> Optional[int]:
    return None if x is None or math.isnan(float(x)) else int(x)


def _facts(route, home: tuple[float, float]) -> dict[str, Any]:
    prof = route.features.get("profile", {})
    return {
        "title": route.title,
        "length_km": round(route.length_km, 1),
        "ascent_m": route.ascent_m,
        "loop": route.is_loop,
        "difficulty": route.difficulty,
        "max_grade": route.max_grade,
        "paved_share": prof.get("paved_share"),
        "home_km": round(float(haversine_km(home[0], home[1], route.start_lat, route.start_lon)), 1),
        "locality": route.locality,
        "tags": list(route.tags),
    }


def _lost_at(violations: list[str], fused_rank, ltr_rank, mmr_after, final_rank, by_needs: bool = False) -> str:
    if violations:
        return "filter"
    if final_rank is not None and final_rank <= 10:
        return "shown"
    if by_needs:
        return "needs"
    if fused_rank is None or fused_rank > 50 or ltr_rank is None:
        return "retrieval"
    if ltr_rank > 10:
        return "ranker"
    if mmr_after is not None and mmr_after > 10:
        return "mmr"
    return "exploration"


def _client():
    return make_client("failures", embed_provider="vertex", rank_provider="openrouter", cache_only=True)


def analyse(client=None, *, system: str = SYSTEM, split: str = "test", n: int = N_WORST) -> dict[str, Any]:
    """The analysis for ``system``'s ``n`` worst personas of ``split`` (``make failures``: the served system on
    test; ``--split dev`` is the development-split version, which is what an improvement may be tuned on)."""
    catalog = get_catalog()
    personas = {p.persona_id: p for p in load_personas()}
    split_of = {pid: p.split for pid, p in personas.items()}
    results = json.loads((paths.RESULTS_DIR / f"{system}.json").read_text(encoding="utf-8"))
    run, _ = load_run(system)
    qrels = Qrels.load()
    worst = worst_personas(results, split_of, n, split)

    # who put each route in a top 10 (the pool): stored systems and the pool-only contributors
    pooled_by: dict[tuple[str, int], list[str]] = {}
    for name in stored_systems():
        r, _ = load_run(name)
        for pid in worst:
            for rid in r[pid]["ids"][:POOL_DEPTH]:
                pooled_by.setdefault((pid, rid), []).append(name)
    for name, r in contributor_runs(catalog, [personas[p] for p in worst], POOL_DEPTH).items():
        for pid, ids in r.items():
            for rid in ids[:POOL_DEPTH]:
                pooled_by.setdefault((pid, int(rid)), []).append(name)

    compare = {}
    for name in COMPARE:
        p = paths.RESULTS_DIR / f"{name}.json"
        if p.exists():
            compare[name] = json.loads(p.read_text(encoding="utf-8"))["per_persona"]

    pipe = make_system(system, catalog, llm=client or _client())
    out: list[dict[str, Any]] = []
    for pid in worst:
        p = personas[pid]
        q = p.to_query()
        rec = pipe.recommend(q, 50)
        if rec.route_ids != run[pid]["ids"]:
            raise RuntimeError(f"{pid}: replay differs from eval/runs/{system}.jsonl; rerun make eval first")
        ret = run_sync(pipe._retrieve(q))  # stage arrays over the whole eligible set
        pos_in_elig = {int(i): j for j, i in enumerate(ret.elig)}
        hard_ok = {int(i) for i in ret.fr.indices}  # before the stated-need stage
        hc = p.constraints()
        judged = {rid: rec_ for (pp, rid), rec_ in qrels.records.items() if pp == pid}

        def stage_ranks(rid: int, rec=rec, ret=ret, pos_in_elig=pos_in_elig, hc=hc, hard_ok=hard_ok) -> dict[str, Any]:
            route = catalog.get(rid)
            j = pos_in_elig.get(catalog.index_of(rid))
            c = rec.contributions(rid)
            final = rec.route_ids.index(rid) + 1 if rid in rec.route_ids else None
            fused = None if j is None else _int(ret.stage["fused_rank"][j])
            ltr_rank = c.get("ltr", {}).get("rank")
            mmr_after = c.get("mmr", {}).get("rank_after")
            viol = hc.violations(route)
            by_needs = j is None and catalog.index_of(rid) in hard_ok and ret.needs_levels is not None
            return {
                "violations": viol,
                "eligible": j is not None,
                "bm25_rank": None if j is None else _int(ret.stage["bm25_rank"][j]),
                "bm25_score": None if j is None else round(float(ret.stage["bm25"][j]), 3),
                "dense_rank": None if j is None else _int(ret.stage["dense_rank"][j]),
                "fused_rank": fused,
                "ltr_rank": ltr_rank,
                "ltr_top": [
                    {"name": f["name"], "value": f["value"], "contribution": f["contribution"]}
                    for f in c.get("ltr", {}).get("features", [])[:3]
                ],
                "mmr_before": c.get("mmr", {}).get("rank_before"),
                "mmr_after": mmr_after,
                "explore": "explore" in c,
                "final_rank": final,
                "lost_at": _lost_at(viol, fused, ltr_rank, mmr_after, final, by_needs),
            }

        def row(rid: int, judged=judged, q=q, pid=pid, stage_ranks=stage_ranks) -> dict[str, Any]:
            j = judged.get(rid)
            return {
                "route_id": rid,
                "grade": None if j is None else j["grade"],
                "reason": None if j is None else j["reason"],
                "facts": _facts(catalog.get(rid), q.home),
                "pooled_by": sorted(pooled_by.get((pid, rid), [])),
                **stage_ranks(rid),
            }

        top = [row(rid) for rid in rec.route_ids[:10]]
        relevant = sorted((rid for rid, j in judged.items() if j["grade"] >= 2), key=lambda r: (-judged[r]["grade"], r))
        rel_rows = {rid: stage_ranks(rid) for rid in relevant}
        best = [row(rid) for rid in [r for r in relevant if rel_rows[r]["lost_at"] != "shown"][:N_BEST]]
        lost = Counter(r["lost_at"] for r in rel_rows.values())
        lost3 = Counter(rel_rows[r]["lost_at"] for r in relevant if judged[r]["grade"] == 3)
        m = results["per_persona"][pid]
        out.append(
            {
                "persona_id": pid,
                "ndcg_at_10": m["ndcg_at_10"],
                "mrr_at_10": m["mrr_at_10"],
                "recall_at_50": m["recall_at_50"],
                "intent": p.intent,
                "seen": {
                    "category": q.category,
                    "home_town": p.profile.get("home_town"),
                    "max_travel_km": q.max_travel_km,
                    "max_distance_km": q.max_distance_km,
                    "questionnaire": p.questionnaire,
                },
                "latent": {k: p.latent.get(k) for k in LATENT_KEYS},
                "n_eligible": int(len(ret.elig)),
                "judged_grades": dict(sorted(Counter(str(j["grade"]) for j in judged.values()).items())),
                "relevant_lost_at": {s: lost.get(s, 0) for s in STAGES},
                "grade3_lost_at": {s: lost3.get(s, 0) for s in STAGES},
                "explore": next((t.info for t in rec.trace if t.stage == "explore"), None),
                "top10": top,
                "best_missed": best,
                "other_systems_ndcg_at_10": {n: pp[pid]["ndcg_at_10"] for n, pp in compare.items() if pid in pp},
            }
        )
    # context for the diagnoses: how many personas of the groups that fail share the ranker's training data
    import re

    from app.recsys.features import CONCEPTS

    train_re = re.compile(CONCEPTS["train"][0], re.I)
    allp = list(personas.values())
    context = {
        "personas_by_split_and_style": dict(sorted(Counter(f"{p.split}/{p.latent['style']}" for p in allp).items())),
        "stroller_personas_by_split": dict(
            sorted(Counter(p.split for p in allp if p.latent["party"].get("stroller")).items())
        ),
        "train_users_by_split": dict(
            sorted(Counter(p.split for p in allp if p.latent["transport"] == "train").items())
        ),
        "train_users_intent_mentions_train_by_split": dict(
            sorted(
                Counter(
                    p.split for p in allp if p.latent["transport"] == "train" and train_re.search(p.intent or "")
                ).items()
            )
        ),
    }
    totals = Counter()
    totals3 = Counter()
    for f in out:
        totals.update(f["relevant_lost_at"])
        totals3.update(f["grade3_lost_at"])
    return {
        "schema": SCHEMA,
        "system": system,
        "split": split,
        "n_worst": n,
        "n_best_traced": N_BEST,
        "qrels_sha256": qrels.sha256(),
        "results_note": results["note"],
        "context": context,
        "relevant_lost_at_total": {s: totals.get(s, 0) for s in STAGES},
        "grade3_lost_at_total": {s: totals3.get(s, 0) for s in STAGES},
        "personas": out,
    }


def replay(check: bool) -> tuple[list[str], list[str]]:
    """For ``make replay-check`` / ``make freeze``: recompute the analysis from the frozen cache and compare it
    with the committed file. Returns (touched cache keys, problems)."""
    if not (paths.RESULTS_DIR / OUT).exists():
        return [], []
    client = _client()
    doc = analyse(client)
    probs = []
    if check and dumps(doc) != (paths.RESULTS_DIR / OUT).read_text(encoding="utf-8"):
        probs.append(f"failure analysis differs: {OUT} (run make failures)")
    return client.touched_keys(), probs


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--split", default="test", choices=("test", "dev"))
    ap.add_argument("-n", type=int, default=N_WORST)
    ap.add_argument("--out", default=None, help=f"results file name (default {OUT} for the served system on test)")
    args = ap.parse_args(argv)
    doc = analyse(system=args.system, split=args.split, n=args.n)
    out = args.out or (OUT if (args.system, args.split, args.n) == (SYSTEM, "test", N_WORST) else None)
    if out is None:
        out = f"_failures_{args.system}_{args.split}.json"
    (paths.RESULTS_DIR / out).write_text(dumps(doc), encoding="utf-8")
    print(f"wrote {paths.RESULTS_DIR / out}" + ("; now run make report" if out == OUT else ""))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- docs/FAILURES.md blocks
def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _f3(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _facts_cell(fa: dict[str, Any]) -> str:
    paved = "?" if fa["paved_share"] is None else f"{100 * fa['paved_share']:.0f}%"
    shape = "loop" if fa["loop"] else "one-way"
    return f"{fa['length_km']} km, {fa['ascent_m']} m up, {shape}, {paved} paved, {fa['home_km']} km from home"


def _stages_cell(r: dict[str, Any]) -> str:
    def v(x):
        return "–" if x is None else str(x)

    mmr = f"{v(r['mmr_before'])}→{v(r['mmr_after'])}" if r["mmr_before"] is not None else "–"
    return (
        f"BM25 {v(r['bm25_rank'])} · dense {v(r['dense_rank'])} · fused {v(r['fused_rank'])} · "
        f"LTR {v(r['ltr_rank'])} · MMR {mmr}"
    )


def _latent_line(lat: dict[str, Any]) -> str:
    party = lat["party"] or {}
    bits = [f"party {party.get('kind')}"]
    if party.get("children_ages"):
        bits.append(f"children aged {', '.join(str(a) for a in party['children_ages'])}")
    bits += [
        f"fitness {lat['fitness']}",
        f"comfortable {lat['distance_band_km'][0]}–{lat['distance_band_km'][1]} km",
        f"ascent ≤ {lat['max_ascent_m']} m",
        f"technical ≤ {lat['technical_max']}",
    ]
    if lat.get("multi_day"):
        bits.append(f"{lat['days']} days")
    wishes = ", ".join(f"{k} {v}" for k, v in sorted(lat["desires"].items(), key=lambda kv: -kv[1]))
    bits.append(f"wishes {wishes}")
    if lat["dislikes"]:
        bits.append(f"dislikes {', '.join(lat['dislikes'])}")
    bits += [f"shape {lat['shape']}", f"surface {lat['surface']}", f"transport {lat['transport']}"]
    return "; ".join(bits)


def _persona_block(f: dict[str, Any]) -> str:
    s = f["seen"]
    q = s["questionnaire"]
    lost = ", ".join(f"{k} {v}" for k, v in f["relevant_lost_at"].items() if v)
    lost3 = ", ".join(f"{k} {v}" for k, v in f["grade3_lost_at"].items() if v) or "none judged"
    others = ", ".join(f"`{n}` {_f3(v)}" for n, v in f["other_systems_ndcg_at_10"].items())
    grades = ", ".join(f"{k}: {v}" for k, v in f["judged_grades"].items())
    L = [
        f"NDCG@10 **{_f3(f['ndcg_at_10'])}**, MRR@10 {_f3(f['mrr_at_10'])}, Recall@50 {_f3(f['recall_at_50'])}; "
        f"{f['n_eligible']} routes pass the hard filters; judged pairs by grade: {grades}.",
        "",
        f'- **Intent (what the user typed):** "{_esc(f["intent"])}"',
        f"- **Slots the system saw:** tab `{s['category']}`, home {s['home_town']}, radius {s['max_travel_km']} km, "
        f"distance cap {s['max_distance_km']} km; questionnaire {q.get('fitness')}, {', '.join(q.get('type') or [])}.",
        f"- **Latent profile (only the judge sees it):** {_esc(_latent_line(f['latent']))}.",
        f"- **Known relevant routes (grade ≥ 2) by where they ended:** {lost}; grade 3 only: {lost3}.",
        f"- **Same request, other systems (NDCG@10):** {others}.",
        "",
        "Returned top 10 (stage ranks: BM25 and dense among the eligible set, fused = RRF, LTR = learned ranker, "
        "MMR before→after):",
        "",
        "| # | Route | Grade | Judge's reason | Facts | Stage ranks |",
        "|---|---|---|---|---|---|",
    ]
    for r in f["top10"]:
        tag = " (exploration pick)" if r["explore"] else ""
        L.append(
            f"| {r['final_rank']} | {_esc(r['facts']['title'])} `{r['route_id']}`{tag} | {r['grade']} | "
            f"{_esc(r['reason'])} | {_facts_cell(r['facts'])} | {_stages_cell(r)} |"
        )
    L += [
        "",
        "Best judged routes not shown, and where they were lost:",
        "",
        "| Grade | Route | Judge's reason | Facts | Stage ranks → final | Lost at | Largest LTR contributions "
        "| Pooled by |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in f["best_missed"]:
        contrib = ", ".join(f"`{c['name']}` {c['contribution']:+.2f}" for c in r["ltr_top"]) or "–"
        final = "–" if r["final_rank"] is None else str(r["final_rank"])
        L.append(
            f"| {r['grade']} | {_esc(r['facts']['title'])} `{r['route_id']}` | {_esc(r['reason'])} | "
            f"{_facts_cell(r['facts'])} | {_stages_cell(r)} → {final} | **{r['lost_at']}** | {contrib} | "
            f"{', '.join(r['pooled_by']) or '–'} |"
        )
    return "\n".join(L)


def _summary_block(doc: dict[str, Any]) -> str:
    cols = [
        c
        for c in ("recsys_v1", "v4_rrf", "v2_bm25")
        if all(c in f["other_systems_ndcg_at_10"] for f in doc["personas"])
    ]
    L = [
        f"*{_esc(doc['results_note'])}*",
        "",
        f"The {doc['n_worst']} test personas with the lowest NDCG@10 of `{doc['system']}` (served default), from "
        f'`eval/results/{OUT}` (`make failures`). Relevant = judged grade ≥ 2; "lost at" counts every such route '
        "of the persona by the stage after which it was no longer in the top 10.",
        "",
        "| Persona | Intent | NDCG@10 | " + " | ".join(f"`{c}`" for c in cols) + " | Relevant routes: lost at |",
        "|---|---|---|" + "---|" * len(cols) + "---|",
    ]
    for f in doc["personas"]:
        lost = ", ".join(f"{k} {v}" for k, v in f["relevant_lost_at"].items() if v)
        L.append(
            f"| [{f['persona_id']}](#{f['persona_id'].lower()}) | {_esc(f['intent'])} | {_f3(f['ndcg_at_10'])} | "
            + " | ".join(_f3(f["other_systems_ndcg_at_10"][c]) for c in cols)
            + f" | {lost} |"
        )
    tot = doc["relevant_lost_at_total"]
    t3 = doc["grade3_lost_at_total"]
    L += [
        "",
        "All relevant routes of the ten personas, by where they were lost: "
        + ", ".join(f"{k} {v}" for k, v in tot.items())
        + f" (total {sum(tot.values())}). Grade 3 only: "
        + ", ".join(f"{k} {v}" for k, v in t3.items())
        + f" (total {sum(t3.values())}).",
    ]
    return "\n".join(L)


def _classes_block(prose: str, doc: dict[str, Any]) -> str:
    """Counts of the hand-assigned failure classes (``**Class:** X`` under each ``### Pxxx`` heading)."""
    import re

    by_class: dict[str, list[str]] = {}
    cur = None
    for line in prose.splitlines():
        m = re.match(r"^### (P\d+)\b", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^\*\*Class:\*\* ([A-Z])\b", line)
        if m and cur:
            by_class.setdefault(m.group(1), []).append(cur)
            cur = None
    want = [f["persona_id"] for f in doc["personas"]]
    got = sorted(p for ps in by_class.values() for p in ps)
    if sorted(want) != got:
        raise ValueError(f"FAILURES.md: every persona needs exactly one '**Class:** X' line; have {got}, need {want}")
    return "\n".join(f"- Class {c}: {len(ps)} of {len(want)} ({', '.join(ps)})" for c, ps in sorted(by_class.items()))


def render_blocks(prose: str) -> dict[str, str]:
    """Generated blocks of docs/FAILURES.md from ``eval/results/_failures.json`` (and the class lines in
    ``prose``, the document's hand-written text)."""
    doc = json.loads((paths.RESULTS_DIR / OUT).read_text(encoding="utf-8"))
    c = doc["context"]
    ctx = "\n".join(
        [
            "- Personas by split and activity style: "
            + ", ".join(f"{k} {v}" for k, v in c["personas_by_split_and_style"].items())
            + ".",
            "- Personas pushing a stroller: "
            + ", ".join(f"{k} {v}" for k, v in c["stroller_personas_by_split"].items())
            + ".",
            "- Personas who travel by train (a hard wish the judge checks): "
            + ", ".join(f"{k} {v}" for k, v in c["train_users_by_split"].items())
            + "; of these, intent mentions the train: "
            + ", ".join(f"{k} {v}" for k, v in c["train_users_intent_mentions_train_by_split"].items())
            + ".",
        ]
    )
    blocks = {
        "failures-summary": _summary_block(doc),
        "failures-context": ctx,
        "failures-classes": _classes_block(prose, doc),
    }
    for f in doc["personas"]:
        blocks[f"failure-{f['persona_id'].lower()}"] = _persona_block(f)
    return blocks
