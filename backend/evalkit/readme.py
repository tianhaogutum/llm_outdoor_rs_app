"""Generated blocks of README.md and docs/REVIEW.md (rendered by ``make report``).

Every number in those two files comes from here, read from ``eval/results/*.json``; the prose around the blocks
carries no metric values. ``tests/evalkit/test_docs_fresh.py`` fails if a committed block differs from what this
module renders from the committed results.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from evalkit import blocks, paths
from evalkit.metrics import paired_bootstrap

SERVED = "recsys_serve_v2"  # since D28
PREV = "recsys_serve"  # the served default from D25 to D28
PHASE3 = "recsys_v1"
CHAIN = ["random", "v0_rules", "v1_filters", "v2_bm25", "v3_dense", "v4_rrf", "v5_rank_api", "v6_ltr", "v7_mmr"]
LABELS = {
    "random": "random",
    "v0_rules": "course project's rule scorer",
    "v1_filters": "+ hard filters",
    "v2_bm25": "+ BM25 (replaces the rule scorer)",
    "v3_dense": "dense retrieval instead of BM25",
    "v4_rrf": "+ RRF fusion of BM25 and dense",
    "v5_rank_api": "+ ranker API on the fused top 50",
    "v6_ltr": "+ learned ranker (LambdaMART, API score as a feature)",
    "v7_mmr": "+ MMR diversity",
    PHASE3: "Phase 3 default: learned ranker without the API, MMR",
    PREV: "served default D25-D28: ranker without the cap feature, feedback, exploration slot",
    SERVED: "**served default** (D28): + stated needs as constraints in front of retrieval",
}
# what each row is compared with in the "Δ vs" column
DELTA_BASE = {
    "v0_rules": "random",
    "v1_filters": "v0_rules",
    "v2_bm25": "v1_filters",
    "v3_dense": "v2_bm25",
    "v4_rrf": "v3_dense",
    "v5_rank_api": "v4_rrf",
    "v6_ltr": "v5_rank_api",
    "v7_mmr": "v6_ltr",
    PHASE3: "v7_mmr",
    PREV: PHASE3,
    SERVED: PREV,
}


# --------------------------------------------------------------------------- helpers
def _load(name: str) -> Optional[dict[str, Any]]:
    p = paths.RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _sys(name: str) -> dict[str, Any]:
    d = _load(f"{name}.json")
    if d is None:
        raise FileNotFoundError(f"eval/results/{name}.json (run make eval SYSTEM={name})")
    return d


def _ci(m: dict[str, Any], pct: bool = False) -> str:
    from evalkit.report import _ci as ci

    return ci(m, pct=pct)


def _m(name: str, key: str = "ndcg_at_10", split: str = "test", pct: bool = False) -> str:
    return _ci(_sys(name)["metrics"][split][key], pct=pct)


def _mean(name: str, key: str = "ndcg_at_10", split: str = "test") -> float:
    return _sys(name)["metrics"][split][key]["mean"]


def _diff(a: str, b: str, split: str = "test", key: str = "ndcg_at_10") -> str:
    """Paired bootstrap of b − a over the split's personas, formatted ``+0.000 [+0.000, +0.000]``."""
    from evalkit.report import _split_of

    da, db = _sys(a), _sys(b)
    pids = sorted(p for p in da["per_persona"] if _split_of(da, p) == split)
    r = paired_bootstrap(
        [da["per_persona"][p][key] for p in pids],
        [db["per_persona"][p][key] for p in pids],
        n_boot=da["protocol"]["bootstrap"]["n"],
        seed=da["protocol"]["bootstrap"]["seed"],
    )
    return "n/a" if r["diff"] is None else f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def _holdout(name: str) -> dict[str, Any]:
    p = paths.RESULTS_DIR / "holdout" / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} (run make eval SYSTEM={name} HOLDOUT=1)")
    return json.loads(p.read_text(encoding="utf-8"))


def _hm(name: str, key: str = "ndcg_at_10", pct: bool = False) -> str:
    return _ci(_holdout(name)["metrics"]["holdout"][key], pct=pct)


def _hdiff(a: str, b: str) -> str:
    da, db = _holdout(a), _holdout(b)
    pids = sorted(da["per_persona"])
    r = paired_bootstrap(
        [da["per_persona"][p]["ndcg_at_10"] for p in pids],
        [db["per_persona"][p]["ndcg_at_10"] for p in pids],
        n_boot=da["protocol"]["bootstrap"]["n"],
        seed=da["protocol"]["bootstrap"]["seed"],
    )
    return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def _cf(name: str) -> str:
    p = paths.RESULTS_DIR / "crossfit" / f"{name}.json"
    return _ci(json.loads(p.read_text(encoding="utf-8"))["metrics"]["dev"]["ndcg_at_10"])


def _caption() -> str:
    from evalkit.report import _caption as cap

    return cap(_sys(SERVED))


def _n(split: str = "test") -> int:
    return _sys(SERVED)["data"]["split_sizes"][split]


def _sim() -> dict[str, Any]:
    d = _load("_simulation.json")
    if d is None:
        raise FileNotFoundError("eval/results/_simulation.json (run make simulate)")
    return d


def _sim_pair(a: str, b: str, which: str = "rounds_2_10") -> str:
    from evalkit.report import _pair_cell

    p = next(p for p in _sim()["paired"] if p["a"] == a and p["b"] == b)
    if which == "rounds_2_10":
        return _pair_cell(p[which])
    return _pair_cell(p["per_round"][0] if which == "round_1" else p["per_round"][-1])


def _kappa_line() -> str:
    k = _load("_kappa.json") or {}
    o = k.get("owner", {})
    if o.get("status") == "owner labels missing" or not o:
        own = "owner labels: **missing** (the 150 blind pairs are in `eval/owner_labels/`)"
    else:
        own = f"owner vs judge on {o['labelled']} pairs: QWK {o['qwk']:.3f}"
    m = k.get("llm_vs_llm")
    llm = (
        f"; interim, **not a human check**: a second LLM judge (`{', '.join(m['second_models'])}`) agrees with the "
        f"production judge at QWK {m['qwk']:.3f} (exact {m['exact']:.2f}, within one grade {m['within_one']:.2f}) "
        f"on the same {m['n']} pairs"
        if m
        else ""
    )
    return own + llm


def _pilot(run: str) -> dict[str, Any]:
    p = _load("_judge_pilot.json")
    return next(c for c in p["candidates"] if c["run"] == run)


def _production_judge() -> str:
    return _sys(SERVED)["judge"]["config"]["name"]


# --------------------------------------------------------------------------- README blocks
def headline() -> str:
    sim = _sim()["metrics"]
    j = _sys(SERVED)["judge"]
    q = _load("_kappa.json")["llm_vs_llm"]["qwk"]
    val = j["owner_validation"]
    judge_status = (
        f"that agrees with the owner's labels at QWK {val['owner_qwk']:.2f}"
        if val.get("validated")
        else "that is **not yet validated against the owner's labels**, and the only agreement figure so far is "
        f"LLM-vs-LLM (QWK {q:.2f})"
    )
    hn = _holdout(SERVED)["data"]["split_sizes"]["holdout"]
    return (
        f"On {hn} holdout personas generated after the decision rule was written down and used for nothing else, the "
        f"served default `{SERVED}` reaches **NDCG@10 {_hm(SERVED)}**, against {_hm(PREV)} for the previous default "
        f"`{PREV}` (paired difference {_hdiff(PREV, SERVED)}; [D28](docs/DECISIONS.md)). On the {_n()} test "
        f"personas it reaches {_m(SERVED)}, against {_m('v0_rules')} for the course project's rule scorer and "
        f"{_m('random')} for random order; the best single-request system without stated needs, `{PHASE3}`, "
        f"reaches {_m(PHASE3)} there, and the served default differs from it by {_diff(PHASE3, SERVED)} (paired). "
        f"In a ten-round feedback simulation the served default keeps NDCG@10 at "
        f"{_ci(sim[SERVED]['ndcg_rounds_2_10'])} over rounds 2-10 ({_ci(sim[PREV]['ndcg_rounds_2_10'])} for "
        f"`{PREV}`), while the Phase 3 pipeline without feedback falls to {_ci(sim[PHASE3]['ndcg_rounds_2_10'])} "
        "because it keeps showing routes the user has already done.\n\n"
        f"**Caveat on every number in this README:** the users are simulated ({_sys(SERVED)['data']['n_personas']} "
        f"personas plus {hn} holdout personas, all with LLM-written requests), relevance is graded by an LLM judge "
        f"(`{', '.join(sorted(j['models']))}`, rubric v1) {judge_status}. 95% bootstrap intervals over personas "
        "in brackets."
    )


def data_line() -> str:
    d = _sys(SERVED)["data"]
    q = _load("_qrels.json")
    h = _holdout(SERVED)["data"]
    return (
        f"Catalogue: {d['catalogue_routes']:,} routes. Simulated users: {d['n_personas']} personas "
        f"({d['split_sizes']['dev']} dev for every choice and all training, {d['split_sizes']['test']} test, read "
        f"for reporting and, in iteration 2, to find a failure pattern) plus {h['n_personas']} holdout personas "
        "drawn after the iteration-3 rule was registered (D28). Judgements: "
        f"{q['pairs'] + h['qrels_pairs']:,} (persona, route) pairs ({h['qrels_pairs']:,} of them on the holdout, in "
        f"their own file), {100 * q['rule_share']:.0f}% of the main pairs graded 0 by rule (hard-constraint "
        "violations), the rest by the LLM judge."
    )


def results_table() -> str:
    rows = [*CHAIN, PHASE3, PREV, SERVED]
    L = [
        _caption(),
        "",
        "| System | What it adds | NDCG@10 | Δ NDCG@10 vs | MRR@10 | Recall@50 | Violations@10 | Cost/request |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in rows:
        m = _sys(n)["metrics"]["test"]
        base = DELTA_BASE.get(n)
        delta = f"{_diff(base, n)} (`{base}`)" if base else "–"
        L.append(
            f"| `{n}` | {LABELS[n]} | {_ci(m['ndcg_at_10'])} | {delta} | {_ci(m['mrr_at_10'])} | "
            f"{_ci(m['recall_at_50'])} | {_ci(m['violation_at_10'], pct=True)} | {m['cost_usd']['mean']:.6f} USD |"
        )
    L += [
        "",
        f"Test split, {_n()} personas; Δ = paired bootstrap of the per-persona difference. Every top 10 is fully "
        "judged. All systems, the dev split and every other comparison: [RESULTS.md](docs/RESULTS.md).",
    ]
    return "\n".join(L)


SIM_ROWS = [
    ("v0_rules", "course project's rule scorer with its own feedback"),
    ("v1_filters", "hard filters + rule scorer, same feedback"),
    (PHASE3, "Phase 3 default, feedback off (control)"),
    ("recsys_v1_fb", "Phase 3 default + feedback loop"),
    ("recsys_serve_noexplore", f"`{PREV}` without the exploration slot"),
    (PREV, "served default D25-D28"),
    (f"{SERVED}_noexplore", f"`{SERVED}` without the exploration slot"),
    (SERVED, "**served default** (D28): `" + PREV + "` + stated needs as constraints"),
]


def simulation_table() -> str:
    d = _sim()
    met = d["metrics"]
    L = [
        f"*Simulated users ({d['n_personas']} test personas) **and** a simulated behaviour model written for this "
        "project (EVAL.md section 10); LLM-judged relevance, judge not yet validated against the owner's labels.*",
        "",
        "| System | | NDCG@10, round 1 | NDCG@10, mean of rounds 2-10 | NDCG@10, round 10 "
        "| Cumulative coverage after 10 rounds |",
        "|---|---|---|---|---|---|",
    ]
    for n, lab in SIM_ROWS:
        if n not in met:
            continue
        r = met[n]["rounds"]
        L.append(
            f"| `{n}` | {lab} | {_ci(r[0]['ndcg_at_10'])} | {_ci(met[n]['ndcg_rounds_2_10'])} | "
            f"{_ci(r[-1]['ndcg_at_10'])} | {100 * r[-1]['cumulative_coverage']:.1f}% |"
        )
    L += [
        "",
        f"Rounds 2-10, paired: stated-need stage in the loop (`{SERVED}` − `{PREV}`) {_sim_pair(PREV, SERVED)}, "
        f"round 1 {_sim_pair(PREV, SERVED, 'round_1')}, round 10 {_sim_pair(PREV, SERVED, 'round_10')}; "
        f"exploration slot (`{SERVED}` − `{SERVED}_noexplore`) {_sim_pair(f'{SERVED}_noexplore', SERVED)}; "
        f"feedback (`{SERVED}` − `{PHASE3}`) {_sim_pair(PHASE3, SERVED)} NDCG@10.",
    ]
    return "\n".join(L)


def failures_summary() -> str:
    from evalkit import failures

    doc = _load(failures.HOLDOUT_OUT)
    prose = blocks.strip_blocks((paths.DOCS_DIR / "FAILURES.md").read_text(encoding="utf-8"))
    tot = doc["relevant_lost_at_total"]
    return (
        f"Known-relevant routes (grade ≥ 2) of the {doc['n_worst']} worst {doc['split']} queries of `{doc['system']}`, "
        "by where "
        "they dropped out of the top 10: "
        + ", ".join(f"{k} {v}" for k, v in tot.items() if k != "shown")
        + f"; shown {tot['shown']} (of {sum(tot.values())}).\n\n"
        + failures._classes_block(prose, doc)
    )


def cost_latency() -> str:
    lat = _load("_latency.json")
    L = [
        f"Warm benchmark (`make latency`: {lat['n']} sequential requests after {lat['warmup']} warm-up requests, "
        f"LLM cache off, live calls, one {lat['machine']} laptop); cost per request = mean nominal upstream cost over "
        "the test personas.",
        "",
        "| System | Ranker API | p50 ms | p95 ms | max ms | Largest stage (p95, upstream) | Local compute p95 ms "
        "| Cost/request | p95 budget |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, v in lat["systems"].items():
        up = max(v["upstream_ms_p95"].items(), key=lambda kv: kv[1])
        cost = _mean(n, "cost_usd") if (paths.RESULTS_DIR / f"{n}.json").exists() else v["cost_usd_per_request"]
        L.append(
            f"| `{n}` | {'yes' if v['uses_rank_api'] else 'no'} | {v['wall_ms']['p50']:.0f} | "
            f"{v['wall_ms']['p95']:.0f} | "
            f"{v['wall_ms']['max']:.0f} | {up[0]} {up[1]:.0f} ms | {v['local_compute_ms']['p95']:.1f} | "
            f"{cost:.6f} USD | {v['budget_ms_p95']:.0f} ms ({'within' if v['within_budget'] else '**over**'}) |"
        )
    return "\n".join(L)


def limits() -> str:
    ltr = _load("_ltr.json")
    full = ltr["variants"]["full"]
    imp = ltr["models"]["v1_noapi"]["importance"][0]
    ref = _load("_reference.json")["metrics"]["test"]["ndcg_at_10"]
    pj = _pilot(_production_judge())
    order = next(s for s in _load("_judge_pilot.json")["batch_order_sensitivity"] if s["config"] == pj["run"])
    return "\n".join(
        [
            f"- Test split: {_n()} personas; dev: {_n('dev')}; holdout: "
            f"{_holdout(SERVED)['data']['split_sizes']['holdout']} (D28).",
            f"- Judge pilot ({pj['n']} pairs): QWK {pj['qwk_vs_reference']:.2f} against a stronger LLM reference, "
            f"mean grade {pj['mean_grade_minus_reference']:+.2f} above it; QWK {order['qwk_between_orders']:.2f} "
            "between two batch orders of the same pairs.",
            f"- Judge validation: {_kappa_line()}.",
            f"- Learned ranker (`v6_ltr`, judged candidates): NDCG@10 {full['train_ndcg_at_10']:.3f} in-sample on dev, "
            f"{_ci(full['cv_ndcg_at_10'])} in grouped cross-validation, {_ci(full['test_ndcg_at_10'])} on test. "
            f"System level, `{PHASE3}`: dev (in-sample) {_m(PHASE3, split='dev')}, test {_m(PHASE3)}.",
            f"- Largest feature of the Phase 3 ranker: `{imp['feature']}` "
            f"({100 * imp['gain_share']:.0f}% of the gain), "
            f"which encodes how the persona generator draws distance caps. Dropping it costs "
            f"{_diff(PHASE3, 'recsys_v1_nocap')} on test; with the cap slot left empty, `{PREV}`'s ranker is "
            f"{_diff('recsys_v1_nocapslot', 'recsys_serve_nocap')} ahead of `{PHASE3}`'s.",
            f"- For comparison, the pool-only latent-profile matcher (reads the hidden profile, not a system): "
            f"{_ci(ref)} on test; `{PHASE3}`: {_m(PHASE3)}.",
        ]
    )


NEEDS = ("recsys_serve_needs", "recsys_serve_needs_promote", "recsys_serve_needs_demote")
NEEDS_TEXT = {
    "recsys_serve_needs": "filter in front of retrieval",
    "recsys_serve_needs_promote": "filter, strict matches promoted",
    "recsys_serve_needs_demote": "post-ranker demotion",
}


def iteration2() -> str:
    """Iteration 2 (D27): the pre-registered dev decision, the post-hoc cross-fitted check, the single test run."""
    nd = _load("_needs_dev.json")
    cf = nd["dev_crossfit"]["paired_vs_baseline"]
    fa = _load("_failures.json")
    s = _sys(PREV)
    L = [
        "| Variant (on top of the served default) | Dev, pre-registered rule: Δ NDCG@10 | Accepted | "
        "Dev, cross-fitted ranker (post hoc): Δ | Test, reported once: Δ | Test violations@10 |",
        "|---|---|---|---|---|---|",
    ]
    for n in NEEDS:
        dev = _diff(PREV, n, split="dev")
        lo = float(dev.split("[")[1].split(",")[0])
        c = cf[n + "@crossfit"]["ndcg_at_10"]
        L.append(
            f"| `{n}` ({NEEDS_TEXT[n]}) | {dev} | {'yes' if lo > 0 else 'no'} | "
            f"{c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | {_diff(PREV, n)} | "
            f"{_m(n, 'violation_at_10', pct=True)} |"
        )
    best = "recsys_serve_needs_promote"
    b = _sys(best)
    rows = []
    for f in fa["personas"]:
        pid = f["persona_id"]
        old, new = s["per_persona"][pid]["ndcg_at_10"], b["per_persona"][pid]["ndcg_at_10"]
        rows.append(f"{pid} {old:.2f}→{new:.2f}")
    L += [
        "",
        f"Iteration 2 left the served default unchanged: `{PREV}` {_m(PREV)} on test, {_m(PREV, split='dev')} on dev "
        "(in-sample ranker). Its ten worst test queries (FAILURES.md, test analysis) under "
        f"`{best}` (NDCG@10 before→after): " + ", ".join(rows) + ".",
    ]
    return "\n".join(L)


def iteration3() -> str:
    """Iteration 3 (D28): the cross-fitted dev gate, the holdout comparison and the decision (``_holdout.json``)."""
    from evalkit import holdout as H

    d = _load(H.OUT)
    ps = d["personas"]
    sim = d["simultaneous"]
    styles = ", ".join(f"{k.split('/')[-1]} {v}" for k, v in ps["activity_style"].items())
    party = ", ".join(f"{k} {v}" for k, v in sorted(ps["party"].items(), key=lambda kv: -kv[1]))
    L = [
        f"*Holdout: {ps['n']} simulated users (seed {ps['seed']}; {styles}; parties {party}), every top 10 judged by "
        "the same LLM judge, which is not yet validated against the owner's labels.*",
        "",
        "| System | Dev, cross-fitted: Δ NDCG@10 (gate) | Holdout NDCG@10 | Holdout Δ vs `recsys_serve` (95% CI) | "
        f"Simultaneous lower bound (c = {sim['c']:.2f}) | Holdout violations@10 | Eligible |",
        "|---|---|---|---|---|---|---|",
        f"| `{H.BASELINE}` | – | {_ci(d['systems'][H.BASELINE]['ndcg_at_10'])} | – | – | "
        f"{_ci(d['systems'][H.BASELINE]['violation_at_10'], pct=True)} | – |",
    ]
    for v in H.CANDIDATES:
        g, pr, ver = d["gate_dev_crossfit"][v], d["paired_95"][v], d["verdict"][v]
        lb = "not tested" if ver["simultaneous_lower"] is None else f"{ver['simultaneous_lower']:+.3f}"
        win = " **(chosen)**" if d["decision"].get("winner") == v else ""
        L.append(
            f"| `{v}` ({NEEDS_TEXT[v]}){win} | {g['diff']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}] | "
            f"{_ci(d['systems'][v]['ndcg_at_10'])} | {pr['diff']:+.3f} [{pr['lo']:+.3f}, {pr['hi']:+.3f}] | {lb} | "
            f"{_ci(d['systems'][v]['violation_at_10'], pct=True)} | {'yes' if ver['eligible'] else 'no'} |"
        )
    dec = d["decision"]
    out = (
        f"Decision by the rule registered before the holdout existed: `{dec['winner']}` is served as `{dec['serve']}`."
        if dec["outcome"] == "switch"
        else f"Decision by the rule registered before the holdout existed: null result, `{dec['serve']}` stays."
    )
    L += [
        "",
        out + " Simultaneous bound: one-sided 95% for the three variants together (max-T over a joint paired "
        f"bootstrap, {sim['n_boot']:,} resamples); the 95% CIs beside it are per variant and decide nothing.",
    ]
    return "\n".join(L)


def readme_blocks() -> dict[str, str]:
    return {
        "readme-headline": headline(),
        "readme-data": data_line(),
        "readme-results": results_table(),
        "readme-simulation": simulation_table(),
        "readme-failures": failures_summary(),
        "readme-iteration2": iteration2(),
        "readme-iteration3": iteration3(),
        "readme-limits": limits(),
        "readme-cost-latency": cost_latency(),
    }


# --------------------------------------------------------------------------- docs/REVIEW.md blocks
def review_pooling() -> str:
    q = _load("_qrels.json")
    s = _sys(SERVED)
    rel = q["relevant_per_persona"]
    return "\n".join(
        [
            "- Pool: top 10 of every registered system plus the pool-only contributors "
            + ", ".join(f"`{c}`" for c in s["protocol"]["pool_only_contributors"])
            + ".",
            f"- {q['pairs']:,} judged pairs over {q['personas']} personas; {q['by_source']['rule']:,} "
            f"({100 * q['rule_share']:.1f}%) graded 0 by rule, {q['by_source']['llm']:,} by the LLM judge.",
            f"- Known relevant (grade ≥ 2) routes per persona: median {rel['median']}, min {rel['min']}, max "
            f"{rel['max']}; personas without any: {rel['personas_without_relevant']}.",
            f"- Reference, the latent matcher that supplies much of the relevant set: NDCG@10 "
            f"{_ci(_load('_reference.json')['metrics']['test']['ndcg_at_10'])} on test (`_reference.json`).",
            f"- Holdout (D28): its own qrels file, {_holdout(SERVED)['data']['qrels_pairs']:,} judged pairs; pool = "
            "the top 10 of the four D28 candidates plus the same two contributors.",
        ]
    )


def review_agreement() -> str:
    pj = _pilot(_production_judge())
    order = next(s for s in _load("_judge_pilot.json")["batch_order_sensitivity"] if s["config"] == pj["run"])
    return "\n".join(
        [
            f"- {_kappa_line()}.",
            f"- Judge pilot: production judge `{pj['run']}` vs an LLM reference on {pj['n']} pairs: QWK "
            f"{pj['qwk_vs_reference']:.3f}, exact {pj['exact_agreement']:.2f}, mean grade "
            f"{pj['mean_grade_minus_reference']:+.2f} relative to the reference; between two batch orders QWK "
            f"{order['qwk_between_orders']:.3f}.",
        ]
    )


def review_rrf() -> str:
    cg = _load("_cg_dev.json")
    ks = ", ".join(
        f"k {k}: {_ci(v['recall_at_50'])}" for k, v in sorted(cg["rrf_k"].items(), key=lambda kv: int(kv[0]))
    )
    return "\n".join(
        [
            f"- Test NDCG@10 / Recall@50: BM25 `v2_bm25` {_m('v2_bm25')} / {_m('v2_bm25', 'recall_at_50')}; dense "
            f"`v3_dense` {_m('v3_dense')} / {_m('v3_dense', 'recall_at_50')}; RRF `v4_rrf` {_m('v4_rrf')} / "
            f"{_m('v4_rrf', 'recall_at_50')}.",
            f"- RRF − BM25 (test, paired): NDCG@10 {_diff('v2_bm25', 'v4_rrf')}, Recall@50 "
            f"{_diff('v2_bm25', 'v4_rrf', key='recall_at_50')}. RRF − dense: NDCG@10 {_diff('v3_dense', 'v4_rrf')}.",
            f"- RRF constant on dev (Recall@50): {ks}.",
            f"- Cosine band of one dev persona's eligible set: "
            f"min {cg['quantisation']['example_cosine_range'][0]:.3f}, "
            f"max {cg['quantisation']['example_cosine_range'][2]:.3f} (BM25 scores are unbounded and query-dependent).",
        ]
    )


def review_stages() -> str:
    L = [
        "| Step | ΔNDCG@10 (test, paired) | ΔNDCG@10 (dev, paired) |",
        "|---|---|---|",
    ]
    for n in [*CHAIN[1:], PHASE3, PREV, SERVED]:
        b = DELTA_BASE[n]
        L.append(f"| `{n}` − `{b}` | {_diff(b, n)} | {_diff(b, n, split='dev')} |")
    L += [
        "",
        f"Also: `v6_ltr_noapi` − `v4_rrf` (learned ranker without the API) {_diff('v4_rrf', 'v6_ltr_noapi')}; "
        f"`v6_ltr` − `v6_ltr_noapi` (the API score as a feature) {_diff('v6_ltr_noapi', 'v6_ltr')}; questionnaire "
        f"only → + intent (`recsys_v1_qonly` → `{PHASE3}`) {_diff('recsys_v1_qonly', PHASE3)}; + chat instead of "
        f"the raw intent (`recsys_v1_chat` − `{PHASE3}`) {_diff(PHASE3, 'recsys_v1_chat')}. Dev rows of the learned "
        "ranker are in-sample.",
    ]
    return "\n".join(L)


def review_overfit() -> str:
    ltr = _load("_ltr.json")
    v = ltr["variants"]
    imp = ltr["models"]["v1_noapi"]["importance"][0]
    return "\n".join(
        [
            f"- `v6_ltr` on judged candidates: in-sample {v['full']['train_ndcg_at_10']:.3f}, grouped CV "
            f"{_ci(v['full']['cv_ndcg_at_10'])}, test {_ci(v['full']['test_ndcg_at_10'])}.",
            f"- System level: `v6_ltr` dev (in-sample) {_m('v6_ltr', split='dev')} vs test {_m('v6_ltr')}; "
            f"`{PHASE3}` dev {_m(PHASE3, split='dev')} vs test {_m(PHASE3)}.",
            f"- `{imp['feature']}`: {100 * imp['gain_share']:.1f}% of the gain of the Phase 3 ranker. Without it "
            f"(`recsys_v1_nocap` − `{PHASE3}`): test {_diff(PHASE3, 'recsys_v1_nocap')}, dev "
            f"{_diff(PHASE3, 'recsys_v1_nocap', split='dev')}. Without the whole suspicious group "
            f"(`recsys_v1_nosusp`): test {_diff(PHASE3, 'recsys_v1_nosusp')}.",
            f"- Cap slot left empty: `{PHASE3}` loses {_diff(PHASE3, 'recsys_v1_nocapslot')}, `{PREV}` "
            f"{_diff(PREV, 'recsys_serve_nocap')}; `{PREV}` minus Phase 3 without the cap "
            f"{_diff('recsys_v1_nocapslot', 'recsys_serve_nocap')}.",
            f"- Cross-fitted dev (each dev persona ranked by a ranker copy that never saw it, EVAL.md section 13): "
            f"`{PREV}` in-sample {_m(PREV, split='dev')}, cross-fitted {_cf(PREV)}, test {_m(PREV)}; `{PHASE3}` "
            f"in-sample {_m(PHASE3, split='dev')}, cross-fitted {_cf(PHASE3)}, test {_m(PHASE3)}.",
            "- Feedback parameters: chosen on dev (`slots_only`, D21; rerun on the D29-corrected labels the dev "
            f"rule's maximum is `{(_load('_sim_tuning.json') or {}).get('best')}`, within noise, D29); on the test "
            f"simulation the light full loop beats the dev choice by "
            f"{_sim_pair('recsys_v1_fb', 'recsys_v1_fb_light')} (rounds 2-10).",
        ]
    )


def review_cost() -> str:
    lat = _load("_latency.json")
    s = lat["systems"].get(SERVED)
    stages = ", ".join(
        f"{k} {v['local'] + v['upstream']:.1f} ms"
        for k, v in sorted(s["per_stage_ms_p50"].items(), key=lambda kv: -(kv[1]["local"] + kv[1]["upstream"]))
    )
    return cost_latency() + (
        f"\n\nMedian time per stage of `{SERVED}` (local + upstream): {stages}. Ranker API on (`v7_mmr`) − "
        f"`{PHASE3}`: NDCG@10 {_diff(PHASE3, 'v7_mmr')} on test."
    )


def review_blocks() -> dict[str, str]:
    return {
        "review-pooling": review_pooling(),
        "review-agreement": review_agreement(),
        "review-rrf": review_rrf(),
        "review-stages": review_stages(),
        "review-overfit": review_overfit(),
        "review-cost": review_cost(),
        "review-failures": failures_summary(),
        "review-iteration2": iteration2(),
        "review-iteration3": iteration3(),
    }
