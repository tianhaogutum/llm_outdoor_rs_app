"""Render ``docs/RESULTS.md`` from ``eval/results/*.json`` only (``make report``). No number is typed by hand."""

from __future__ import annotations

import json
from typing import Any, Optional

from evalkit import paths
from evalkit.metrics import paired_bootstrap

# Phase 3 ablation chain: each row adds one component to the previous one (docs/RECSYS.md)
ABLATION = ["v0_rules", "v1_filters", "v2_bm25", "v3_dense", "v4_rrf", "v5_rank_api", "v6_ltr", "v7_mmr"]
ABLATION_LABELS = {
    "v0_rules": "baseline rule scorer",
    "v1_filters": "+ hard filters",
    "v2_bm25": "+ BM25 (replaces the rule scorer)",
    "v3_dense": "+ dense (instead of BM25)",
    "v4_rrf": "+ RRF (BM25 and dense fused)",
    "v5_rank_api": "+ ranker (API) on the fused top 50",
    "v6_ltr": "+ ranker (learned, with the API score as a feature)",
    "v7_mmr": "+ MMR",
}
# further paired comparisons shown below the ablation
EXTRA_PAIRS = [
    ("v4_rrf", "v6_ltr_noapi"),
    ("v6_ltr_noapi", "v6_ltr"),
    ("v5_rank_api", "v6_ltr"),
    ("v6_ltr_noapi", "recsys_v1"),
    ("v7_mmr", "recsys_v1"),
]
ORDER = ["random", *ABLATION]  # other systems follow alphabetically
COLUMNS = [
    ("ndcg_at_10", "NDCG@10"),
    ("mrr_at_10", "MRR@10"),
    ("recall_at_50", "Recall@50"),
    ("violation_at_10", "Violations@10"),
    ("coverage_at_10", "Coverage@10"),
    ("ild_at_10", "ILD@10"),
]


def _load(name: str) -> Optional[dict[str, Any]]:
    p = paths.RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _systems() -> list[dict[str, Any]]:
    docs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(paths.RESULTS_DIR.glob("*.json"))
        if not p.name.startswith("_")
    ]
    rank = {n: i for i, n in enumerate(ORDER)}
    return sorted(docs, key=lambda d: (rank.get(d["system"], len(ORDER)), d["system"]))


def _ci(m: dict[str, Any], pct: bool = False) -> str:
    if m.get("mean") is None:
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    if m.get("lo") is None:
        return f"{f(m['mean'])} (no CI)"
    return f"{f(m['mean'])} [{f(m['lo'])}, {f(m['hi'])}]"


def _caption(doc: dict[str, Any]) -> str:
    j = doc["judge"]
    models = ", ".join(sorted(j["models"])) or j["config"]["model"]
    val = j["owner_validation"]
    v = (
        f"validated against the owner's labels (QWK {val['owner_qwk']:.2f})"
        if val.get("validated")
        else "**not yet validated against the owner's labels**"
    )
    return (
        f"*Simulated users ({doc['data']['n_personas']} personas, LLM-written intents); LLM-judged relevance "
        f"(judge `{j['config']['name']}` = `{models}`, {j['rubric']}); the judge is {v}.*"
    )


def _table(docs: list[dict[str, Any]], split: str) -> list[str]:
    head = "| System | " + " | ".join(c for _, c in COLUMNS) + " | Cost/request | Judged@10 |"
    out = [head, "|" + "---|" * (len(COLUMNS) + 3)]
    for d in docs:
        m = d["metrics"][split]
        cells = [_ci(m[k], pct=k in ("violation_at_10", "coverage_at_10")) for k, _ in COLUMNS]
        cost = m["cost_usd"]["mean"]
        out.append(
            f"| `{d['system']}` | " + " | ".join(cells) + f" | {cost:.5f} USD | {100 * m['judged_share_at_10']:.0f}% |"
        )
    n = docs[0]["metrics"][split]
    out.append("")
    out.append(
        f"n = {n['n_personas']} personas; NDCG undefined (no judged relevant route) for "
        + ", ".join(f"`{d['system']}` {d['metrics'][split]['ndcg_undefined']}" for d in docs)
        + ". 95% percentile-bootstrap CIs over personas. Coverage@10 is a set statistic (distinct routes in all "
        "top-10 lists / catalogue size); it has no bootstrap CI and depends on the number of personas."
    )
    return out


def _paired_rows(by: dict[str, dict[str, Any]], pairs: list[tuple[str, str]], split: str) -> list[str]:
    out = [
        "| Comparison | ΔNDCG@10 [95% CI] | ΔMRR@10 | ΔRecall@50 | P(Δ ≤ 0) for NDCG | n |",
        "|---|---|---|---|---|---|",
    ]
    for an, bn in pairs:
        if an not in by or bn not in by:
            continue
        a, b = by[an], by[bn]
        pids = sorted(pid for pid in a["per_persona"] if split == "all" or _split_of(a, pid) == split)
        cells = []
        res_ndcg = None
        for key in ("ndcg_at_10", "mrr_at_10", "recall_at_50"):
            r = paired_bootstrap(
                [a["per_persona"][p][key] for p in pids],
                [b["per_persona"][p][key] for p in pids],
                n_boot=a["protocol"]["bootstrap"]["n"],
                seed=a["protocol"]["bootstrap"]["seed"],
            )
            res_ndcg = res_ndcg or r
            cells.append("n/a" if r["diff"] is None else f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
        out.append(
            f"| `{b['system']}` − `{a['system']}` | "
            + " | ".join(cells)
            + f" | {res_ndcg['p_le_0']:.3f} | {res_ndcg['n']} |"
        )
    return out if len(out) > 2 else []


def _paired(docs: list[dict[str, Any]], split: str) -> list[str]:
    """Baselines: v0_rules vs random. Phase 3: consecutive rows of the ablation chain, then EXTRA_PAIRS."""
    by = {d["system"]: d for d in docs}
    pairs = [("random", "v0_rules")]
    chain = [n for n in ABLATION if n in by]
    pairs += list(zip(chain, chain[1:], strict=False))
    pairs += EXTRA_PAIRS
    return _paired_rows(by, pairs, split)


_SPLITS: dict[str, str] = {}


def _split_of(doc: dict[str, Any], pid: str) -> str:
    if not _SPLITS:
        from evalkit.personas import load_personas

        _SPLITS.update({p.persona_id: p.split for p in load_personas()})
    return _SPLITS[pid]


def _m(d: dict[str, Any], split: str, key: str) -> str:
    return _ci(d["metrics"][split][key])


def _candidate_generation(docs: list[dict[str, Any]], cap: str) -> list[str]:
    by = {d["system"]: d for d in docs}
    gens = [("v2_bm25", "BM25F"), ("v3_dense", "dense (exact cosine)"), ("v4_rrf", "RRF(BM25, dense)")]
    if not any(n in by for n, _ in gens):
        return []
    L = [
        "## Candidate generation (Phase 3b)",
        "",
        cap,
        "",
        "Each generator over the eligible set, ranked by its own score (the `v2`-`v4` systems). Recall@50 is "
        "pool-relative: the share of a persona's known relevant routes in the generator's top 50.",
        "",
        "| Generator | System | Recall@50 (test) | NDCG@10 (test) | Recall@50 (dev) | NDCG@10 (dev) |",
        "|---|---|---|---|---|---|",
    ]
    for n, label in gens:
        if n in by:
            d = by[n]
            L.append(
                f"| {label} | `{n}` | {_m(d, 'test', 'recall_at_50')} | {_m(d, 'test', 'ndcg_at_10')} | "
                f"{_m(d, 'dev', 'recall_at_50')} | {_m(d, 'dev', 'ndcg_at_10')} |"
            )
    cg = _load("_cg_dev.json")
    if cg:
        L += [
            "",
            f"Dev-only tuning ({cg['n_personas']} dev personas, `eval/results/_cg_dev.json`; `python -m evalkit.cg tune`)."
            " Measured against the qrels at the time of tuning, without new judgements: Recall@50 and a *condensed* "
            "NDCG@10 (unjudged routes dropped before the cut). The pool contains plain BM25 of the intent "
            "(`pool_intent_lexical`), which favours the `plain` variant.",
            "",
            "| BM25 variant | Recall@50 (dev) | condensed NDCG@10 (dev) | ΔRecall@50 vs plain |",
            "|---|---|---|---|",
        ]
        for name, v in cg["bm25"].items():
            dv = v.get("recall_vs_plain")
            dcell = f"{dv['diff']:+.3f} [{dv['lo']:+.3f}, {dv['hi']:+.3f}]" if dv else "–"
            L.append(f"| {name} | {_ci(v['recall_at_50'])} | {_ci(v['condensed_ndcg_at_10'])} | {dcell} |")
        L += ["", "| Dense document text | Recall@50 (dev) | condensed NDCG@10 (dev) |", "|---|---|---|"]
        for name, v in cg["dense"].items():
            if name == "paired_recall":
                continue
            L.append(f"| `{name}` | {_ci(v['recall_at_50'])} | {_ci(v['condensed_ndcg_at_10'])} |")
        for k, v in cg["dense"].get("paired_recall", {}).items():
            L.append(f"\nPaired ΔRecall@50 ({k}): {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}].")
        L += ["", "| RRF k | Recall@50 (dev) | condensed NDCG@10 (dev) |", "|---|---|---|"]
        for k, v in sorted(cg["rrf_k"].items(), key=lambda kv: int(kv[0])):
            L.append(f"| {k} | {_ci(v['recall_at_50'])} | {_ci(v['condensed_ndcg_at_10'])} |")
        qz = cg.get("quantisation")
        if qz:
            L += [
                "",
                f"Committed document matrix: {qz['committed_precision']}, {qz['committed_file_bytes'] / 1e6:.1f} MB "
                f"(float32 would be {qz['float32_bytes'] / 1e6:.1f} MB). Rankings vs float32 on the dev queries:",
                "",
                "| Precision | top-10 overlap | identical top-10 order | identical top-50 order | max abs cosine diff |",
                "|---|---|---|---|---|",
            ]
            for prec in ("float16", "int8"):
                z = qz[prec]
                L.append(
                    f"| {prec} | {z['top10_overlap_mean']:.4f} | {100 * z['top10_identical_order_share']:.1f}% | "
                    f"{100 * z['top50_identical_order_share']:.1f}% | {z['max_abs_cosine_diff']:.2e} |"
                )
            lo, med, hi = qz["example_cosine_range"]
            L.append(
                f"\nCosines are compressed into a narrow band (one dev persona's eligible set: min {lo:.3f}, median "
                f"{med:.3f}, max {hi:.3f}), so differences of 1e-5 already swap near-ties."
            )
        ex = cg.get("exact_search")
        if ex:
            L.append(
                f"\nExact search over the whole catalogue ({ex['n_docs']} x {ex['dim']}, no filter): "
                f"{ex['ms_per_query_full_catalogue']:.3f} ms per query ({ex['ns_per_doc']:.0f} ns per document; "
                f"{ex['machine_note']})."
            )
    L.append("")
    return L


def render() -> str:
    docs = _systems()
    q = _load("_qrels.json")
    pilot = _load("_judge_pilot.json")
    kap = _load("_kappa.json")
    pers = _load("_personas.json")
    L = [
        "# Results",
        "",
        "Generated by `make report` from `eval/results/*.json`; do not edit by hand. Protocol and limitations: "
        "[EVAL.md](EVAL.md).",
        "",
    ]
    if not docs:
        L.append("No results yet: run `make eval SYSTEM=<name>`.")
        return "\n".join(L) + "\n"
    cap = _caption(docs[0])
    for split, title in (("test", "Test split (headline)"), ("dev", "Dev split")):
        L += [f"## {title}", "", cap, ""] + _table(docs, split) + [""]
        paired = _paired(docs, split)
        if paired:
            L += [f"Paired bootstrap of the per-persona difference ({split}):", "", cap, ""] + paired + [""]
    L += _candidate_generation(docs, cap)
    L += [
        "## Latency (locally measured, varies run to run)",
        "",
        cap,
        "",
        "| System | p50 ms (test) | p95 ms (test) | p50 ms (dev) | p95 ms (dev) | upstream p95 ms (test) |",
        "|---|---|---|---|---|---|",
    ]
    for d in docs:
        v = d["volatile"]["latency_ms_nominal"]
        L.append(
            f"| `{d['system']}` | {v['test']['p50']:.1f} | {v['test']['p95']:.1f} | {v['dev']['p50']:.1f} | "
            f"{v['dev']['p95']:.1f} | {d['metrics']['test']['upstream_latency_ms']['p95']:.1f} |"
        )
    L += [
        "",
        "Nominal latency = local compute + the recorded latency of any upstream (LLM / embedding / rank) call, "
        "also when served from cache. The baselines make no upstream calls.",
        "",
    ]
    d0 = docs[0]
    L += [
        "## Data and judgements",
        "",
        f"- Catalogue: {d0['data']['catalogue_routes']} routes, sha256 `{d0['data']['catalogue_sha256'][:12]}`.",
        f"- Personas: {d0['data']['n_personas']} (test {d0['data']['split_sizes']['test']}, dev "
        f"{d0['data']['split_sizes']['dev']}), sha256 `{d0['data']['personas_sha256'][:12]}`.",
    ]
    if q:
        g = q["grades_all"]
        gl = q["grades_llm"]
        L += [
            f"- Qrels: {q['pairs']} judged pairs over {q['personas']} personas, sha256 `{q['qrels_sha256'][:12]}`; "
            f"{q['by_source'].get('rule', 0)} graded 0 by rule ({100 * q['rule_share']:.1f}%: hard-constraint "
            f"violations, no LLM call), {q['by_source'].get('llm', 0)} by the LLM judge.",
            "- Grade distribution, all pairs: "
            + ", ".join(f"{k}: {v}" for k, v in g.items())
            + "; LLM-judged pairs only: "
            + ", ".join(f"{k}: {v}" for k, v in gl.items())
            + ".",
            "- Rule-graded by constraint: " + ", ".join(f"{k}: {v}" for k, v in q["rules"].items()) + ".",
            f"- Known relevant (grade ≥ 2) routes per persona: median {q['relevant_per_persona']['median']}, "
            f"min {q['relevant_per_persona']['min']}, max {q['relevant_per_persona']['max']}; personas without any: "
            f"{q['relevant_per_persona']['personas_without_relevant']}.",
            "- Pool: top-10 of every registered system plus the pool-only contributors "
            + ", ".join(f"`{c}`" for c in d0["protocol"]["pool_only_contributors"])
            + " (never reported as systems).",
        ]
    if pers:
        L += [
            "- Persona eligible-set sizes (routes passing the hard constraints): "
            f"median {pers['eligible_n']['median']:.0f}, "
            f"p10 {pers['eligible_n']['p10']:.0f}, min {pers['eligible_n']['min']:.0f}."
        ]
    L.append("")
    if pilot:
        L += [
            "## Judge pilot",
            "",
            f"{pilot['n_pairs']} pairs from {pilot['n_personas']} dev personas, graded by each candidate and by the "
            f"reference `{pilot['reference_meta']['config']['model']}` (reasoning on). The reference is itself an LLM, "
            "not ground truth.",
            "",
            "| Candidate | Model | Thinking | Batch | QWK vs reference | Exact | Mean bias | Cost/pair (nominal) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in pilot["candidates"]:
            c = r["config"]
            L.append(
                f"| `{r['run']}` | `{c['model']}` | {c['thinking']} | {c['batch_size']} | "
                f"{r['qwk_vs_reference']:.3f} | "
                f"{r['exact_agreement']:.2f} | {r['mean_grade_minus_reference']:+.2f} | "
                f"{r['nominal_cost_per_pair_usd']:.5f} USD |"
            )
        L += [
            "",
            "Batch-order sensitivity (same pilot pairs, different filler routes and order):",
            "",
            "| Config | QWK between orders | Exact |",
            "|---|---|---|",
        ]
        for s in pilot["batch_order_sensitivity"]:
            L.append(f"| `{s['config']}` | {s['qwk_between_orders']:.3f} | {s['exact_agreement']:.2f} |")
        L.append("")
    if kap:
        L += ["## Judge validation", ""]
        o = kap["owner"]
        if o["status"] == "owner labels missing":
            L.append(
                "- Owner labels: **missing** (150 blind pairs prepared in `eval/owner_labels/`; run `make kappa` "
                "once `owner_labels.csv` exists)."
            )
        else:
            L.append(f"- Owner vs judge on {o['labelled']} pairs: QWK {o['qwk']:.3f}, exact {o['exact']:.2f}.")
        if "llm_vs_llm" in kap:
            m = kap["llm_vs_llm"]
            L.append(
                f"- Interim, **not a human check**: second LLM judge `{m['second']}` "
                f"({', '.join(m['second_models'])}) vs the production judge on the same {m['n']} pairs: "
                f"QWK {m['qwk']:.3f}, exact agreement {m['exact']:.2f}, within one grade {m['within_one']:.2f}."
            )
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    paths.RESULTS_MD.write_text(render(), encoding="utf-8")
    print(f"wrote {paths.RESULTS_MD}")


if __name__ == "__main__":
    main()
