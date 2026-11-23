"""Render ``docs/RESULTS.md`` from ``eval/results/*.json`` only (``make report``), and the generated blocks of
README.md, docs/FAILURES.md and docs/REVIEW.md (``evalkit.readme``, ``evalkit.failures``). No number is typed by
hand."""

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
            f"Dev-only tuning ({cg['n_personas']} dev personas, `eval/results/_cg_dev.json`; "
            "`python -m evalkit.cg tune`). Measured against the qrels at the time of tuning, without new "
            "judgements: Recall@50 and a *condensed* "
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
                "| Precision | top-10 overlap | identical top-10 order | identical top-50 order | "
                "max abs cosine diff |",
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
            per_doc_ms = ex["ns_per_doc"] / 1e6
            for budget_ms in (10.0, 50.0):
                L.append(
                    f"At that rate exact search alone takes {budget_ms:.0f} ms at about "
                    f"{budget_ms / per_doc_ms / 1e6:.1f} million routes (linear in size and dimension)."
                )
    L.append("")
    return L


def _rank_pilot() -> list[str]:
    rp = _load("_rank_pilot.json")
    if not rp:
        return []
    L = [
        "## Ranker API pilot (Phase 3c i, dev)",
        "",
        f"Every OpenRouter reranker re-ranks the same candidate lists ({rp['candidates']}) of {rp['n_personas']} "
        "seeded dev personas, so the judged subset is identical for all models; quality is the condensed NDCG@10 "
        "over the judged candidates, cost is OpenRouter's reported charge per call, latency the measured request "
        f"time. RRF order on the same lists: {_ci(rp['rrf_order'])}. `eval/results/_rank_pilot.json`.",
        "",
        "| Model (requested) | Reported id | Document text | condensed NDCG@10 | Δ vs RRF order | Cost/call | "
        "Latency p50 / p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rp["models"] + rp["doc_formulations"][1:]:
        if "error" in r:
            L.append(f"| `{r['model']}` | – | {r['doc']} | error: {r['error'][:60]} | | | |")
            continue
        v = r["vs_rrf_order"]
        L.append(
            f"| `{r['model']}` | {', '.join(f'`{m}`' for m in r['reported_models'])} | {r['doc']} | "
            f"{_ci(r['condensed_ndcg_at_10'])} | {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
            f"{r['cost_usd_per_call']:.5f} USD | {r['latency_ms']['p50']:.0f} / {r['latency_ms']['p95']:.0f} ms |"
        )
    L += ["", f"Best on dev: `{rp['best_by_condensed_ndcg']}`.", ""]
    return L


VARIANT_TEXT = {
    "full": "all features (model `v1`, system `v6_ltr`)",
    "no_rank_api": "without the ranker-API score (model `v1_noapi`, system `v6_ltr_noapi`)",
    "no_slot_ratios": "without length / cap and distance / radius ratios",
    "no_home_distance": "without distance from home (km and ratio)",
    "no_intent_numbers": "without the intent's km figure",
    "no_rule_subscores": "without the three v0 rule sub-scores",
    "no_suspicious": "without slot ratios, home distance and intent km",
    "retrieval_and_api_only": "BM25, dense, RRF and ranker-API scores only",
}


def _ltr(docs: list[dict[str, Any]]) -> list[str]:
    d = _load("_ltr.json")
    if not d:
        return []
    dt = d["data"]
    L = [
        "## Learned ranker (Phase 3c ii)",
        "",
        f"LambdaMART (LightGBM, lambdarank) over {len(d['features'])} features of the "
        f"{d['candidates']}; trained on **dev personas only** (`eval/results/_ltr.json`, `make ltr`). Training rows: "
        f"{dt['rows']} judged candidates of {dt['personas']} dev personas (median {dt['rows_per_persona_median']:.0f} "
        f"per persona; {100 * dt['candidates_judged_share']:.0f}% of all candidates), of which "
        f"{dt['random_sample_pairs']} come from a seeded uniform sample of {dt['sample_per_persona']} previously "
        "unjudged candidates per persona (judged for training; unjudged candidates are left out, never treated as "
        "negatives). Labels: " + ", ".join(f"{g}: {c}" for g, c in dt["labels"].items()) + ".",
        "",
        "NDCG@10 here is computed over each persona's *judged candidates* (the rows the model can be scored on), "
        "so it is not the system NDCG@10 of the tables above. Cross-validation: "
        f"{d['folds']} folds grouped by persona on dev; the number of rounds is the best mean fold score. *Train* = "
        "in-sample on all dev rows. *Test* = the frozen model on test personas' judged candidates (reporting only; "
        "never used for a choice).",
        "",
        "| Variant | Rounds | Train (dev, in-sample) | CV (dev, grouped) | Test |",
        "|---|---|---|---|---|",
    ]
    for name, v in d["variants"].items():
        L.append(
            f"| {VARIANT_TEXT.get(name, name)} | {v['best_rounds']} | {v['train_ndcg_at_10']:.3f} | "
            f"{_ci(v['cv_ndcg_at_10'])} | {_ci(v['test_ndcg_at_10'])} |"
        )
    full, noapi = d["variants"]["full"], d["variants"]["no_rank_api"]
    diffs = [a - b for a, b in zip(full["cv_fold_means"], noapi["cv_fold_means"], strict=True)]
    L += [
        "",
        "Ranker-API feature, per CV fold (full − without): "
        + ", ".join(f"{x:+.3f}" for x in diffs)
        + f" (mean {sum(diffs) / len(diffs):+.3f}).",
        "",
        "Single-feature orderings of the same judged candidates (dev / test): "
        + ", ".join(
            f"`{f}` {d['single_feature_dev'][f]:.3f} / {d['single_feature_test'][f]:.3f}"
            for f in d["single_feature_dev"]
        )
        + ".",
        "",
    ]
    for mname, m in d["models"].items():
        L += [
            f"Feature importance, model `{mname}` ({m['trees']} trees; gain share and mean |Saabas contribution| on "
            "dev rows), top 12:",
            "",
            "| Feature | Gain share | Splits | Mean abs contribution |",
            "|---|---|---|---|",
        ]
        for imp in m["importance"][:12]:
            L.append(
                f"| `{imp['feature']}` | {100 * imp['gain_share']:.1f}% | {imp['splits']} | "
                f"{imp['mean_abs_contribution']:.3f} |"
            )
        L.append("")
    by = {x["system"]: x for x in docs}
    if "v6_ltr" in by:
        L.append(
            "System NDCG@10 of the learned ranker on dev is in-sample (the dev judgements are its training labels): "
            + ", ".join(
                f"`{n}` dev {by[n]['metrics']['dev']['ndcg_at_10']['mean']:.3f} vs test "
                f"{by[n]['metrics']['test']['ndcg_at_10']['mean']:.3f}"
                for n in ("v6_ltr", "v6_ltr_noapi")
                if n in by
            )
            + ". Only the test split measures it."
        )
        L.append("")
    return L


def _mmr_sweep() -> list[str]:
    d = _load("_mmr_sweep.json")
    if not d:
        return []
    L = [
        "## Diversity: MMR sweep (Phase 3d, dev)",
        "",
        f"`{d['base']}` + MMR ({d['similarity']}) for each lambda on {d['n_personas']} dev personas; every top 10 "
        f"judged. Diversity: {d['diversity_metric']}. Dev NDCG of a dev-trained ranker is optimistic; the trade-off's "
        "shape is what is used. Figure: [mmr_tradeoff.png](mmr_tradeoff.png). `eval/results/_mmr_sweep.json`.",
        "",
        "| lambda | NDCG@10 (dev) | ΔNDCG vs lambda 1 | ILD@10 (dev) | Coverage@10 (dev) |",
        "|---|---|---|---|---|",
    ]
    for r in d["rows"]:
        v = r.get("ndcg_vs_lambda_1")
        dc = f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" if v else "–"
        L.append(
            f"| {r['lambda']} | {_ci(r['ndcg_at_10'])} | {dc} | {_ci(r['ild_at_10'])} | "
            f"{100 * r['coverage_at_10']:.1f}% |"
        )
    L.append("")
    return L


def _ablation(docs: list[dict[str, Any]], cap: str) -> list[str]:
    by = {d["system"]: d for d in docs}
    chain = [n for n in ABLATION if n in by]
    if len(chain) < 2:
        return []
    L = [
        "## Phase 3 ablation (test split)",
        "",
        cap,
        "",
        "Each row is a complete registered system; each adds one component to the row above (`v2`-`v4` rank by "
        "the generator's own score, so `+ dense` replaces BM25 and `+ RRF` fuses both). The learned ranker "
        "(`v6`, `v7`) was trained on dev judgements only. Every top 10 is 100% judged.",
        "",
        "| Step | System | NDCG@10 | MRR@10 | Recall@50 | Violations@10 | ILD@10 | Cost/request | Judged@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n in chain:
        m = by[n]["metrics"]["test"]
        L.append(
            f"| {ABLATION_LABELS[n]} | `{n}` | {_ci(m['ndcg_at_10'])} | {_ci(m['mrr_at_10'])} | "
            f"{_ci(m['recall_at_50'])} | {_ci(m['violation_at_10'], pct=True)} | {_ci(m['ild_at_10'])} | "
            f"{m['cost_usd']['mean']:.5f} USD | {100 * m['judged_share_at_10']:.0f}% |"
        )
    L += [
        "",
        "Paired bootstrap, consecutive rows (test; per-persona differences, 1,000 resamples):",
        "",
    ]
    L += _paired_rows(by, list(zip(chain, chain[1:], strict=False)), "test")
    extra = _paired_rows(by, EXTRA_PAIRS, "test")
    if extra:
        L += ["", "Other comparisons (test):", ""] + extra
    L.append("")
    return L


def _default_system(docs: list[dict[str, Any]]) -> list[str]:
    lat = _load("_latency.json")
    by = {d["system"]: d for d in docs}
    if not lat or not any(n in by for n in lat["systems"]):
        return []
    L = [
        "## Latency and cost against the budget (served default and Phase 3 default)",
        "",
        "Budget for this project: p95 ≤ 1.5 s nominal for a pipeline that calls the ranker API, p95 ≤ 500 ms for "
        f"one that does not (query embedding included). Warm benchmark (`make latency`, {lat['n']} sequential dev "
        f"requests after {lat['warmup']} warm-up requests, LLM cache off, live Vertex embedding and OpenRouter "
        f"ranking, one {lat['machine']} laptop): wall time of `Pipeline.arecommend`. Cost per request: the mean "
        "nominal upstream cost over the test personas in the evaluation.",
        "",
        "| System | Ranker API | p50 ms | p95 ms | max ms | Upstream p95 ms by stage | Local compute p95 ms | "
        "Cost/request | Budget p95 | Within |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, v in lat["systems"].items():
        up = ", ".join(f"{k} {x:.0f}" for k, x in v["upstream_ms_p95"].items())
        cost = by[n]["metrics"]["test"]["cost_usd"]["mean"] if n in by else v["cost_usd_per_request"]
        L.append(
            f"| `{n}` | {'yes' if v['uses_rank_api'] else 'no'} | {v['wall_ms']['p50']:.0f} | "
            f"{v['wall_ms']['p95']:.0f} | {v['wall_ms']['max']:.0f} | {up} | {v['local_compute_ms']['p95']:.1f} | "
            f"{cost:.6f} USD | {v['budget_ms_p95']:.0f} ms | {'yes' if v['within_budget'] else '**no**'} |"
        )
    L += [
        "",
        "The evaluation's own nominal latency (section *Latency* below) adds the latency recorded when each upstream "
        "call was first made, 8 calls at a time from a cold process; its p95 includes those cold starts.",
        "",
    ]
    return L


def _reference(cap: str) -> list[str]:
    r = _load("_reference.json")
    if not r:
        return []
    L = [
        "## Reference: the latent matcher (not a system)",
        "",
        "**Not a system: reads the simulated user's latent profile.** `pool_latent` is the pool-only contributor "
        "that scores routes with the persona's hidden preferences (EVAL.md section 4). It is shown only as a "
        "rough indication of headroom and is not part of the ablation or any comparison. Its top 10 is fully "
        "judged by construction (it is pooled), and the judge reads the same latent profile, so this is an "
        "optimistic ceiling for what a system could reach from the visible query.",
        "",
        cap,
        "",
        "| Split | NDCG@10 | MRR@10 | Recall@50 | ILD@10 | Judged@10 |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("test", "dev"):
        m = r["metrics"][split]
        L.append(
            f"| {split} | {_ci(m['ndcg_at_10'])} | {_ci(m['mrr_at_10'])} | {_ci(m['recall_at_50'])} | "
            f"{_ci(m['ild_at_10'])} | {100 * m['judged_share_at_10']:.0f}% |"
        )
    L.append("")
    return L


def _explanation_example() -> list[str]:
    ex = _load("_explanation_example.json")
    if not ex:
        return []
    return [
        "## Explanation example (Phase 3 default)",
        "",
        f"Real output of `{ex['system']}` for test persona `{ex['persona_id']}` (`python -m evalkit.explain_example`; "
        "schema `recsys.explanation/v1`, documented in [RECSYS.md](RECSYS.md)). Intent: "
        f"\"{ex['intent']}\". Top route `{ex['top_route']['route_id']}` ({ex['top_route']['title']}):",
        "",
        "```json",
        json.dumps(ex["top_route"], indent=1, ensure_ascii=False),
        "```",
        "",
    ]


# --------------------------------------------------------------------------- Phase 5 and Part D sections
def _pair_cell(r: dict[str, Any]) -> str:
    return "n/a" if r.get("diff") is None else f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


ROBUST = ["recsys_v1", "recsys_v1_nocap", "recsys_v1_nosusp"]
ROBUST_TEXT = {
    "recsys_v1": "Phase 3 default (ranker `v1_noapi`, all non-API features)",
    "recsys_v1_nocap": "Phase 3 default, ranker without `length_cap_ratio` (`v1_noapi_nocap`; the served ranker)",
    "recsys_v1_nosusp": "Phase 3 default, ranker without slot ratios, home distance and intent km (`v1_noapi_nosusp`)",
}


def _robustness(docs: list[dict[str, Any]], cap: str) -> list[str]:
    by = {d["system"]: d for d in docs}
    if not all(n in by for n in ROBUST):
        return []
    rb = _load("_ltr_robust.json") or {}
    ltr = _load("_ltr.json") or {}
    L = [
        "## Robustness of the learned ranker: without the simulator-shaped features (system level)",
        "",
        cap,
        "",
        "**Robustness rows, not new defaults.** `length_cap_ratio` (route length / the user's distance cap) is the "
        "learned ranker's largest feature and encodes how the persona generator draws caps (RECSYS.md section 4, "
        "DECISIONS.md D15). These rows retrain the Phase 3 default's ranker on the same pinned dev labels without "
        "that feature, or without the whole suspicious group, and run the same pipeline otherwise (filters, "
        "BM25 + dense, RRF, MMR 0.8, no ranker API). Every top 10 is 100% judged. Dev rows are in-sample for all "
        "three rankers (trained on dev labels).",
        "",
        "| System | Ranker | NDCG@10 test | MRR@10 test | NDCG@10 dev (in-sample) | CV NDCG@10 (dev, grouped) | "
        "Judged@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    cv = {"recsys_v1": (ltr.get("variants", {}).get("no_rank_api") or {}).get("cv_ndcg_at_10")}
    for v in (rb.get("variants") or {}).values():
        cv["recsys_v1_nocap" if v["model"].endswith("nocap") else "recsys_v1_nosusp"] = v["cv_ndcg_at_10"]
    for n in ROBUST:
        m = by[n]["metrics"]
        L.append(
            f"| `{n}` | {ROBUST_TEXT[n]} | {_ci(m['test']['ndcg_at_10'])} | {_ci(m['test']['mrr_at_10'])} | "
            f"{_ci(m['dev']['ndcg_at_10'])} | {_ci(cv[n]) if cv.get(n) else 'n/a'} | "
            f"{100 * min(m['test']['judged_share_at_10'], m['dev']['judged_share_at_10']):.0f}% |"
        )
    L += ["", "Paired differences against `recsys_v1` (per-persona, 1,000 resamples):", ""]
    for split in ("test", "dev"):
        pairs = [("recsys_v1", "recsys_v1_nocap"), ("recsys_v1", "recsys_v1_nosusp")]
        L += [f"*{split}*", ""] + _paired_rows(by, pairs, split) + [""]
    if rb:
        L.append(
            "Largest features without `length_cap_ratio` (gain share): "
            + "; ".join(
                f"`{v['model']}`: "
                + ", ".join(f"`{i['feature']}` {100 * i['gain_share']:.1f}%" for i in v["importance_top"][:4])
                for v in rb["variants"].values()
            )
            + ". `eval/results/_ltr_robust.json`, `make ltr-robust`."
        )
        L.append("")
    return L


E2E = [("recsys_v1_qonly", "questionnaire only (user vector + the form's profile slots)"),
       ("recsys_v1", "questionnaire + raw intent text (the Phase 3 default)"),
       ("recsys_v1_chat", "questionnaire + chat (intent sent as one chat message; slots merged, restated intent)")]


def _e2e(docs: list[dict[str, Any]], cap: str) -> list[str]:
    by = {d["system"]: d for d in docs}
    if not all(n in by for n, _ in E2E):
        return []
    chat_cfg = by["recsys_v1_chat"]["config"].get("chat", {})
    L = [
        "## Phase 4 end-to-end: questionnaire only vs + free text vs + chat (test split)",
        "",
        cap,
        "",
        f"Same pipeline (`recsys_v1`) behind three ways of collecting preferences. The chat variant runs the real turn "
        f"logic with `{chat_cfg.get('model')}` (prompt `{chat_cfg.get('prompt_version')}`), starting from the form's "
        "slots. Every top 10 is 100% judged. **What this can show:** how much the pipeline gains from free text, and "
        "whether routing that text through the slot extractor helps or hurts. **What it cannot show:** the intents "
        "were written by an LLM from the same latent profile the judge reads, and every persona sends exactly one "
        "well-formed message, so the free-text rows measure recovery of what the generator put in, not real users' "
        "messages; the chat row also inherits the extractor's optimism (CHAT.md section 8).",
        "",
        "| Input | System | NDCG@10 | MRR@10 | Recall@50 | Violations@10 | Cost/request | Judged@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, label in E2E:
        m = by[n]["metrics"]["test"]
        L.append(
            f"| {label} | `{n}` | {_ci(m['ndcg_at_10'])} | {_ci(m['mrr_at_10'])} | {_ci(m['recall_at_50'])} | "
            f"{_ci(m['violation_at_10'], pct=True)} | {m['cost_usd']['mean']:.5f} USD | "
            f"{100 * m['judged_share_at_10']:.0f}% |"
        )
    L += [
        "",
        "Violations@10 of the chat row are routes outside the persona's *own* radius or cap: when the message names a "
        "town, the chat moves the place to that catalogue locality (CHAT.md merge rule 6), so the radius is centred "
        "elsewhere. The judge grades such routes 0 by rule.",
        "",
        "Paired differences (test):",
        "",
    ]
    L += _paired_rows(by, [("recsys_v1_qonly", "recsys_v1"), ("recsys_v1_qonly", "recsys_v1_chat"),
                           ("recsys_v1", "recsys_v1_chat")], "test")
    L.append("")
    return L


def _chat_eval() -> list[str]:
    d = _load("_chat_eval.json")
    if not d:
        return []
    L = [
        "## Chat slot extraction: scripted evaluation (Phase 4)",
        "",
        f"{d['cases']} scripted cases (`eval/chat/utterances.jsonl`), every candidate through the real turn logic; "
        "`make chat-eval` replays them from `eval/chat/cache/` (results `eval/results/chat/`, summary "
        "`eval/results/_chat_eval.json`). Matching rules and caveats: CHAT.md section 8. **The cases, expected slots, "
        "prompt and keyword baseline were written by the same agent: these numbers measure agreement with that "
        "agent's labels, not robustness to real users.** Default model: DECISIONS.md D18.",
        "",
        "| Candidate | Model | Overall P / R | enum P / R | number P / R | place P / R | tags P / R | "
        "Clarify decision acc. | Schema violations |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def pr(x: dict[str, Any]) -> str:
        return f"{x['precision']:.3f} / {x['recall']:.3f}"

    for r in d["candidates"]:
        t = r["per_type"]
        L.append(
            f"| `{r['candidate']}` | `{r['model']}` | {pr(r['overall'])} | {pr(t['enum'])} | {pr(t['number'])} | "
            f"{pr(t['place'])} | {pr(t['tags'])} | {r['clarify_decision_accuracy']:.3f} | "
            f"{100 * (r['schema_violation_rate'] or 0):.1f}% ({r['schema_failed_calls']} failed calls) |"
        )
    det = _load("chat/haiku55.json")
    if det:
        L += ["", "Per slot, default model (`haiku55`):", "", "| Slot | TP | FP | FN | Precision | Recall |",
              "|---|---|---|---|---|---|"]
        for slot, v in det["metrics"]["per_slot"].items():
            L.append(f"| `{slot}` | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.3f} | {v['recall']:.3f} |")
    L += ["", "`rules` is the offline keyword extractor (the mock provider's answer), not an LLM.", ""]
    return L


GRID_KEYS = ("beta", "gamma", "eta_emb", "eta_tag", "nu_dup", "difficulty_step", "cap_step", "radius_step")
SIM_ORDER = ["v0_rules", "v1_filters", "recsys_v1", "recsys_v1_fb_route", "recsys_v1_fb", "recsys_v1_fb_light",
             "recsys_v1_fb_explore", "recsys_serve_noexplore", "recsys_serve"]

SERVED = "recsys_serve"


def _served(docs: list[dict[str, Any]], cap: str) -> list[str]:
    """Headline: the system the API serves (D25), single-request and in the simulation."""
    by = {d["system"]: d for d in docs}
    if SERVED not in by or "recsys_v1" not in by:
        return []
    L = [
        "## Served default: `recsys_serve` (D25)",
        "",
        cap,
        "",
        "**The system the API serves**: the ranker trained without `length_cap_ratio` (`recsys_v1_nocap`, D19), the "
        "feedback loop as chosen on dev (`recsys_v1_fb`'s settings, D21) and the exploration slot at rank 10 (D22). "
        "A judgement call made without real users (DECISIONS.md D25). In a single request there is no feedback, so "
        "it differs from `recsys_v1_nocap` only by the exploration pick at rank 10. Every top 10 is 100% judged.",
        "",
        "| System | NDCG@10 test | MRR@10 test | Recall@50 test | Violations@10 test | NDCG@10 dev (in-sample) | "
        "Cost/request |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in ("recsys_v1", "recsys_v1_nocap", SERVED):
        if n not in by:
            continue
        m = by[n]["metrics"]
        lab = f"**`{n}` (served default)**" if n == SERVED else f"`{n}`"
        L.append(
            f"| {lab} | {_ci(m['test']['ndcg_at_10'])} | {_ci(m['test']['mrr_at_10'])} | "
            f"{_ci(m['test']['recall_at_50'])} | {_ci(m['test']['violation_at_10'], pct=True)} | "
            f"{_ci(m['dev']['ndcg_at_10'])} | {m['test']['cost_usd']['mean']:.5f} USD |"
        )
    for split in ("test", "dev"):
        L += ["", f"Paired differences ({split}):", ""]
        L += _paired_rows(by, [("recsys_v1", SERVED), ("recsys_v1_nocap", SERVED)], split)
    sim = _load("_simulation.json")
    if sim and SERVED in sim["metrics"]:
        met = sim["metrics"]
        L += [
            "",
            "In the 10-round feedback simulation (test; simulated users *and* a simulated behaviour model, section "
            "\"Feedback loop\"):",
            "",
            "| System | NDCG@10, mean of rounds 2-10 | NDCG@10, round 10 | Cumulative coverage, round 10 |",
            "|---|---|---|---|",
        ]
        for n in ("recsys_v1", "recsys_v1_fb", "recsys_serve_noexplore", SERVED):
            if n in met:
                lab = f"**`{n}` (served default)**" if n == SERVED else f"`{n}`"
                L.append(
                    f"| {lab} | {_ci(met[n]['ndcg_rounds_2_10'])} | {_ci(met[n]['rounds'][-1]['ndcg_at_10'])} | "
                    f"{100 * met[n]['rounds'][-1]['cumulative_coverage']:.1f}% |"
                )
        L += ["", "| Comparison | ΔNDCG@10, rounds 2-10 [95% CI] | ΔNDCG@10, round 10 |", "|---|---|---|"]
        for pr_ in sim["paired"]:
            if pr_["b"] in (SERVED, "recsys_serve_noexplore") and pr_["a"] in (
                "recsys_v1", "recsys_v1_fb", "recsys_serve_noexplore"
            ):
                L.append(
                    f"| `{pr_['b']}` − `{pr_['a']}` | {_pair_cell(pr_['rounds_2_10'])} | "
                    f"{_pair_cell(pr_['per_round'][-1])} |"
                )
    L.append("")
    return L


EMPTY = [
    ("recsys_serve", "all slots (home, radius, cap)"),
    ("recsys_serve_nocap", "no distance cap"),
    ("recsys_serve_nohome", "no home (so no radius filter)"),
    ("recsys_serve_noslots", "no home and no cap"),
    ("recsys_v1", "all slots"),
    ("recsys_v1_nocapslot", "no distance cap"),
]


def _empty_slots(docs: list[dict[str, Any]], cap: str) -> list[str]:
    by = {d["system"]: d for d in docs}
    if not all(n in by for n, _ in EMPTY):
        return []
    L = [
        "## Empty profile slots: what the served system does when a user skips them (test)",
        "",
        cap,
        "",
        "The same requests with slots removed before the pipeline sees them (`app/recsys/systems/empty_slots.py`); "
        "relevance is still judged against the persona's real limits, so a route beyond the persona's own radius or "
        "cap is graded 0 by rule. Every top 10 is 100% judged. `recsys_v1_nocapslot` removes the cap from "
        "`recsys_v1`, whose ranker uses `length_cap_ratio` and never saw it missing in training (D19).",
        "",
        "| System | Slots given | NDCG@10 | MRR@10 | Violations@10 | Judged@10 |",
        "|---|---|---|---|---|---|",
    ]
    for n, label in EMPTY:
        m = by[n]["metrics"]["test"]
        L.append(
            f"| `{n}` | {label} | {_ci(m['ndcg_at_10'])} | {_ci(m['mrr_at_10'])} | "
            f"{_ci(m['violation_at_10'], pct=True)} | {100 * m['judged_share_at_10']:.0f}% |"
        )
    L += ["", "Paired differences (test):", ""]
    L += _paired_rows(
        by,
        [("recsys_serve", "recsys_serve_nocap"), ("recsys_serve", "recsys_serve_nohome"),
         ("recsys_serve", "recsys_serve_noslots"), ("recsys_v1", "recsys_v1_nocapslot"),
         ("recsys_v1_nocapslot", "recsys_serve_nocap")],
        "test",
    )
    L.append("")
    return L


def _simulation() -> list[str]:
    d = _load("_simulation.json")
    if not d:
        return []
    met = d["metrics"]
    labs = [n for n in SIM_ORDER if n in met]
    L = [
        "## Feedback loop: 10-round simulation (Phase 5, test split)",
        "",
        f"*Simulated users ({d['n_personas']} test personas) **and a simulated behaviour model written for this "
        "project** (EVAL.md section 10); LLM-judged relevance, judge not yet validated against the owner's labels. A "
        "feedback loop that looks good here looks good on these assumptions.*",
        "",
        "Each round (one week apart) every system shows its top 10 for the persona's query plus the feedback of the "
        "earlier rounds; every shown route is judged (holes filled on demand, `eval/qrels/sim_qrels.jsonl`); the "
        "simulated user then completes at most one route and rejects a few, with a reason. **Per-round NDCG@10:** "
        "routes completed or rejected in earlier rounds have gain 0 and are removed from the ideal ranking. "
        "`make simulate`; `eval/results/_simulation.json`; figure [convergence.png](convergence.png).",
        "",
        "Systems: " + "; ".join(f"`{n}` = {d['systems'][n]['label']}" for n in labs) + ".",
        "",
        "NDCG@10 per round (mean over personas, 95% bootstrap CI):",
        "",
        "| Round | " + " | ".join(f"`{n}`" for n in labs) + " |",
        "|---|" + "---|" * len(labs),
    ]
    for t in range(d["rounds"]):
        L.append(f"| {t + 1} | " + " | ".join(_ci(met[n]["rounds"][t]["ndcg_at_10"]) for n in labs) + " |")
    L.append("| mean of rounds 2-10 | " + " | ".join(f"**{_ci(met[n]['ndcg_rounds_2_10'])}**" for n in labs) + " |")
    L += [
        "",
        "Coverage@10 per round (distinct routes in the 100 top-10 lists / catalogue) and cumulative coverage (distinct "
        "routes shown in any round so far); no CI (set statistics):",
        "",
        "| Round | " + " | ".join(f"`{n}`" for n in labs) + " |",
        "|---|" + "---|" * len(labs),
    ]
    for t in range(d["rounds"]):
        L.append(
            f"| {t + 1} | "
            + " | ".join(
                f"{100 * met[n]['rounds'][t]['coverage_at_10']:.1f}% / "
                f"{100 * met[n]['rounds'][t]['cumulative_coverage']:.1f}%"
                for n in labs
            )
            + " |"
        )
    L += [
        "",
        "Simulated reactions over the 10 rounds (per persona): completions, and all events by reason:",
        "",
        "| System | Completions / persona | Events by reason (total) | Consumed routes re-shown / list, round 10 |",
        "|---|---|---|---|",
    ]
    for n in labs:
        ev = ", ".join(f"{k} {v}" for k, v in met[n]["events_by_reason"].items())
        L.append(
            f"| `{n}` | {met[n]['completions_total']:.2f} | {ev} | "
            f"{met[n]['rounds'][-1]['reshown_consumed_per_list']:.2f} |"
        )
    L += [
        "",
        "Paired differences (b − a; per persona; mean of rounds 2-10, and round 10 alone):",
        "",
        "| Comparison | ΔNDCG@10, rounds 2-10 [95% CI] | P(Δ ≤ 0) | ΔNDCG@10, round 10 | n |",
        "|---|---|---|---|---|",
    ]
    for pr_ in d["paired"]:
        r = pr_["rounds_2_10"]
        L.append(
            f"| `{pr_['b']}` − `{pr_['a']}` | {_pair_cell(r)} | {r['p_le_0']:.3f} | "
            f"{_pair_cell(pr_['per_round'][-1])} | {r['n']} |"
        )
    for ex in (d.get("exploration") or {}).values():
        a, b = ex["a"], ex["b"]
        p_ex = next((p for p in d["paired"] if p["a"] == a and p["b"] == b), None)
        if not p_ex:
            continue
        L += [
            "",
            f"Exploration on `{a}`: NDCG@10 cost (`{b}` − `{a}`, paired) next to the coverage@10 gain, per round:",
            "",
            "| Round | ΔNDCG@10 [95% CI] | Δcoverage@10 (points) |",
            "|---|---|---|",
        ]
        for t, row in enumerate(ex["per_round_coverage_gain"]):
            L.append(f"| {t + 1} | {_pair_cell(p_ex['per_round'][t])} | {100 * row['coverage_gain']:+.2f} |")
        cm = met[b]["rounds"][-1]["cumulative_coverage"]
        cb = met[a]["rounds"][-1]["cumulative_coverage"]
        L += [
            "",
            f"Mean ΔNDCG@10 over rounds 2-10: {_pair_cell(p_ex['rounds_2_10'])}. Cumulative coverage after 10 rounds: "
            f"{100 * cb:.1f}% without, {100 * cm:.1f}% with the exploration slot.",
        ]
    L.append("")
    tu = _load("_sim_tuning.json")
    if tu:
        L += [
            "### Dev tuning of the feedback parameters",
            "",
            f"Same simulation on {len(tu['personas'])} seeded dev personas, 10 rounds, one system per grid point "
            f"(`make sim-tune`, `eval/results/_sim_tuning.json`). Rule: {tu['rule']}. Chosen: **`{tu['best']}`**. The "
            "learned ranker is in-sample on dev (trained on dev labels), so dev lists start near the ideal and leave "
            "less for feedback to fix than test lists do.",
            "",
            "| Grid point | NDCG@10 rounds 2-10 | Δ vs route effects only [95% CI] | Completions / persona | "
            "Parameters |",
            "|---|---|---|---|---|",
        ]
        for n, m in tu["metrics"].items():
            g = tu["grid"].get(n)
            params = (
                "feedback off"
                if g is None
                else ", ".join(f"{k} {g[k]}" for k in GRID_KEYS)
            )
            dv = tu["vs_route_only"].get(n)
            L.append(
                f"| `{n}` | {_ci(m['ndcg_rounds_2_10'])} | {_pair_cell(dv) if dv else '–'} | {m['completions']:.2f} | "
                f"{params} |"
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
    L += _served(docs, cap)
    L += _ablation(docs, cap)
    L += _default_system(docs)
    L += _reference(cap)
    for split, title in (("test", "All systems, test split"), ("dev", "All systems, dev split")):
        L += [f"## {title}", "", cap, ""] + _table(docs, split) + [""]
        paired = _paired(docs, split)
        if paired:
            L += [f"Paired bootstrap of the per-persona difference ({split}):", "", cap, ""] + paired + [""]
    L += _robustness(docs, cap)
    L += _empty_slots(docs, cap)
    L += _simulation()
    L += _e2e(docs, cap)
    L += _chat_eval()
    L += _candidate_generation(docs, cap)
    L += _rank_pilot()
    L += _explanation_example()
    L += _ltr(docs)
    L += _mmr_sweep()
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


def doc_blocks() -> dict[str, dict[str, str]]:
    """Generated blocks of the hand-written documents, by file (relative to the repo root)."""
    from evalkit import blocks, failures, readme

    failures_md = paths.DOCS_DIR / "FAILURES.md"
    out = {"README.md": readme.readme_blocks(), "docs/REVIEW.md": readme.review_blocks()}
    out["docs/FAILURES.md"] = failures.render_blocks(blocks.strip_blocks(failures_md.read_text(encoding="utf-8")))
    return out


def main() -> None:
    from evalkit import blocks

    paths.RESULTS_MD.write_text(render(), encoding="utf-8")
    print(f"wrote {paths.RESULTS_MD}")
    for rel, bl in doc_blocks().items():
        path = paths.REPO_ROOT / rel
        if not path.exists():
            continue
        changed = blocks.update_file(path, bl)
        print(f"{'updated' if changed else 'unchanged'} {len(bl)} generated blocks in {rel}")


if __name__ == "__main__":
    main()
