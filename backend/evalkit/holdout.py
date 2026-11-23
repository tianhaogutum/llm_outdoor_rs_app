"""Iteration 3: a fresh holdout of simulated users and the decision pre-registered in docs/DECISIONS.md D28.

    cd backend && python -m evalkit.holdout generate          # make holdout-personas (intents: live once, cached)
    cd backend && python -m evalkit.evaluate --estimate recsys_serve recsys_serve_needs ... --holdout  # cost first
    make eval SYSTEM=<candidate> HOLDOUT=1                    # each candidate on the holdout, holes judged
    cd backend && python -m evalkit.holdout decide            # eval/results/_holdout.json (eval-all rewrites it)

**Generation.** The same generator and parameters as the main 300 (``evalkit.personas.PARAMS``) with a new seed
(``SEED``), every persona in split ``holdout`` with ids ``H001``... (the draw index). Draws are taken in order; a
draw whose latent profile, or whose profile slots a system sees (questionnaire vector, tab, home, travel radius,
distance cap), equal those of a main persona or an earlier holdout persona is skipped, so the id sequence shows
any skip. Intents are written by the same writer as before (``evalkit.intents``: model, prompt, temperature).
Identical intent texts are counted and reported, not removed: two people may type the same words.

**Use.** Only the decision below, then reporting. The holdout has its own qrels file
(``eval/qrels/holdout_qrels.jsonl``); its pool is the top 10 of ``BASELINE`` and ``CANDIDATES`` plus the two
pool-only contributors; nothing is trained, tuned or chosen on it.

**The rule** (``RULE``; the text in D28 is the commitment, this code implements it): see :func:`decide`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from app.recsys import Catalog, get_catalog
from evalkit import paths
from evalkit.personas import PARAMS, Persona, generate, load_personas, persona_stats, write_personas

SEED = 20261128  # fixed in D28 before any holdout persona existed
N = 150
SPARE = 30  # extra draws available to replace skipped duplicates
ID_PREFIX = "H"
SPLIT = "holdout"
BASELINE = "recsys_serve"
CANDIDATES = ("recsys_serve_needs", "recsys_serve_needs_promote", "recsys_serve_needs_demote")
NEW_SERVED_NAME = "recsys_serve_v2"  # the name a winning variant is registered under (recsys_serve stays as is)
ALPHA = 0.05
N_BOOT_GATE = 1000  # the harness standard (evaluate.N_BOOT)
N_BOOT_DECISION = 10000
OUT = "_holdout.json"
SCHEMA = "evalkit.holdout_decision/v1"

RULE: dict[str, Any] = {
    "registered": "docs/DECISIONS.md D28, committed before the holdout personas existed",
    "baseline": BASELINE,
    "candidates": list(CANDIDATES),
    "metric": "NDCG@10, every top 10 judged (judge g35f-min-b8, rubric/v1)",
    "gate_dev_crossfit": "paired bootstrap 95% CI (1,000 resamples, seed 20261122) of mean cross-fitted dev "
    "NDCG@10(variant) - NDCG@10(recsys_serve) has its lower bound above 0; variants failing it are not tested",
    "holdout_test": "for the m gated variants jointly: one-sided simultaneous 95% lower bounds of the mean paired "
    "holdout difference NDCG@10(variant) - NDCG@10(recsys_serve), single-step max-T over a joint paired bootstrap "
    "of the holdout personas (10,000 resamples, seed 20261122); family-wise error 5% over the m variants",
    "eligible": "simultaneous lower bound > 0 and mean holdout Violations@10 not above recsys_serve's",
    "choose": "the eligible variant with the largest simultaneous lower bound (ties: order of candidates); it is "
    f"registered as {NEW_SERVED_NAME} with the variant's config and becomes the API default",
    "null": "no eligible variant: recsys_serve stays the served default, the variants stay registered and off",
    "precondition": "every top 10 of every compared system on the holdout is judged (judged@10 = 100%)",
}


# --------------------------------------------------------------------------------------------- generation
def personas_file():
    return paths.PERSONA_DIR / "holdout.jsonl"


def params_file():
    return paths.PERSONA_DIR / "holdout_params.json"


def stats_file():
    return paths.PERSONA_DIR / "holdout_stats.json"


def _latent_key(p: Persona) -> str:
    return json.dumps(p.latent, sort_keys=True)


def _visible_key(p: Persona) -> str:
    q = p.to_query()
    return json.dumps(
        {
            "user_vector": q.user_vector,
            "category": q.category,
            "home": list(q.home) if q.home else None,
            "max_travel_km": q.max_travel_km,
            "max_distance_km": q.max_distance_km,
        },
        sort_keys=True,
    )


def draw(
    existing: list[Persona],
    *,
    catalog: Optional[Catalog] = None,
    towns: Optional[list[dict]] = None,
    n: int = N,
    spare: int = SPARE,
    seed: int = SEED,
) -> tuple[list[Persona], dict[str, Any]]:
    """The first ``n`` draws of the seeded stream that duplicate neither ``existing`` nor an earlier draw
    (latent profile or visible profile slots). Raises if the spare draws run out."""
    cands = generate(catalog, towns, n=n + spare, seed=seed, id_prefix=ID_PREFIX, split=SPLIT)
    lat = {_latent_key(p): p.persona_id for p in existing}
    vis = {_visible_key(p): p.persona_id for p in existing}
    chosen: list[Persona] = []
    skipped: list[dict[str, str]] = []
    for p in cands:
        if len(chosen) == n:
            break
        lk, vk = _latent_key(p), _visible_key(p)
        if lk in lat or vk in vis:
            skipped.append({"draw": p.persona_id, "duplicates": lat.get(lk) or vis[vk]})
            continue
        lat[lk] = vis[vk] = p.persona_id
        chosen.append(p)
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} non-duplicate draws in {n + spare}")
    return chosen, {"draws_used": int(chosen[-1].persona_id[len(ID_PREFIX) :]), "skipped": skipped}


def _norm(text: Optional[str]) -> str:
    return " ".join((text or "").lower().split())


def intent_overlap(holdout: list[Persona], existing: list[Persona]) -> list[dict[str, str]]:
    """Holdout intents whose normalised text equals a main intent or an earlier holdout intent (reported only)."""
    seen = {_norm(p.intent): p.persona_id for p in existing}
    out = []
    for p in holdout:
        k = _norm(p.intent)
        if k in seen:
            out.append({"persona": p.persona_id, "same_text_as": seen[k], "intent": p.intent or ""})
        else:
            seen[k] = p.persona_id
    return out


def generate_holdout(client=None) -> tuple[list[Persona], dict[str, Any]]:
    """Draw, check and write the holdout personas (``eval/personas/holdout.jsonl`` and its params / stats)."""
    from evalkit import intents

    existing = load_personas()
    chosen, rep = draw(existing)
    intents.add_intents(chosen, client or intents.make_writer_client(max_concurrency=6))
    rep["identical_intent_texts"] = intent_overlap(chosen, existing)
    sha = write_personas(chosen, personas_file())
    params_file().write_text(
        json.dumps(
            {
                "seed": SEED,
                "n": N,
                "spare_draws": SPARE,
                "id_prefix": ID_PREFIX,
                "split": SPLIT,
                "generator_params": PARAMS,
                "intent_writer": intents.writer_config(),
                "dedup": rep,
                "note": "same generator and parameters as eval/personas/personas.jsonl, new seed (D28)",
            },
            indent=1,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    st = persona_stats(chosen)
    st["personas_sha256"] = sha
    stats_file().write_text(json.dumps(st, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return chosen, rep


# ----------------------------------------------------------------------------------------------- decision
def _load(path) -> Optional[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def decide() -> dict[str, Any]:
    """Apply ``RULE`` to the stored results: the cross-fitted dev gate (``eval/results/crossfit/``), then the
    holdout comparison (``eval/results/holdout/``). Pure function of committed files; no judging."""
    from evalkit import evaluate as E
    from evalkit.metrics import paired_bootstrap, simultaneous_lower_bounds

    seed = E.BOOT_SEED
    doc: dict[str, Any] = {"schema": SCHEMA, "rule": RULE, "seed": SEED}
    hs = _load(stats_file())
    hp = _load(params_file())
    if hs:
        doc["personas"] = {
            "n": hs["n"],
            "seed": SEED,
            "sha256": hs.get("personas_sha256"),
            "activity_style": hs["activity_style"],
            "party": hs["party"],
            "transport": hs["transport"],
            "multi_day": hs["multi_day"],
            "intent_styles": hs["intent_styles"],
            "eligible_n": hs["eligible_n"],
            "dedup": (hp or {}).get("dedup"),
        }
    # 1. gate on cross-fitted dev
    cf = {n: _load(E.CROSSFIT.results_dir / f"{n}.json") for n in (BASELINE, *CANDIDATES)}
    gate: dict[str, Any] = {}
    if cf[BASELINE]:
        b = cf[BASELINE]["per_persona"]
        for v in CANDIDATES:
            if not cf[v]:
                gate[v] = {"status": "missing", "passed": False}
                continue
            pids = sorted(b)
            r = paired_bootstrap(
                [b[p]["ndcg_at_10"] for p in pids],
                [cf[v]["per_persona"][p]["ndcg_at_10"] for p in pids],
                n_boot=N_BOOT_GATE,
                seed=seed,
            )
            gate[v] = {**r, "passed": bool(r["lo"] is not None and r["lo"] > 0)}
    doc["gate_dev_crossfit"] = gate
    gated = [v for v in CANDIDATES if gate.get(v, {}).get("passed")]
    # 2. holdout
    ho = {n: _load(E.HOLDOUT.results_dir / f"{n}.json") for n in (BASELINE, *CANDIDATES)}
    if not ho[BASELINE] or not all(ho[v] for v in CANDIDATES):
        doc["status"] = "incomplete: not every candidate has been evaluated on the holdout"
        return doc
    systems = {}
    for n, d in ho.items():
        m = d["metrics"][SPLIT]
        systems[n] = {
            k: m[k] for k in ("ndcg_at_10", "mrr_at_10", "recall_at_50", "violation_at_10", "judged_share_at_10")
        }
        systems[n]["qrels_sha256"] = d["data"]["qrels_sha256"]
    doc["systems"] = systems
    if any(s["judged_share_at_10"] < 1.0 for s in systems.values()):
        doc["status"] = "invalid: a top 10 is not fully judged (judge the holes first)"
        return doc
    base = ho[BASELINE]["per_persona"]
    units = sorted(p for p in base if all(ho[n]["per_persona"][p]["ndcg_at_10"] is not None for n in ho))
    doc["units"] = {"personas": len(units), "ndcg_undefined": len(base) - len(units)}
    paired = {}
    for v in CANDIDATES:
        x = [base[p]["ndcg_at_10"] for p in units]
        y = [ho[v]["per_persona"][p]["ndcg_at_10"] for p in units]
        paired[v] = paired_bootstrap(x, y, n_boot=N_BOOT_GATE, seed=seed)
        paired[v]["better_by_0_1"] = int(sum(b_ - a_ > 0.1 for a_, b_ in zip(x, y, strict=True)))
        paired[v]["worse_by_0_1"] = int(sum(b_ - a_ < -0.1 for a_, b_ in zip(x, y, strict=True)))
    doc["paired_95"] = paired
    sim: dict[str, Any] = {}
    if gated:
        sim = simultaneous_lower_bounds(
            {v: [ho[v]["per_persona"][p]["ndcg_at_10"] - base[p]["ndcg_at_10"] for p in units] for v in gated},
            n_boot=N_BOOT_DECISION,
            seed=seed,
            alpha=ALPHA,
        )
    doc["simultaneous"] = sim
    verdict = {}
    for v in CANDIDATES:
        lb = sim["by_name"][v]["lower"] if v in gated else None
        viol_ok = systems[v]["violation_at_10"]["mean"] <= systems[BASELINE]["violation_at_10"]["mean"]
        verdict[v] = {
            "gated": v in gated,
            "simultaneous_lower": lb,
            "violations_ok": bool(viol_ok),
            "eligible": bool(v in gated and lb is not None and lb > 0 and viol_ok),
        }
    doc["verdict"] = verdict
    eligible = [v for v in CANDIDATES if verdict[v]["eligible"]]
    if eligible:
        best = max(eligible, key=lambda v: (verdict[v]["simultaneous_lower"], -CANDIDATES.index(v)))
        doc["decision"] = {
            "outcome": "switch",
            "winner": best,
            "serve": NEW_SERVED_NAME,
            "text": f"{best} wins by the pre-registered rule; it is served as {NEW_SERVED_NAME}",
        }
    else:
        doc["decision"] = {
            "outcome": "null",
            "winner": None,
            "serve": BASELINE,
            "text": "no variant is eligible by the pre-registered rule; recsys_serve stays the served default",
        }
    doc["status"] = "complete"
    return doc


def write_decision() -> dict[str, Any]:
    from evalkit.evaluate import dumps

    doc = decide()
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS_DIR / OUT).write_text(dumps(doc), encoding="utf-8")
    return doc


# ------------------------------------------------------------------------------------------------- replay
def replay(check: bool) -> tuple[list[str], list[str]]:
    """For ``make freeze`` / ``make replay-check``: regenerate the holdout personas (intents from the cache),
    replay every judge batch of the holdout qrels and re-run every stored holdout run; compare with the files."""
    if not personas_file().exists():
        return [], []
    from evalkit import evaluate as E
    from evalkit import intents
    from evalkit.judge import replay_problems
    from evalkit.qrels import Qrels

    keys: list[str] = []
    probs: list[str] = []
    catalog = get_catalog()
    committed = {p.persona_id: p for p in load_personas(personas_file())}
    chosen, _ = draw(load_personas(), catalog=catalog)
    wc = intents.make_writer_client()
    intents.add_intents(chosen, wc)
    keys += wc.touched_keys()
    if [p.persona_id for p in chosen] != list(committed):
        probs.append("holdout draw differs (persona ids)")
    for p in chosen:
        c = committed.get(p.persona_id)
        if c is None or c.intent != p.intent or c.latent != p.latent:
            probs.append(f"holdout persona differs: {p.persona_id}")
            break
    pmap = dict(committed)
    recs = list(Qrels.load(E.HOLDOUT.qrels_file).records.values())
    k, pr = replay_problems(catalog, pmap, recs, E.HOLDOUT.qrels_file.name)
    keys += k
    probs += pr[:1]
    client = E.system_client()
    for name in E.stored_systems(E.HOLDOUT):
        client.reset_touched()
        run, _, _ = E.run_system(name, catalog, list(committed.values()))
        keys += client.touched_keys()
        if check:
            stored, _ = E.load_run(name, E.HOLDOUT)
            if any(run[pid]["ids"] != stored[pid]["ids"] for pid in stored):
                probs.append(f"holdout run differs: {name}")
    return keys, probs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("generate", "decide"))
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        if personas_file().exists():
            raise SystemExit(f"{personas_file()} exists; the holdout is drawn once (D28)")
        ps, rep = generate_holdout()
        st = persona_stats(ps)
        print(
            json.dumps(
                {"n": len(ps), "seed": SEED, "dedup": rep, **{k: st[k] for k in ("activity_style", "party")}},
                indent=1,
                ensure_ascii=False,
            )
        )
        return
    doc = write_decision()
    print(json.dumps({k: doc.get(k) for k in ("status", "decision", "verdict")}, indent=1))
    print(f"wrote {paths.RESULTS_DIR / OUT}")


if __name__ == "__main__":
    main()
