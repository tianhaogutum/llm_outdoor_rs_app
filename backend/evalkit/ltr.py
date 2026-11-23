"""Learned ranker (Phase 3c ii): training data, LambdaMART training, overfitting analysis. **Dev only.**

    cd backend && python -m evalkit.ltr sample   # judge a deeper random sample of dev candidates (costs money)
    cd backend && python -m evalkit.ltr train    # make ltr: CV, final models, analysis (needs LightGBM)

Candidates are the ``v4_rrf`` fused top 50 per persona (what the ranker re-ranks at serving time).
Labels are the judged grades (0-3) of those candidates. The pooled qrels alone are shallow and biased
towards routes some system already put in a top 10, so ``sample`` adds a seeded uniform random sample of
each dev persona's *unjudged* candidates (``SAMPLE_PER_PERSONA``), judged by the production judge and
marked ``"sample": "ltr-train/v1"`` in the qrels. Unjudged candidates are never treated as negatives;
they are left out of training.

Model selection uses only dev: 5-fold cross-validation grouped by persona (a persona's rows are never
split across folds), number of boosting rounds = the best mean fold NDCG@10. Test personas appear only
in the *reporting* of the frozen models (``test`` blocks), never in training, tuning or early stopping.

Outputs: ``backend/app/recsys/artifacts/ltr_<name>.json`` (numpy-evaluable model + training metadata)
and ``eval/results/_ltr.json`` (data, CV, importance, ablations, dev-vs-test).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any

import numpy as np

from app.llm import run_sync
from app.recsys import get_catalog
from app.recsys.features import FEATURES, GROUPS
from app.recsys.ltr import FORMAT, TreeEnsemble, flatten_lightgbm_dump, model_path
from evalkit import metrics as M
from evalkit import paths
from evalkit.evaluate import dumps, make_system
from evalkit.personas import Persona, load_personas
from evalkit.qrels import Qrels, fill_holes

SAMPLE_PER_PERSONA = 8
SAMPLE_TAG = "ltr-train/v1"
SEED = 20261122
N_FOLDS = 5
MAX_ROUNDS = 400
PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}
OUT = "_ltr.json"
# Training labels are pinned: the qrels as they were when the models were trained (all pairs judged up to
# v5_rank_api plus the random sample). Later evaluations of the learned ranker itself add judged pairs to
# the qrels; reading those would make the training data depend on the model (and change the model on
# every retrain). ``sample`` writes this file; ``train`` reads only it.
TRAIN_LABELS = paths.QRELS_DIR / "ltr_train_v1.jsonl"


def train_labels() -> Qrels:
    recs = [json.loads(line) for line in TRAIN_LABELS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Qrels(recs)


def write_train_labels(qrels: Qrels) -> None:
    lines = [
        json.dumps({"persona_id": pid, "route_id": rid, "grade": r["grade"]}, sort_keys=True)
        for (pid, rid), r in sorted(qrels.records.items())
    ]
    TRAIN_LABELS.write_text("\n".join(lines) + "\n", encoding="utf-8")


CANDIDATE_SYSTEM = "v4_rrf"

# model variants: name -> features left out (ablations and the with/without ranker-API comparison)
VARIANTS: dict[str, tuple[str, ...]] = {
    "full": (),
    "no_rank_api": GROUPS["rank_api"],
    "no_slot_ratios": GROUPS["slot_ratios"],
    "no_home_distance": GROUPS["home_distance"],
    "no_intent_numbers": GROUPS["intent_numbers"],
    "no_rule_subscores": GROUPS["rule_subscores"],
    "no_suspicious": GROUPS["slot_ratios"] + GROUPS["home_distance"] + GROUPS["intent_numbers"],
    "retrieval_and_api_only": tuple(f for f in FEATURES if f not in GROUPS["retrieval"] + GROUPS["rank_api"]),
}
# committed models: artefact name -> variant
MODELS = {"v1": "full", "v1_noapi": "no_rank_api"}

# Robustness variants of the default ranker (no ranker-API feature), trained by ``train-robust`` without
# touching the committed v1 models: the simulator-shaped features left out (docs/DECISIONS.md D15, D19).
ROBUST_VARIANTS: dict[str, tuple[str, ...]] = {
    "noapi_no_length_cap": GROUPS["rank_api"] + ("length_cap_ratio",),
    "noapi_no_suspicious": GROUPS["rank_api"]
    + tuple(dict.fromkeys(GROUPS["slot_ratios"] + GROUPS["home_distance"] + GROUPS["intent_numbers"])),
}
ROBUST_MODELS = {"v1_noapi_nocap": "noapi_no_length_cap", "v1_noapi_nosusp": "noapi_no_suspicious"}
ROBUST_OUT = "_ltr_robust.json"


# --------------------------------------------------------------------------- data
def candidate_matrix(personas: list[Persona], *, with_api: bool = True) -> dict[str, tuple[list[int], np.ndarray]]:
    """Per persona: (candidate route ids, feature matrix over FEATURES) from the v4_rrf candidate stage."""
    catalog = get_catalog()
    pipe = make_system(CANDIDATE_SYSTEM, catalog)

    async def go():
        import asyncio

        sem = asyncio.Semaphore(8)

        async def one(p):
            async with sem:
                return p.persona_id, await pipe.afeatures(p.to_query(), with_api=with_api)

        return await asyncio.gather(*(one(p) for p in personas))

    return dict(run_sync(go()))


def dataset(personas: list[Persona], cm, qrels: Qrels) -> dict[str, Any]:
    qp = qrels.by_persona()
    X, y, groups, pids, rids = [], [], [], [], []
    for p in personas:
        ids, F = cm[p.persona_id]
        g = qp.get(p.persona_id, {})
        rows = [j for j, r in enumerate(ids) if r in g]
        if not rows:
            continue
        X.append(F[rows])
        y += [g[ids[j]] for j in rows]
        groups.append(len(rows))
        pids += [p.persona_id] * len(rows)
        rids += [ids[j] for j in rows]
    return {
        "X": np.vstack(X),
        "y": np.asarray(y, dtype=np.int64),
        "groups": groups,
        "pids": pids,
        "rids": rids,
    }


def _cols(drop: tuple[str, ...]) -> list[int]:
    return [j for j, f in enumerate(FEATURES) if f not in drop]


def _ndcg_rows(scores: np.ndarray, y: np.ndarray, groups: list[int], k: int = 10) -> list[float]:
    """Per-group NDCG@k over the judged rows (ideal from the group's own labels); None-free list."""
    out, s = [], 0
    for n in groups:
        sc, lab = scores[s : s + n], y[s : s + n]
        s += n
        order = np.argsort(-sc, kind="stable")
        ideal = M.dcg(sorted(lab.tolist(), reverse=True)[:k])
        if ideal > 0:
            out.append(M.dcg(lab[order][:k].tolist()) / ideal)
    return out


def _subset(ds: dict[str, Any], keep_pids: set[str]) -> dict[str, Any]:
    mask = np.array([p in keep_pids for p in ds["pids"]])
    groups, s = [], 0
    gp = []
    for n in ds["groups"]:
        if ds["pids"][s] in keep_pids:
            groups.append(n)
            gp.append(ds["pids"][s])
        s += n
    return {"X": ds["X"][mask], "y": ds["y"][mask], "groups": groups, "pids": [p for p in ds["pids"] if p in keep_pids]}


def _folds(pids: list[str]) -> list[set[str]]:
    uniq = sorted(set(pids))
    rng = np.random.default_rng(SEED)
    perm = [uniq[i] for i in rng.permutation(len(uniq))]
    return [set(perm[i::N_FOLDS]) for i in range(N_FOLDS)]


def _train(lgb, ds, cols, rounds, valid=None):
    dtrain = lgb.Dataset(
        ds["X"][:, cols],
        label=ds["y"],
        group=ds["groups"],
        feature_name=[FEATURES[c] for c in cols],
        free_raw_data=False,
    )
    ev: dict = {}
    kw: dict[str, Any] = {}
    if valid is not None:
        dval = lgb.Dataset(valid["X"][:, cols], label=valid["y"], group=valid["groups"], reference=dtrain)
        kw = {"valid_sets": [dval], "valid_names": ["val"], "callbacks": [lgb.record_evaluation(ev)]}
    booster = lgb.train(PARAMS, dtrain, num_boost_round=rounds, **kw)
    return booster, ev


def cross_validate(lgb, ds, drop: tuple[str, ...]) -> dict[str, Any]:
    cols = _cols(drop)
    folds = _folds(ds["pids"])
    curves = []
    per_fold = []
    for f in folds:
        tr = _subset(ds, set(ds["pids"]) - f)
        va = _subset(ds, f)
        _, ev = _train(lgb, tr, cols, MAX_ROUNDS, valid=va)
        curves.append(np.asarray(ev["val"]["ndcg@10"]))
    mean_curve = np.mean(curves, axis=0)
    best = int(np.argmax(mean_curve)) + 1
    # fold scores at the chosen round count (our NDCG@10 over judged rows, same definition as below)
    for f in folds:
        tr = _subset(ds, set(ds["pids"]) - f)
        va = _subset(ds, f)
        b, _ = _train(lgb, tr, cols, best)
        per_fold.append(_ndcg_rows(b.predict(va["X"][:, cols]), va["y"], va["groups"]))
    allv = [v for fold in per_fold for v in fold]
    return {
        "best_rounds": best,
        "cv_ndcg_at_10": M.bootstrap_ci(allv, n_boot=1000, seed=SEED),
        "cv_fold_means": [float(np.mean(f)) for f in per_fold],
        "lgb_curve_at": {str(r): float(mean_curve[r - 1]) for r in (10, 50, 100, 200, best) if r <= len(mean_curve)},
    }


def _baseline_scores(ds, name: str) -> list[float]:
    col = FEATURES.index(name)
    s = ds["X"][:, col]
    if name.endswith("_rank"):
        s = -s
    return _ndcg_rows(np.nan_to_num(s, nan=-1e9), ds["y"], ds["groups"])


def export_model(booster, cols, name: str, meta: dict[str, Any], ref_X: np.ndarray) -> TreeEnsemble:
    """Write the numpy-evaluable artefact; the reference rows carry LightGBM's own predictions."""
    ref = [[None if np.isnan(v) else round(float(v), 9) for v in row] for row in ref_X]
    Xr = np.array([[np.nan if v is None else v for v in row] for row in ref], dtype=np.float64)
    art = {
        "format": FORMAT,
        "feature_names": [FEATURES[c] for c in cols],
        "trees": flatten_lightgbm_dump(booster.dump_model()),
        "meta": meta,
        "reference": {"X": ref, "pred": [float(v) for v in booster.predict(Xr)]},
    }
    path = model_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    m = TreeEnsemble.load(path)
    m.check_reference(atol=1e-9)
    return m


# --------------------------------------------------------------------------- commands
def cmd_sample() -> None:
    catalog = get_catalog()
    personas = [p for p in load_personas() if p.split == "dev"]
    qrels = Qrels.load()
    cm = candidate_matrix(personas, with_api=False)
    qp = qrels.by_persona()
    pairs = []
    for p in personas:
        ids, _ = cm[p.persona_id]
        un = sorted(r for r in ids if r not in qp.get(p.persona_id, {}))
        h = int(hashlib.sha256(f"{SEED}:{p.persona_id}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        pick = (
            [un[i] for i in sorted(rng.choice(len(un), min(SAMPLE_PER_PERSONA, len(un)), replace=False))] if un else []
        )
        pairs += [(p.persona_id, r) for r in pick]
    print(f"[ltr] sampling {len(pairs)} unjudged dev candidates", file=sys.stderr)
    before = set(qrels.records)
    rep = fill_holes(qrels, pairs, {p.persona_id: p for p in personas}, catalog)
    for k in set(qrels.records) - before:
        qrels.records[k]["sample"] = SAMPLE_TAG
    qrels.save()
    write_train_labels(qrels)
    print(json.dumps(rep, indent=1))


def cmd_train() -> None:
    import lightgbm as lgb

    allp = load_personas()
    dev = [p for p in allp if p.split == "dev"]
    test = [p for p in allp if p.split == "test"]
    qrels = train_labels()
    cm_dev = candidate_matrix(dev)
    ds = dataset(dev, cm_dev, qrels)
    n_sample = sum(1 for r in Qrels.load().records.values() if r.get("sample") == SAMPLE_TAG)
    lab = np.bincount(ds["y"], minlength=4)
    out: dict[str, Any] = {
        "schema": "evalkit.ltr/v1",
        "candidates": f"{CANDIDATE_SYSTEM} fused top 50",
        "params": PARAMS,
        "folds": N_FOLDS,
        "max_rounds": MAX_ROUNDS,
        "train_labels": {"file": TRAIN_LABELS.name, "pairs": len(qrels), "sha256": qrels.sha256()},
        "data": {
            "split": "dev",
            "personas": len(ds["groups"]),
            "rows": int(len(ds["y"])),
            "rows_per_persona_median": float(np.median(ds["groups"])),
            "labels": {str(g): int(c) for g, c in enumerate(lab)},
            "candidates_judged_share": float(len(ds["y"]) / sum(len(v[0]) for v in cm_dev.values())),
            "random_sample_pairs": n_sample,
            "sample_per_persona": SAMPLE_PER_PERSONA,
        },
        "features": list(FEATURES),
        "groups": {k: list(v) for k, v in GROUPS.items()},
    }
    # single-feature baselines on the same rows
    out["single_feature_dev"] = {
        f: float(np.mean(_baseline_scores(ds, f))) for f in ("fused_rank", "bm25", "dense", "rank_api", "rule_distance")
    }
    # test candidates (reporting only)
    cm_test = candidate_matrix(test)
    ds_test = dataset(test, cm_test, qrels)
    out["data"]["test_rows"] = int(len(ds_test["y"]))
    out["single_feature_test"] = {
        f: float(np.mean(_baseline_scores(ds_test, f)))
        for f in ("fused_rank", "bm25", "dense", "rank_api", "rule_distance")
    }
    variants: dict[str, Any] = {}
    finals: dict[str, Any] = {}
    for vname, drop in VARIANTS.items():
        cv = cross_validate(lgb, ds, drop)
        cols = _cols(drop)
        booster, _ = _train(lgb, ds, cols, cv["best_rounds"])
        finals[vname] = (booster, cols, cv)
        train_nd = _ndcg_rows(booster.predict(ds["X"][:, cols]), ds["y"], ds["groups"])
        test_nd = _ndcg_rows(booster.predict(ds_test["X"][:, cols]), ds_test["y"], ds_test["groups"])
        variants[vname] = {
            "dropped": list(drop),
            **cv,
            "train_ndcg_at_10": float(np.mean(train_nd)),
            "test_ndcg_at_10": M.bootstrap_ci(test_nd, n_boot=1000, seed=SEED),
        }
        print(
            f"[ltr] {vname}: rounds {cv['best_rounds']} train {variants[vname]['train_ndcg_at_10']:.3f} "
            f"cv {cv['cv_ndcg_at_10']['mean']:.3f} test {variants[vname]['test_ndcg_at_10']['mean']:.3f}",
            file=sys.stderr,
        )
    out["variants"] = variants
    # committed models + importance
    ref_rows = ds["X"][:: max(1, len(ds["y"]) // 64)][:64]
    models = {}
    for name, vname in MODELS.items():
        booster, cols, cv = finals[vname]
        meta = {
            "trained_on": "dev personas only",
            "variant": vname,
            "rounds": cv["best_rounds"],
            "rows": int(len(ds["y"])),
            "personas": len(ds["groups"]),
            "lightgbm": lgb.__version__,
            "params": PARAMS,
            "train_labels_sha256": qrels.sha256(),
            "candidates": f"{CANDIDATE_SYSTEM} fused top 50",
        }
        m = export_model(booster, cols, name, meta, ref_rows[:, cols])
        gain = booster.feature_importance("gain")
        split = booster.feature_importance("split")
        _, C, _ = m.predict_contrib(ds["X"][:, cols])
        tot = float(gain.sum()) or 1.0
        imp = sorted(
            (
                {
                    "feature": FEATURES[c],
                    "gain_share": float(gain[j] / tot),
                    "splits": int(split[j]),
                    "mean_abs_contribution": float(np.mean(np.abs(C[:, j]))),
                }
                for j, c in enumerate(cols)
            ),
            key=lambda d: -d["gain_share"],
        )
        models[name] = {"variant": vname, "file": model_path(name).name, "trees": m.n_trees, "importance": imp}
    out["models"] = models
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS_DIR / OUT).write_text(dumps(out), encoding="utf-8")
    print(f"wrote {paths.RESULTS_DIR / OUT}")


def cmd_train_robust() -> None:
    """Train and export the robustness variants of the default ranker (same data, CV and parameters as
    ``train``; no ranker-API call is needed because every variant drops that feature)."""
    import lightgbm as lgb

    allp = load_personas()
    dev = [p for p in allp if p.split == "dev"]
    test = [p for p in allp if p.split == "test"]
    qrels = train_labels()
    ds = dataset(dev, candidate_matrix(dev, with_api=False), qrels)
    ds_test = dataset(test, candidate_matrix(test, with_api=False), qrels)
    ref_rows = ds["X"][:: max(1, len(ds["y"]) // 64)][:64]
    out: dict[str, Any] = {
        "schema": "evalkit.ltr_robust/v1",
        "base_model": "v1_noapi",
        "train_labels": {"file": TRAIN_LABELS.name, "pairs": len(qrels), "sha256": qrels.sha256()},
        "rows": int(len(ds["y"])),
        "personas": len(ds["groups"]),
        "variants": {},
    }
    for name, vname in ROBUST_MODELS.items():
        drop = ROBUST_VARIANTS[vname]
        cv = cross_validate(lgb, ds, drop)
        cols = _cols(drop)
        booster, _ = _train(lgb, ds, cols, cv["best_rounds"])
        meta = {
            "trained_on": "dev personas only",
            "variant": vname,
            "dropped": list(drop),
            "rounds": cv["best_rounds"],
            "rows": int(len(ds["y"])),
            "personas": len(ds["groups"]),
            "lightgbm": lgb.__version__,
            "params": PARAMS,
            "train_labels_sha256": qrels.sha256(),
            "candidates": f"{CANDIDATE_SYSTEM} fused top 50",
        }
        m = export_model(booster, cols, name, meta, ref_rows[:, cols])
        gain = booster.feature_importance("gain")
        tot = float(gain.sum()) or 1.0
        imp = sorted(
            ({"feature": FEATURES[c], "gain_share": float(gain[j] / tot)} for j, c in enumerate(cols)),
            key=lambda d: -d["gain_share"],
        )
        test_nd = _ndcg_rows(booster.predict(ds_test["X"][:, cols]), ds_test["y"], ds_test["groups"])
        out["variants"][vname] = {
            "model": name,
            "dropped": list(drop),
            **cv,
            "train_ndcg_at_10": float(np.mean(_ndcg_rows(booster.predict(ds["X"][:, cols]), ds["y"], ds["groups"]))),
            "test_ndcg_at_10": M.bootstrap_ci(test_nd, n_boot=1000, seed=SEED),
            "trees": m.n_trees,
            "importance_top": imp[:8],
        }
        print(f"[ltr] {name}: rounds {cv['best_rounds']} cv {cv['cv_ndcg_at_10']['mean']:.3f}", file=sys.stderr)
    (paths.RESULTS_DIR / ROBUST_OUT).write_text(dumps(out), encoding="utf-8")
    print(f"wrote {paths.RESULTS_DIR / ROBUST_OUT}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["sample", "train", "train-robust"])
    args = ap.parse_args(argv)
    {"sample": cmd_sample, "train": cmd_train, "train-robust": cmd_train_robust}[args.cmd]()


if __name__ == "__main__":
    main()
