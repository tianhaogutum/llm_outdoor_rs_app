"""Cross-fitted dev evaluation for systems with a learned ranker (docs/EVAL.md section 13).

    make eval SYSTEM=recsys_serve CROSSFIT=1     # python -m evalkit.evaluate --system recsys_serve --crossfit

Every learned ranker is trained on the dev personas' judgements (docs/RECSYS.md section 4), so the system-level
dev row of any system that contains one is **in-sample**: the ranker has seen exactly these users' grades. That
row is optimistic for the system itself and, worse, for any comparison between a system that keeps the ranker's
ordering and one that overrides it (D21, D27).

Cross-fitting removes that. The system's ranker is retrained ``K`` times, exactly like the committed model (same
pinned dev labels ``eval/qrels/ltr_train_v1.jsonl``, same feature set, same LightGBM parameters and the same
number of boosting rounds), each time without one fold of dev personas; the folds are those of the ranker's own
grouped cross-validation (``evalkit.ltr._folds``, grouped by persona). Each dev persona is then ranked by the copy
that never saw its labels. Everything else in the system is unchanged. Results go to
``eval/results/crossfit/<system>.json`` (dev only), runs to ``eval/runs/crossfit/``; new top-10 routes are judged
into the main qrels like any dev hole.

The fold models are written once to ``eval/crossfit/ltr_<model>_cf<k>.json.gz`` (numpy-evaluable, like the served
artefacts) so that rescoring, ``make eval-all`` and ``make replay-check`` never need LightGBM; training them does
(the Makefile sets the macOS OpenMP path). One leak remains and is accepted: the number of rounds was chosen by
cross-validation over all dev personas, so each fold copy inherits a hyper-parameter that saw its held-out fold.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from typing import Any, Optional

import numpy as np

from app.recsys import Catalog
from evalkit import paths
from evalkit.personas import Persona, load_personas

FORMAT_NOTE = "cross-fitted copy of a committed learned ranker (evalkit.crossfit)"


def models_dir():
    return paths.EVAL_DIR / "crossfit"


def ltr_model_name(config: dict[str, Any]) -> Optional[str]:
    """Name of the learned-ranker model a system's stored config uses (searched recursively, so wrappers such as
    the chat or empty-slot systems are found too); None if the system has no learned ranker."""
    if isinstance(config, dict):
        ltr = config.get("ltr")
        if isinstance(ltr, dict) and ltr.get("name"):
            return str(ltr["name"])
        for v in config.values():
            n = ltr_model_name(v)
            if n:
                return n
    return None


def has_learned_ranker(name: str, catalog: Catalog) -> bool:
    from evalkit.evaluate import make_system

    return ltr_model_name(make_system(name, catalog).config()) is not None


def model_file(model: str, k: int):
    return models_dir() / f"ltr_{model}_cf{k}.json.gz"


def _write_gz(path, obj: dict[str, Any]) -> None:
    raw = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)


def _read_gz(path) -> dict[str, Any]:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _drop_of(base) -> tuple[str, ...]:
    from evalkit import ltr as LT

    variant = base.meta.get("variant")
    drop = LT.VARIANTS.get(variant, LT.ROBUST_VARIANTS.get(variant))
    if drop is None:
        raise ValueError(f"model {base.name!r}: unknown training variant {variant!r}")
    return tuple(drop)


def train_fold_models(model: str) -> None:
    """Train and write the ``K`` fold copies of committed model ``model`` (needs LightGBM)."""
    import lightgbm as lgb

    from app.recsys.features import FEATURES
    from app.recsys.ltr import FORMAT, flatten_lightgbm_dump, load_model
    from evalkit import ltr as LT

    base = load_model(model)
    drop = _drop_of(base)
    cols = LT._cols(drop)
    if [FEATURES[c] for c in cols] != list(base.feature_names):
        raise ValueError(f"model {model!r}: feature set does not match its training variant")
    rounds = int(base.meta["rounds"])
    with_api = "rank_api" in base.feature_names
    dev = [p for p in load_personas() if p.split == "dev"]
    labels = LT.train_labels()
    if base.meta.get("train_labels_sha256") not in (None, labels.sha256()):
        raise ValueError(f"model {model!r} was trained on other labels than {LT.TRAIN_LABELS.name}")
    ds = LT.dataset(dev, LT.candidate_matrix(dev, with_api=with_api), labels)
    for k, held in enumerate(LT._folds(ds["pids"])):
        booster, _ = LT._train(lgb, LT._subset(ds, set(ds["pids"]) - held), cols, rounds)
        ref = [
            [None if np.isnan(v) else round(float(v), 9) for v in row]
            for row in ds["X"][np.array([p in held for p in ds["pids"]])][:32][:, cols]
        ]
        ref_X = np.array([[np.nan if v is None else v for v in row] for row in ref], dtype=np.float64)
        art = {
            "format": FORMAT,
            "feature_names": list(base.feature_names),
            "trees": flatten_lightgbm_dump(booster.dump_model()),
            "meta": {
                "note": FORMAT_NOTE,
                "crossfit_of": model,
                "fold": k,
                "folds": LT.N_FOLDS,
                "held_out": sorted(held),
                "rounds": rounds,
                "variant": base.meta.get("variant"),
                "params": LT.PARAMS,
                "lightgbm": lgb.__version__,
                "train_labels_sha256": labels.sha256(),
                "trained_on": "dev personas outside this fold",
            },
            "reference": {
                "X": ref,
                "pred": [float(v) for v in booster.predict(ref_X)],
            },
        }
        _write_gz(model_file(model, k), art)


def fold_models(model: str, *, train: bool = True) -> list[tuple[set[str], Any]]:
    """[(held-out dev persona ids, TreeEnsemble)] for each fold; trains and writes them first if missing."""
    from app.recsys.ltr import TreeEnsemble
    from evalkit import ltr as LT

    files = [model_file(model, k) for k in range(LT.N_FOLDS)]
    if not all(f.exists() for f in files):
        if not train:
            raise FileNotFoundError(f"cross-fitted models of {model!r} missing in {models_dir()}")
        train_fold_models(model)
    out = []
    for k, f in enumerate(files):
        d = _read_gz(f)
        m = TreeEnsemble.from_dict(d, name=f"{model}_cf{k}")
        m.check_reference(atol=1e-9)
        out.append((set(d["meta"]["held_out"]), m))
    return out


def assignment(models: list[tuple[set[str], Any]], personas: list[Persona]) -> dict[str, int]:
    """Fold index per persona: the copy whose training set excluded it. A dev persona without any training label
    (none at present) was in no copy's training data; it gets fold 0."""
    return {p.persona_id: next((k for k, (held, _) in enumerate(models) if p.persona_id in held), 0) for p in personas}


def config(name: str, catalog: Catalog) -> dict[str, Any]:
    """The system's own config plus what the cross-fitting used (model, folds, rounds, file digests)."""
    from evalkit.evaluate import make_system

    cfg = make_system(name, catalog).config()
    model = ltr_model_name(cfg)
    if model is None:
        raise ValueError(f"{name}: no learned ranker, so there is nothing to cross-fit")
    models = fold_models(model, train=False)
    cfg["crossfit"] = {
        "model": model,
        "folds": len(models),
        "fold_sizes": [len(h) for h, _ in models],
        "rounds": models[0][1].meta["rounds"],
        "files": {
            model_file(model, k).name: hashlib.sha256(model_file(model, k).read_bytes()).hexdigest()
            for k in range(len(models))
        },
        "note": "each dev persona ranked by the copy trained without its fold (docs/EVAL.md section 13)",
    }
    return cfg


def run(name: str, catalog: Catalog, personas: list[Persona]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run ``name`` on dev ``personas``, each with the fold copy of its learned ranker trained without it."""
    from evalkit.evaluate import make_system, run_system

    if any(p.split != "dev" for p in personas):
        raise ValueError("cross-fitting is defined for dev personas only (the rankers never saw test or holdout)")
    model = ltr_model_name(make_system(name, catalog).config())
    if model is None:
        raise ValueError(f"{name}: no learned ranker, so there is nothing to cross-fit")
    models = fold_models(model)
    fold = assignment(models, personas)
    out_run: dict[str, Any] = {}
    out_t: dict[str, Any] = {}
    for k, (_, m) in enumerate(models):
        ps = [p for p in personas if fold[p.persona_id] == k]
        if ps:
            r, t, _ = run_system(name, catalog, ps, ltr_model=m)
            out_run.update(r)
            out_t.update(t)
    return out_run, out_t, config(name, catalog)


def replay(check: bool) -> tuple[list[str], list[str]]:
    """For ``make replay-check``: re-run every stored cross-fitted run from the committed fold models (query
    embeddings from the frozen layer) and compare its rankings with the stored run."""
    from app.recsys import get_catalog
    from evalkit import evaluate as E

    names = E.stored_systems(E.CROSSFIT)
    if not names:
        return [], []
    catalog = get_catalog()
    dev = E.CROSSFIT.personas()
    client = E.system_client()
    client.reset_touched()
    probs = []
    for n in names:
        got, _, _ = run(n, catalog, dev)
        if check:
            stored, _ = E.load_run(n, E.CROSSFIT)
            if any(got[pid]["ids"] != stored[pid]["ids"] for pid in stored):
                probs.append(f"cross-fitted run differs: {n}")
    return client.touched_keys(), probs
