"""Cross-fitted dev evaluation (evalkit.crossfit): every dev persona is ranked by the fold copy of the learned
ranker that was trained without it; fold artefacts are deterministic; systems without a ranker are refused."""

from __future__ import annotations

import json

import pytest

from app.recsys.ltr import model_path
from evalkit import crossfit
from evalkit import evaluate as E
from evalkit.intents import add_intents
from evalkit.personas import generate
from tests.recsys.factory import TOWNS


def test_ltr_model_name_is_found_in_nested_configs():
    assert crossfit.ltr_model_name({"pipeline": {}, "ltr": {"name": "v1_noapi"}}) == "v1_noapi"
    assert crossfit.ltr_model_name({"system": "x", "base": {"config": {"ltr": {"name": "m"}}}}) == "m"
    assert crossfit.ltr_model_name({"pipeline": {"ltr_model": None}}) is None


def test_fold_artefacts_are_byte_deterministic(tmp_path):
    obj = {"b": [1.0, None], "a": {"x": 1}}
    crossfit._write_gz(tmp_path / "a.json.gz", obj)
    crossfit._write_gz(tmp_path / "b.json.gz", obj)
    assert (tmp_path / "a.json.gz").read_bytes() == (tmp_path / "b.json.gz").read_bytes()
    assert crossfit._read_gz(tmp_path / "a.json.gz") == obj


def _fake_folds(model: str, dev_ids: list[str], k: int = 5) -> list[set[str]]:
    """Fold files holding the committed model's trees, each with its own held-out personas."""
    base = json.loads(model_path(model).read_text(encoding="utf-8"))
    folds = [set(dev_ids[i::k]) for i in range(k)]
    for i, held in enumerate(folds):
        art = {**base, "meta": {**base["meta"], "held_out": sorted(held), "rounds": base["meta"]["rounds"]}}
        crossfit._write_gz(crossfit.model_file(model, i), art)
    return folds


def test_each_dev_persona_is_ranked_by_the_copy_that_never_saw_it(cat, mock_client, eval_tree, monkeypatch):
    ps = generate(cat, TOWNS, n=15, seed=4)
    add_intents(ps, mock_client)
    dev = [p for p in ps if p.split == "dev"]
    folds = _fake_folds("v1_noapi_nocap", [p.persona_id for p in dev])
    seen: dict[str, str] = {}

    def fake_run_system(name, catalog, personas, depth=E.RUN_DEPTH, **cfg):
        m = cfg["ltr_model"]
        for p in personas:
            seen[p.persona_id] = m.name
        return {p.persona_id: {"ids": []} for p in personas}, {}, {}

    monkeypatch.setattr(E, "run_system", fake_run_system)
    run, _, cfg = crossfit.run("recsys_serve", cat, dev)
    assert set(run) == {p.persona_id for p in dev}
    for k, held in enumerate(folds):
        for pid in held:
            assert seen[pid] == f"v1_noapi_nocap_cf{k}"  # the copy whose training set excluded pid
    assert cfg["crossfit"]["model"] == "v1_noapi_nocap" and cfg["crossfit"]["folds"] == 5
    assert sum(cfg["crossfit"]["fold_sizes"]) == len(dev)


def test_crossfit_refuses_test_personas_and_systems_without_a_ranker(cat, mock_client, eval_tree):
    ps = generate(cat, TOWNS, n=9, seed=4)
    add_intents(ps, mock_client)
    with pytest.raises(ValueError, match="dev personas only"):
        crossfit.run("recsys_serve", cat, ps)
    with pytest.raises(ValueError, match="no learned ranker"):
        crossfit.run("v4_rrf", cat, [p for p in ps if p.split == "dev"])


def test_bench_paths_are_separate(eval_tree):
    assert E.CROSSFIT.runs_dir != E.MAIN.runs_dir and E.HOLDOUT.results_dir != E.MAIN.results_dir
    assert E.CROSSFIT.qrels_file == E.MAIN.qrels_file  # cross-fitted dev holes are ordinary dev judgements
    assert E.HOLDOUT.qrels_file != E.MAIN.qrels_file  # the holdout keeps its own judgements
    assert E.CROSSFIT.splits == ("dev",) and E.HOLDOUT.splits == ("holdout",)
