"""Iteration 3 (D28): the holdout draw skips duplicates, the simultaneous lower bounds behave as documented, and
the decision follows the pre-registered rule (gate, simultaneous bound, violations, choice, null result)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evalkit import evaluate as E
from evalkit import holdout as H
from evalkit.metrics import simultaneous_lower_bounds
from evalkit.personas import generate
from tests.recsys.factory import TOWNS


# ------------------------------------------------------------------------------------------------- draw
def test_draw_skips_duplicates_of_existing_personas_and_keeps_draw_ids(cat):
    first = generate(cat, TOWNS, n=4, seed=11, id_prefix="H", split="holdout")
    chosen, rep = H.draw(first[:2], catalog=cat, towns=TOWNS, n=3, spare=3, seed=11)
    assert [p.persona_id for p in chosen] == ["H003", "H004", "H005"]
    assert [s["draw"] for s in rep["skipped"]] == ["H001", "H002"]
    assert rep["draws_used"] == 5 and all(p.split == "holdout" for p in chosen)
    with pytest.raises(RuntimeError, match="non-duplicate"):
        H.draw(first, catalog=cat, towns=TOWNS, n=3, spare=1, seed=11)


def test_holdout_stream_does_not_disturb_the_main_generator(cat):
    a = generate(cat, TOWNS, n=6, seed=3)
    b = generate(cat, TOWNS, n=6, seed=3, id_prefix="H", split="holdout")
    assert [p.latent for p in a] == [p.latent for p in b]
    assert [p.persona_id for p in b][:2] == ["H001", "H002"]


def test_identical_intents_are_reported_not_removed(cat):
    ps = generate(cat, TOWNS, n=3, seed=2, id_prefix="H", split="holdout")
    for p, t in zip(ps, ["Easy lake walk", "easy  LAKE walk", "Summit hike"], strict=True):
        p.intent = t
    assert [d["persona"] for d in H.intent_overlap(ps, [])] == ["H002"]


# -------------------------------------------------------------------------------- simultaneous bounds
def test_simultaneous_bounds_properties():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.1, 150)
    one = simultaneous_lower_bounds({"a": x}, n_boot=2000, seed=1)
    lb = one["by_name"]["a"]
    assert lb["lower"] < lb["diff"] and 1.3 < one["c"] < 2.0  # about a one-sided 95% normal quantile
    # identical variants: the family is one variant, so the critical value does not grow
    two = simultaneous_lower_bounds({"a": x, "b": x.copy()}, n_boot=2000, seed=1)
    assert two["c"] == pytest.approx(one["c"])
    # independent variants: a multiplicity correction makes every bound lower
    three = simultaneous_lower_bounds(
        {"a": x, "b": rng.normal(0.05, 0.1, 150), "c": rng.normal(0.05, 0.1, 150)}, n_boot=2000, seed=1
    )
    assert three["c"] > one["c"] + 0.2
    const = simultaneous_lower_bounds({"a": np.full(20, 0.3)}, n_boot=100)
    assert const["by_name"]["a"]["lower"] == pytest.approx(0.3)


# ------------------------------------------------------------------------------------------- decision
def _doc(per: dict[str, float], viol: float = 0.0, judged: float = 1.0, split: str = "holdout") -> dict:
    return {
        "per_persona": {p: {"ndcg_at_10": v} for p, v in per.items()},
        "metrics": {
            split: {
                "ndcg_at_10": {"mean": float(np.mean(list(per.values())))},
                "mrr_at_10": {"mean": 0.5},
                "recall_at_50": {"mean": 0.5},
                "violation_at_10": {"mean": viol},
                "judged_share_at_10": judged,
            }
        },
        "data": {"qrels_sha256": "x"},
    }


def _write(bench, name: str, doc: dict) -> None:
    bench.results_dir.mkdir(parents=True, exist_ok=True)
    (bench.results_dir / f"{name}.json").write_text(json.dumps(doc))


def _setup(eval_tree, holdout_effect: dict[str, float], dev_effect: dict[str, float], viol=None, judged=1.0):
    rng = np.random.default_rng(7)
    dev = {f"P{i:03d}": float(v) for i, v in enumerate(rng.uniform(0.3, 0.9, 200))}
    ho = {f"H{i:03d}": float(v) for i, v in enumerate(rng.uniform(0.3, 0.9, 150))}
    _write(E.CROSSFIT, H.BASELINE, _doc(dev, split="dev"))
    _write(E.HOLDOUT, H.BASELINE, _doc(ho))
    for v in H.CANDIDATES:
        # centred noise: the mean paired difference is exactly the stated effect
        e_dev = rng.normal(0, 0.02, len(dev))
        e_ho = rng.normal(0, 0.02, len(ho))
        e_dev, e_ho = e_dev - e_dev.mean(), e_ho - e_ho.mean()
        _write(
            E.CROSSFIT,
            v,
            _doc({p: x + dev_effect[v] + e for (p, x), e in zip(dev.items(), e_dev, strict=True)}, split="dev"),
        )
        _write(
            E.HOLDOUT,
            v,
            _doc(
                {p: x + holdout_effect[v] + e for (p, x), e in zip(ho.items(), e_ho, strict=True)},
                viol=(viol or {}).get(v, 0.0),
                judged=judged,
            ),
        )


def test_decision_picks_the_largest_simultaneous_lower_bound(eval_tree):
    n, p, d = H.CANDIDATES
    _setup(eval_tree, {n: 0.10, p: 0.20, d: 0.40}, {n: 0.05, p: 0.05, d: -0.05})
    doc = H.decide()
    assert doc["status"] == "complete"
    assert not doc["verdict"][d]["gated"]  # fails the cross-fitted dev gate: never tested on the holdout
    assert doc["verdict"][n]["eligible"] and doc["verdict"][p]["eligible"]
    assert doc["decision"] == {**doc["decision"], "outcome": "switch", "winner": p, "serve": H.NEW_SERVED_NAME}
    assert set(doc["simultaneous"]["by_name"]) == {n, p}


def test_null_result_and_violation_guard(eval_tree):
    n, p, d = H.CANDIDATES
    _setup(eval_tree, {n: 0.0, p: 0.0, d: 0.30}, {n: 0.05, p: 0.05, d: 0.05}, viol={d: 0.02})
    doc = H.decide()
    assert not doc["verdict"][d]["eligible"] and not doc["verdict"][d]["violations_ok"]
    assert doc["decision"]["outcome"] == "null" and doc["decision"]["serve"] == H.BASELINE


def test_decision_refuses_unjudged_lists_and_waits_for_every_candidate(eval_tree):
    n, p, d = H.CANDIDATES
    _setup(eval_tree, {n: 0.2, p: 0.2, d: 0.2}, {n: 0.05, p: 0.05, d: 0.05}, judged=0.9)
    assert H.decide()["status"].startswith("invalid")
    (E.HOLDOUT.results_dir / f"{d}.json").unlink()
    assert H.decide()["status"].startswith("incomplete")
