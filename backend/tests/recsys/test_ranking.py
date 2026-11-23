"""Phase 3c/3d: feature building, the numpy tree-ensemble evaluator and model loading, MMR."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.recsys import Catalog, UserQuery
from app.recsys.features import FEATURES, build_features, parse_intent, stage_arrays
from app.recsys.ltr import FORMAT, MODELS_COMMITTED, TreeEnsemble, flatten_lightgbm_dump, load_model, model_path
from app.recsys.mmr import mmr
from tests.recsys.factory import route_record


# ----------------------------------------------------------------------------- intent parsing
def test_parse_intent():
    a = parse_intent("An easy loop with a lake and a castle, around 6–10 km, no busy roads please")
    assert a.km_target == 8.0 and a.wants_loop and a.wants_easy and not a.wants_mtb
    assert set(a.concepts) == {"lake", "castle"} and a.negated == ("busy_roads",)
    b = parse_intent("A challenging mountain bike ride of about 40 km with a hut, without exposed sections")
    assert b.km_target == 40.0 and b.wants_mtb and "hut" in b.concepts and "alpine" in b.negated
    c = parse_intent(None)
    assert c.km_target is None and c.concepts == () and not c.wants_loop


# ----------------------------------------------------------------------------- features
def test_build_features_values():
    recs = [
        route_record(
            1, lat=48.1, lon=11.0, length_m=8000, ascent_m=150, tags=("Lake", "Castle", "Round trip"), is_loop=True
        ),
        route_record(2, lat=48.0, lon=11.0, length_m=20000, ascent_m=900, tags=("Busy road sections",), is_loop=False),
    ]
    cat = Catalog.from_records(recs)
    q = UserQuery(
        request_id="x",
        user_vector={"difficulty_range": [0, 1], "min_distance_km": 5, "max_distance_km": 15, "preferred_tags": []},
        category="hiking",
        home=(48.0, 11.0),
        max_travel_km=40.0,
        max_distance_km=16.0,
        intent="An easy loop with a lake and a castle, around 8 km, no busy roads",
    )
    from app.recsys.constraints import HardConstraints

    home = HardConstraints.from_query(q).home_distance_km(cat)
    st = stage_arrays(2)
    st["dense"] = np.array([0.7, 0.6])
    X = build_features(cat, q, np.array([0, 1]), st, home)
    f = {n: X[:, j] for j, n in enumerate(FEATURES)}
    assert X.shape == (2, len(FEATURES))
    assert f["home_km"][0] == pytest.approx(11.12, abs=0.05) and f["home_km"][1] == pytest.approx(0.0, abs=1e-6)
    assert f["home_ratio"][0] == pytest.approx(f["home_km"][0] / 40.0)
    assert f["length_cap_ratio"].tolist() == pytest.approx([0.5, 1.25])
    assert f["length_band_pos"].tolist() == pytest.approx([0.3, 1.5])
    assert f["intent_km_gap"].tolist() == pytest.approx([0.0, 1.5])
    assert f["loop_match"].tolist() == [1.0, 0.0] and f["intent_loop"].tolist() == [1.0, 1.0]
    assert f["tag_cover"].tolist() == [1.0, 0.0] and f["tag_hits"].tolist() == [2.0, 0.0]
    assert f["neg_hits"].tolist() == [0.0, 1.0]
    assert f["ascent_per_km"][1] == pytest.approx(45.0)
    assert f["dense"].tolist() == [0.7, 0.6] and np.isnan(f["bm25"]).all() and np.isnan(f["rank_api"]).all()
    assert f["rule_distance"][0] == 1.0  # 8 km inside the questionnaire's 5-15 km band
    # no intent: intent-derived features are missing or 0, never invented
    q2 = UserQuery(request_id="y", user_vector=q.user_vector, category="hiking")
    X2 = build_features(cat, q2, np.array([0]), stage_arrays(1), None)
    f2 = {n: X2[0, j] for j, n in enumerate(FEATURES)}
    assert np.isnan(f2["home_km"]) and np.isnan(f2["intent_km_gap"]) and np.isnan(f2["tag_cover"])
    assert f2["intent_loop"] == 0.0


# ----------------------------------------------------------------------------- tree ensemble
def _dump() -> dict:
    """Two hand-made trees in LightGBM's dump_model() layout over features [a, b]."""
    t1 = {
        "split_feature": 0,
        "threshold": 0.5,
        "decision_type": "<=",
        "default_left": True,
        "missing_type": "NaN",
        "internal_value": 0.1,
        "left_child": {"leaf_value": -1.0},
        "right_child": {
            "split_feature": 1,
            "threshold": 2.0,
            "decision_type": "<=",
            "default_left": False,
            "missing_type": "None",
            "internal_value": 0.6,
            "left_child": {"leaf_value": 0.5},
            "right_child": {"leaf_value": 2.0},
        },
    }
    t2 = {
        "split_feature": 1,
        "threshold": 0.0,
        "decision_type": "<=",
        "default_left": False,
        "missing_type": "Zero",
        "internal_value": 0.0,
        "left_child": {"leaf_value": 0.25},
        "right_child": {"leaf_value": -0.25},
    }
    return {"tree_info": [{"tree_structure": t1}, {"tree_structure": t2}]}


def _ensemble(ref=None) -> TreeEnsemble:
    return TreeEnsemble.from_dict(
        {
            "format": FORMAT,
            "feature_names": ["a", "b"],
            "trees": flatten_lightgbm_dump(_dump()),
            "meta": {},
            "reference": ref or {},
        },
        name="hand",
    )


def test_tree_ensemble_predictions_and_missing_values():
    m = _ensemble()
    X = np.array([[0.0, 5.0], [1.0, 1.0], [1.0, 3.0], [np.nan, 3.0], [1.0, np.nan], [1.0, 0.0]])
    # tree 1: a<=0.5 -> -1 ; else b<=2 -> 0.5 else 2 ; NaN a -> default left ; NaN b (missing None) -> 0 -> left
    # tree 2: b<=0 -> 0.25 else -0.25 ; missing Zero: b == 0 or NaN -> default (right)
    expected = [-1 - 0.25, 0.5 - 0.25, 2 - 0.25, -1 - 0.25, 0.5 - 0.25, 0.5 - 0.25]
    assert m.predict(X).tolist() == pytest.approx(expected)
    pred, C, bias = m.predict_contrib(X)
    assert bias == pytest.approx(0.1)
    assert (bias + C.sum(axis=1)).tolist() == pytest.approx(pred.tolist())
    # row 2: root 0.1 -> right node 0.6 (credit a +0.5) -> leaf 2.0 (credit b +1.4); tree 2: 0 -> -0.25 (credit b)
    assert C[2].tolist() == pytest.approx([0.5, 1.4 - 0.25])
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 3)))


def test_reference_check_detects_a_broken_artefact():
    good = _ensemble({"X": [[1.0, 3.0], [None, 1.0]], "pred": [1.75, -1.25]})
    assert good.check_reference() == pytest.approx(0.0)
    bad = _ensemble({"X": [[1.0, 3.0]], "pred": [0.0]})
    with pytest.raises(ValueError):
        bad.check_reference()


@pytest.mark.parametrize("name", MODELS_COMMITTED)
def test_committed_models_load_without_lightgbm(name):
    path = model_path(name)
    assert path.exists(), f"committed model artefact {path.name} missing"
    m = load_model(name)  # runs the reference check against LightGBM's own predictions
    assert m.n_trees > 0 and set(m.feature_names) <= set(FEATURES)
    meta = json.loads(path.read_text())["meta"]
    assert meta["trained_on"] == "dev personas only" and meta["rounds"] == m.n_trees
    X = np.array(json.loads(path.read_text())["reference"]["X"], dtype=float)
    pred, C, bias = m.predict_contrib(X)
    assert np.allclose(bias + C.sum(axis=1), pred)


# ----------------------------------------------------------------------------- MMR
def test_mmr_lambda_one_keeps_order_and_zero_diversifies():
    rel = [0.9, 0.8, 0.7]
    S = np.array([[1.0, 0.95, 0.1], [0.95, 1.0, 0.2], [0.1, 0.2, 1.0]])
    order, det = mmr(rel, S, 1.0)
    assert order == [0, 1, 2] and all(d["penalty"] == 0 for d in det)
    order, det = mmr(rel, S, 0.5, normalise_sim=False)
    # step 2: route 1 -> 0.5*0.5 - 0.5*0.95 = -0.225 ; route 2 -> 0 - 0.5*0.1 = -0.05 -> route 2 first
    assert order == [0, 2, 1]
    assert det[1]["mmr_score"] == pytest.approx(-0.05) and det[1]["max_similarity"] == pytest.approx(0.1)
    assert det[0]["relevance"] == 1.0 and det[2]["relevance"] == pytest.approx(0.5)
    assert mmr([], np.zeros((0, 0)), 0.5) == ([], [])
    with pytest.raises(ValueError):
        mmr(rel, S, 1.5)


def test_mmr_negative_similarity_is_clipped_and_ties_keep_input_order():
    S = np.array([[1.0, -0.5, -0.5], [-0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]])
    order, _ = mmr([1.0, 0.5, 0.5], S, 0.3, normalise_sim=False)
    assert order == [0, 1, 2]


def test_mmr_normalises_similarity_over_the_candidates():
    # all pairs alike (0.80-0.90): raw cosines barely separate them, normalised ones do
    rel = [1.0, 0.9, 0.8]
    S = np.array([[1.0, 0.90, 0.80], [0.90, 1.0, 0.85], [0.80, 0.85, 1.0]])
    assert mmr(rel, S, 0.7, normalise_sim=False)[0] == [0, 1, 2]
    order, det = mmr(rel, S, 0.7)
    # normalised: sim(0,1) = 1, sim(0,2) = 0 -> step 2: route 1 0.7*0.5 - 0.3*1 = 0.05 ; route 2 0 - 0 = 0.0
    assert order == [0, 1, 2] and det[1]["mmr_score"] == pytest.approx(0.05)
    order, _ = mmr(rel, S, 0.5)
    assert order == [0, 2, 1]
