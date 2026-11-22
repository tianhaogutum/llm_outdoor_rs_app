"""Metrics against hand-computed cases."""

from __future__ import annotations

import math

import pytest

from evalkit import metrics as M


def test_dcg_and_ndcg_hand_computed():
    assert M.dcg([3, 2, 0]) == pytest.approx(7 + 3 / math.log2(3))
    # ranking a, b, c with grades 0, 3, 2; judged set also has d=3
    q = {1: 0, 2: 3, 3: 2, 4: 3}
    dcg = 0 + 7 / math.log2(3) + 3 / 2
    idcg = 7 + 7 / math.log2(3) + 3 / 2
    assert M.ndcg_at_k([1, 2, 3], q, k=3) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k([4, 2, 3], q, k=3) == pytest.approx(1.0)
    # ideal DCG uses the best k of ALL judged pairs, not only the retrieved ones
    assert M.ndcg_at_k([3], q, k=1) == pytest.approx(3 / 7)


def test_ndcg_undefined_and_unjudged():
    assert M.ndcg_at_k([1, 2], {1: 0, 2: 0}) is None
    with pytest.raises(M.UnjudgedError):
        M.ndcg_at_k([1, 99], {1: 2})


def test_mrr():
    q = {1: 1, 2: 2, 3: 3}
    assert M.mrr_at_k([1, 2, 3], q) == 0.5
    assert M.mrr_at_k([3, 1], q) == 1.0
    assert M.mrr_at_k([1], q) == 0.0
    assert M.mrr_at_k([1, 1, 3], q, k=2) == 0.0  # cut at k


def test_recall():
    q = {1: 2, 2: 3, 3: 1, 4: 2}
    assert M.recall_at_k([1, 3, 9], q, k=50) == pytest.approx(1 / 3)
    assert M.recall_at_k([1, 2, 4], q, k=2) == pytest.approx(2 / 3)
    assert M.recall_at_k([1], {1: 1}) is None


def test_violation_rate():
    assert M.violation_rate_at_k([1, 2, 3, 4], lambda r: r % 2 == 0, k=4) == 0.5
    assert M.violation_rate_at_k([], lambda r: True) is None


def test_intra_list_diversity_and_coverage():
    sets = [{"a", "b"}, {"a"}, {"c"}]
    assert M.intra_list_diversity(sets) == pytest.approx((0.5 + 1 + 1) / 3)
    assert M.intra_list_diversity([{"a"}]) is None
    assert M.intra_list_diversity([set(), set()]) == 0.0
    assert M.coverage([[1, 2, 3], [3, 4]], 10, k=2) == pytest.approx(4 / 10)


def test_weighted_kappa_hand_computed():
    assert M.weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert M.weighted_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    # O: (0,0)=1, (0,3)=1, (3,3)=2; sum w*O = 1; E(0,3)=1.5, E(3,0)=0.5 -> sum w*E = 2; kappa = 0.5
    assert M.weighted_kappa([0, 0, 3, 3], [0, 3, 3, 3]) == pytest.approx(0.5)
    assert M.weighted_kappa([], []) is None
    assert M.confusion_matrix([0, 3], [1, 3]) == [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]


def test_bootstrap_ci_and_paired():
    c = M.bootstrap_ci([0.4] * 20, seed=1)
    assert c == {"mean": pytest.approx(0.4), "lo": pytest.approx(0.4), "hi": pytest.approx(0.4), "n": 20}
    x = [i / 10 for i in range(10)]
    a, b = M.bootstrap_ci(x, seed=3), M.bootstrap_ci(x, seed=3)
    assert a == b and a["lo"] < a["mean"] < a["hi"]
    assert M.bootstrap_ci([None, None])["n"] == 0
    p = M.paired_bootstrap(x, [v + 0.1 for v in x], seed=2)
    assert p["diff"] == pytest.approx(0.1) and p["lo"] == pytest.approx(0.1) and p["p_le_0"] == 0.0
    p2 = M.paired_bootstrap([0.1, None, 0.3], [0.2, 0.5, None])
    assert p2["n"] == 1
