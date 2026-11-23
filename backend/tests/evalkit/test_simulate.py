"""Phase 5 simulation: the per-round metric, the behaviour model, and a deterministic end-to-end run."""

from __future__ import annotations

import pytest

from evalkit import judge as J
from evalkit import simulate as S
from evalkit.intents import add_intents
from evalkit.metrics import dcg
from evalkit.personas import generate, write_personas
from evalkit.qrels import Qrels
from tests.evalkit.conftest import judge_responder
from tests.recsys.factory import TOWNS


def test_round_ndcg_removes_consumed_routes_from_gain_and_ideal():
    grades = {1: 3, 2: 2, 3: 1, 4: 0, 5: 2}
    # nothing consumed: ordinary NDCG@10 over the judged pairs
    assert S.round_ndcg([2, 1, 3], grades, set()) == pytest.approx(dcg([2, 3, 1]) / dcg([3, 2, 2, 1, 0]))
    # route 1 completed earlier: shown again it is worth 0, and the ideal no longer contains it
    assert S.round_ndcg([1, 2, 5], grades, {1}) == pytest.approx(dcg([0, 2, 2]) / dcg([2, 2, 1, 0]))
    assert S.round_ndcg([2, 5, 3], grades, {1}) == pytest.approx(1.0)
    assert S.round_ndcg([4], {4: 0, 1: 3}, {1}) is None  # nothing relevant left: undefined


def test_behaviour_is_seeded_bounded_and_skips_consumed(cat, eval_tree, mock_client):
    ps = generate(cat, TOWNS, n=6, seed=3)
    p = ps[0]
    ids = [int(cat.ids[i]) for i in range(10)]
    grades = {r: 3 for r in ids}
    bm = S.BehaviourModel()
    a = bm.react(p, cat, ids, grades, 1)
    assert a == bm.react(p, cat, ids, grades, 1)  # common random numbers: same draw, same reaction
    assert sum(1 for _, r in a if r == "completed") <= 1 and len(a) <= 1 + bm.max_rejections
    allbad = {r: 0 for r in ids}
    rej = [S.BehaviourModel(examine_decay=1.0, p_reject=(1.0, 0, 0, 0)).react(p, cat, ids, allbad, 2)]
    assert len(rej[0]) == 3 and all(r != "completed" for _, r in rej[0])  # capped at max_rejections
    for rid, reason in rej[0]:
        assert reason == bm.reason(p, cat, rid) and reason in S.REASONS_ALL
    sure = S.BehaviourModel(examine_decay=1.0, p_complete=(0, 0, 0, 1.0))
    assert sure.react(p, cat, ids, grades, 1) == [(ids[0], "completed")]
    assert sure.react(p, cat, ids, grades, 1, frozenset({ids[0]})) == [(ids[1], "completed")]


def test_simulation_is_deterministic_and_fully_judged(cat, eval_tree, mock_client):
    ps = generate(cat, TOWNS, n=8, seed=9)
    add_intents(ps, mock_client)
    write_personas(ps)
    J.client_for(J.production_judge()).mock.register(judge_responder, tag="judge")
    systems = {"v0_rules": ("v0_rules", {}), "recsys_v1": ("recsys_v1", {}),
               "recsys_v1_fb": ("recsys_v1_fb", {}), "recsys_v1_fb_explore": ("recsys_v1_fb_explore", {})}
    out1, g1 = S.simulate(systems, ps, rounds=4, catalog=cat)
    sim = Qrels.load(S.sim_qrels_file())
    assert len(sim) > 0 and not (set(sim.records) & set(Qrels.load().records))
    for lab in systems:
        for pid, rounds in out1[lab].items():
            assert [r["round"] for r in rounds] == [1, 2, 3, 4]
            judged = g1.grades(pid)
            assert all(rid in judged for r in rounds for rid in r["ids"])  # every round 100% judged
    # a replay judges nothing and reproduces every list, event and metric
    out2, _ = S.simulate(systems, ps, rounds=4, catalog=cat, mode="replay")
    assert out2 == out1
    # the control never changes its list; the feedback loop never re-shows what was consumed
    for pid in out1["recsys_v1"]:
        assert len({tuple(r["ids"]) for r in out1["recsys_v1"][pid]}) == 1
        assert all(r["reshown_consumed"] == 0 for r in out1["recsys_v1_fb"][pid])
    summ = S.summarise(out1, len(cat), list(systems))
    assert [r["round"] for r in summ["recsys_v1_fb"]["rounds"]] == [1, 2, 3, 4]
