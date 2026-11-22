"""End-to-end harness on a synthetic catalogue with the mock provider: pooling, holes, metrics,
determinism of results (except the volatile block), eval-all consistency, report, frozen replay."""

from __future__ import annotations

import json

from app.llm import LLMClient, LLMConfig
from evalkit import evaluate as E
from evalkit import judge as J
from evalkit import paths
from evalkit.intents import add_intents
from evalkit.personas import generate, write_personas
from evalkit.qrels import Qrels
from tests.evalkit.conftest import judge_responder
from tests.recsys.factory import TOWNS


def _setup(cat, mock_client):
    ps = generate(cat, TOWNS, n=12, seed=9)
    add_intents(ps, mock_client)
    write_personas(ps)
    c = J.client_for(J.production_judge())
    c.mock.register(judge_responder, tag="judge")
    return ps, c


def _strip(p):
    d = json.loads(p.read_text())
    d.pop("volatile")
    return json.dumps(d, sort_keys=True)


def test_eval_is_deterministic_and_fully_judged(cat, mock_client, eval_tree):
    ps, jc = _setup(cat, mock_client)
    E.evaluate("random", catalog=cat, personas=ps)
    E.evaluate("v0_rules", catalog=cat, personas=ps)
    res = paths.RESULTS_DIR / "v0_rules.json"
    doc = json.loads(res.read_text())
    for split in ("test", "dev", "all"):
        assert doc["metrics"][split]["judged_share_at_10"] == 1.0
    assert "not yet validated" in doc["note"] and "volatile" in doc
    q = Qrels.load()
    st = q.stats()
    assert st["by_source"]["rule"] > 0 and st["by_source"]["llm"] > 0
    assert all(r["grade"] == 0 for r in q.records.values() if r["source"] == "rule")

    first = {n: _strip(paths.RESULTS_DIR / f"{n}.json") for n in ("random", "v0_rules")}
    qrels_bytes = paths.QRELS_FILE.read_bytes()
    run_bytes = (paths.RUNS_DIR / "v0_rules.jsonl").read_bytes()
    calls_before = jc.usage_report()["totals"]["generate"]["provider_requests"]
    E.evaluate("v0_rules", catalog=cat, personas=ps)  # second run: identical except the volatile block
    assert _strip(paths.RESULTS_DIR / "v0_rules.json") == first["v0_rules"]
    assert paths.QRELS_FILE.read_bytes() == qrels_bytes
    assert (paths.RUNS_DIR / "v0_rules.jsonl").read_bytes() == run_bytes
    assert jc.usage_report()["totals"]["generate"]["provider_requests"] == calls_before  # no new judging

    # random was scored before v0_rules added pairs to the pool: eval-all rescores it against the grown qrels
    assert E.evaluate_all(catalog=cat, personas=ps) == ["random", "v0_rules"]
    assert _strip(paths.RESULTS_DIR / "v0_rules.json") == first["v0_rules"]
    assert _strip(paths.RESULTS_DIR / "random.json") != first["random"]
    consistent = {n: _strip(paths.RESULTS_DIR / f"{n}.json") for n in ("random", "v0_rules")}
    E.evaluate_all(catalog=cat, personas=ps)
    assert {n: _strip(paths.RESULTS_DIR / f"{n}.json") for n in ("random", "v0_rules")} == consistent
    assert json.loads(consistent["random"])["data"]["qrels_sha256"] == Qrels.load().sha256()

    from evalkit import report

    md = report.render()
    assert "`v0_rules` − `random`" in md and "not yet validated against the owner's labels" in md


def test_frozen_replay_cache_only(cat, mock_client, eval_tree, tmp_path):
    ps, jc = _setup(cat, mock_client)
    E.evaluate("v0_rules", catalog=cat, personas=ps)
    recs = list(Qrels.load().records.values())
    frozen = tmp_path / "frozen"
    man = jc.export_frozen(frozen, jc.touched_keys())
    assert man["missing"] == 0
    replay = LLMClient(LLMConfig(cache_dir=None, frozen_dirs=(frozen,), cache_only=True))
    out = J.replay_batches(J.production_judge(), cat, {p.persona_id: p for p in ps}, recs, client=replay)
    got = {(r["persona_id"], r["route_id"]): r["grade"] for r in out.records}
    assert got == {(r["persona_id"], r["route_id"]): r["grade"] for r in recs if r["source"] == "llm"}


def test_kappa_reports_missing_owner_labels(eval_tree):
    from evalkit import owner

    doc = owner.kappa()
    assert doc["owner"]["status"] == "owner labels missing"
    assert json.loads((paths.RESULTS_DIR / "_kappa.json").read_text())["owner"]["status"] == "owner labels missing"
