"""Fixtures for the evaluation-harness tests: synthetic catalogue, personas and an isolated eval/ tree."""

from __future__ import annotations

import json
import re

import pytest

from app.llm import LLMClient, LLMConfig
from tests.recsys.factory import TOWNS, synthetic_catalog


@pytest.fixture(scope="module")
def cat():
    return synthetic_catalog(300, seed=1)


def judge_responder(req):
    """Mock judge: a deterministic grade per route label and title."""
    n = int(re.search(r"ROUTES \((\d+)\):", req.prompt).group(1))
    titles = re.findall(r"^Title: (.+)$", req.prompt, flags=re.M)
    return {
        "grades": [
            {"route": f"R{i + 1}", "reason": "mock", "grade": (sum(map(ord, titles[i])) + i * 0) % 4} for i in range(n)
        ]
    }


@pytest.fixture
def eval_tree(tmp_path, monkeypatch):
    """Point every evalkit path at a temp copy of the eval/ layout (the rubric is copied from the repo)."""
    from evalkit import paths

    root = tmp_path / "eval"
    (root / "qrels").mkdir(parents=True)
    (root / "personas").mkdir()
    (root / "qrels" / "rubric_v1.md").write_text(paths.QRELS_DIR.joinpath("rubric_v1.md").read_text())
    (root / "personas" / "towns.json").write_text(json.dumps({"towns": TOWNS}))
    for name, sub in [
        ("EVAL_DIR", ""),
        ("PERSONA_DIR", "personas"),
        ("TOWNS_FILE", "personas/towns.json"),
        ("PERSONAS_FILE", "personas/personas.jsonl"),
        ("PERSONA_PARAMS_FILE", "personas/generator_params.json"),
        ("PERSONA_STATS_FILE", "personas/stats.json"),
        ("QRELS_DIR", "qrels"),
        ("QRELS_FILE", "qrels/qrels.jsonl"),
        ("RUNS_DIR", "runs"),
        ("RESULTS_DIR", "results"),
        ("PILOT_DIR", "pilot"),
        ("OWNER_DIR", "owner_labels"),
        ("FROZEN_DIR", "cache"),
    ]:
        monkeypatch.setattr(paths, name, root / sub if sub else root)
    monkeypatch.setattr(paths, "RESULTS_MD", tmp_path / "RESULTS.md")
    from evalkit import judge

    judge._judge_clients.clear()
    yield root
    judge._judge_clients.clear()


@pytest.fixture
def mock_client(tmp_path):
    return LLMClient(LLMConfig(cache_dir=tmp_path / "llm-cache"))
