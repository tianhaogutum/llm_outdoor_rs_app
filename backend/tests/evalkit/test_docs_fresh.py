"""The committed reports match the committed results: ``make report`` would change nothing.

Fails if a results file under ``eval/results/`` changed without ``make report`` (or ``make failures``) being
rerun, if someone edited a generated block by hand, or if a metric value crept into README prose outside a
generated block. Reads only committed files; no provider, no network.
"""

from __future__ import annotations

import json
import re

import pytest

from evalkit import blocks, paths, report
from evalkit.failures import OUT as FAILURES_OUT
from evalkit.failures import SYSTEM as FAILURES_SYSTEM
from evalkit.failures import worst_personas
from evalkit.personas import load_personas

pytestmark = pytest.mark.skipif(
    not (paths.RESULTS_DIR / "recsys_serve.json").exists(), reason="no committed results in this checkout"
)


def test_results_md_is_fresh():
    assert paths.RESULTS_MD.read_text(encoding="utf-8") == report.render(), "docs/RESULTS.md is stale: make report"


@pytest.mark.parametrize("rel", ["README.md", "docs/FAILURES.md", "docs/REVIEW.md"])
def test_generated_blocks_are_fresh(rel):
    expected = report.doc_blocks()[rel]
    text = (paths.REPO_ROOT / rel).read_text(encoding="utf-8")
    assert set(blocks.names(text)) == set(expected), f"{rel}: generated block markers changed"
    for name, body in expected.items():
        assert blocks.body(text, name) == body.strip("\n") + "\n", f"{rel}: block {name} is stale: make report"


def test_failure_analysis_matches_the_results():
    """The analysed personas are still the served system's worst test personas (else: make failures)."""
    doc = json.loads((paths.RESULTS_DIR / FAILURES_OUT).read_text(encoding="utf-8"))
    res = json.loads((paths.RESULTS_DIR / f"{FAILURES_SYSTEM}.json").read_text(encoding="utf-8"))
    split_of = {p.persona_id: p.split for p in load_personas()}
    assert [f["persona_id"] for f in doc["personas"]] == worst_personas(res, split_of, doc["n_worst"])
    for f in doc["personas"]:
        assert f["ndcg_at_10"] == res["per_persona"][f["persona_id"]]["ndcg_at_10"]


# metric-looking values: 0.xxx, percentages, durations and prices
_METRIC = re.compile(r"(?<![\w.])0\.\d{2,}|\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?(?:ms|USD)\b")


@pytest.mark.parametrize("rel", ["README.md", "docs/REVIEW.md"])
def test_prose_has_no_metric_values(rel):
    prose = blocks.strip_blocks((paths.REPO_ROOT / rel).read_text(encoding="utf-8"))
    hits = _METRIC.findall(prose)
    assert not hits, f"{rel}: metric values outside generated blocks: {hits}"
