"""Repository paths used by the evaluation harness."""

from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
EVAL_DIR = REPO_ROOT / "eval"

PERSONA_DIR = EVAL_DIR / "personas"
TOWNS_FILE = PERSONA_DIR / "towns.json"
PERSONAS_FILE = PERSONA_DIR / "personas.jsonl"
PERSONA_PARAMS_FILE = PERSONA_DIR / "generator_params.json"
PERSONA_STATS_FILE = PERSONA_DIR / "stats.json"

QRELS_DIR = EVAL_DIR / "qrels"
QRELS_FILE = QRELS_DIR / "qrels.jsonl"

RUNS_DIR = EVAL_DIR / "runs"
RESULTS_DIR = EVAL_DIR / "results"
PILOT_DIR = EVAL_DIR / "pilot"
OWNER_DIR = EVAL_DIR / "owner_labels"
FROZEN_DIR = EVAL_DIR / "cache"

DOCS_DIR = REPO_ROOT / "docs"
RESULTS_MD = DOCS_DIR / "RESULTS.md"
