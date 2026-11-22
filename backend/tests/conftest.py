"""
Shared test setup.

Runs before any ``app`` import: forces the mock LLM provider (no network), points DATABASE_URL at a
throw-away SQLite file so no test can touch ``backend/data/app.db``, and turns file logging off.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
SEED_FILE = BACKEND_DIR / "data" / "seed" / "routes.jsonl"
GOLDEN_FILE = BACKEND_DIR / "tests" / "golden" / "v0_rules_golden.json"

_SESSION_DIR = Path(tempfile.mkdtemp(prefix="backend-tests-"))
TEST_DB = _SESSION_DIR / "test_app.db"

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB}"
os.environ.setdefault("LOG_ENABLE_FILE", "false")
os.environ.setdefault("LOG_ENABLE_CONSOLE", "false")
# Keep the LLM cache out of the repo's .cache/ and ignore any frozen layers during tests.
os.environ.setdefault("LLM_CACHE_DIR", str(_SESSION_DIR / "llm-cache"))
os.environ["LLM_CACHE_FROZEN_DIRS"] = ""

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def run_alembic(db_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run alembic against ``db_path`` in a subprocess (alembic's env.py caches settings per process)."""
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}"}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def build_seeded_db(db_path: Path, seed_file: Path = SEED_FILE) -> Path:
    from scripts.seed_from_jsonl import load_seed

    db_path.unlink(missing_ok=True)
    run_alembic(db_path, "upgrade", "head")
    load_seed(db_path, seed_file)
    return db_path


@pytest.fixture(scope="session")
def seeded_db() -> Path:
    """
    Migrated DB filled from the seed file. It is also the app's DATABASE_URL in tests, so API tests write
    to it; tests that need pristine seed data or write directly should use ``seeded_db_copy``.
    """
    return build_seeded_db(TEST_DB)


@pytest.fixture
def seeded_db_copy(seeded_db: Path, tmp_path: Path) -> Path:
    """A private, writable copy of the seeded DB."""
    import shutil

    target = tmp_path / "copy.db"
    shutil.copy(seeded_db, target)
    return target


@pytest.fixture(scope="session")
def seed_records() -> list[dict]:
    import json

    with SEED_FILE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
