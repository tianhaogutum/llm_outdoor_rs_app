"""Fixtures for the LLM client tests: isolated clients, no network."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("LLM_PROVIDER", "mock")

from app.llm import LLMClient, LLMConfig, set_client  # noqa: E402


@pytest.fixture
def make_client(tmp_path):
    """Factory for isolated clients: temp cache dir, mock providers, fast retries."""
    created: list[LLMClient] = []

    def _make(*, providers=None, **overrides) -> LLMClient:
        cfg = dict(
            llm_provider="mock",
            embed_provider="mock",
            rank_provider="mock",
            cache_dir=tmp_path / "cache",
            budget_usd=1.0,
            retry_base_delay_s=0.001,
            retry_max_delay_s=0.005,
        )
        cfg.update(overrides)
        c = LLMClient(LLMConfig(**cfg), providers=providers)
        created.append(c)
        return c

    yield _make
    for c in created:
        c.close()


@pytest.fixture
def client(make_client) -> LLMClient:
    return make_client()


@pytest.fixture
def global_client(make_client):
    """Install an isolated mock client as the process-wide client (for genai_service)."""
    c = make_client()
    set_client(c)
    yield c
    set_client(None)
