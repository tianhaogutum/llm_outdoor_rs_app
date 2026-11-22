"""Settings: provider resolution per capability, secrets, paths."""

from __future__ import annotations

import os

import pytest

from app.llm import LLMConfig
from app.settings import DEFAULT_LLM_CACHE_DIR, REPO_ROOT, Settings

_PREFIXES = ("LLM_", "EMBED_", "RANK_", "VERTEX_", "OPENROUTER_", "GOOGLE_CLOUD_", "OLLAMA_")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.upper().startswith(_PREFIXES):
            monkeypatch.delenv(k)


def make(**env) -> Settings:
    return Settings(_env_file=None, **env)


def test_defaults_match_spec():
    s = make()
    assert s.llm_provider == "vertex"
    assert s.resolved_embed_provider == "vertex"
    assert s.resolved_rank_provider == "openrouter"
    assert s.vertex_llm_model == "gemini-2.5-flash"
    assert s.vertex_embed_model == "gemini-embedding-001"
    assert s.vertex_rank_model == "semantic-ranker-default@latest"
    assert s.openrouter_rank_model == "cohere/rerank-v3.5"
    assert s.llm_budget_usd_per_run == 5.0
    assert s.embed_dim == 768
    assert s.resolved_llm_cache_dir == DEFAULT_LLM_CACHE_DIR == REPO_ROOT / ".cache" / "llm"


def test_provider_resolution():
    assert make(llm_provider="mock").resolved_embed_provider == "mock"
    assert make(llm_provider="mock").resolved_rank_provider == "mock"
    assert make(llm_provider="ollama").resolved_embed_provider == "mock"
    assert make(llm_provider="ollama").resolved_rank_provider == "openrouter"
    assert make(llm_provider="openrouter").resolved_embed_provider == "openrouter"
    assert make(llm_provider="mock", rank_provider="vertex").resolved_rank_provider == "vertex"


def test_env_vars_are_read(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("EMBED_DIM", "256")
    monkeypatch.setenv("LLM_CACHE_FROZEN_DIRS", "eval/cache/frozen, /abs/frozen")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secret-value")
    s = Settings(_env_file=None)
    assert s.llm_provider == "openrouter" and s.embed_dim == 256
    assert s.resolved_frozen_dirs == [REPO_ROOT / "eval/cache/frozen", REPO_ROOT.__class__("/abs/frozen")]
    assert "sk-or-v1-secret-value" not in repr(s)
    assert "sk-or-v1-secret-value" not in str(s.model_dump())
    cfg = LLMConfig.from_settings(s)
    assert "sk-or-v1-secret-value" not in repr(cfg)
    assert cfg.openrouter_api_key.get_secret_value() == "sk-or-v1-secret-value"
    assert cfg.embed_dim == 256 and cfg.rank_provider == "openrouter"
