"""Explicit client configuration.

``LLMConfig.from_settings()`` builds it from ``app.settings`` (environment /
``backend/.env``); tests build it directly with a temp cache dir, the mock
provider and a small budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import SecretStr

from .errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from app.settings import Settings

GENERATE_PROVIDERS = ("vertex", "openrouter", "ollama", "mock")
EMBED_PROVIDERS = ("vertex", "openrouter", "mock")
RANK_PROVIDERS = ("openrouter", "vertex", "mock")

MOCK_LLM_MODEL = "mock-llm-1"
MOCK_EMBED_MODEL = "mock-embed-1"
MOCK_RANK_MODEL = "mock-rank-1"


@dataclass(frozen=True)
class LLMConfig:
    llm_provider: str = "mock"
    embed_provider: str = "mock"
    rank_provider: str = "mock"

    # Vertex
    google_cloud_project: Optional[str] = None
    google_cloud_location: Optional[str] = None
    vertex_llm_location: Optional[str] = None
    vertex_rank_location: str = "global"
    vertex_llm_model: str = "gemini-2.5-flash"
    vertex_embed_model: str = "gemini-embedding-001"
    vertex_rank_model: str = "semantic-ranker-default@latest"
    llm_thinking_budget: int = 0
    vertex_embed_batch_size: int = 250
    vertex_embed_max_tokens_per_request: int = 18_000  # API limit is 20k

    # OpenRouter
    openrouter_api_key: Optional[SecretStr] = field(default=None, repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_llm_model: str = "google/gemini-2.5-flash"
    openrouter_embed_model: str = "google/gemini-embedding-001"
    openrouter_rank_model: str = "cohere/rerank-v3.5"
    openrouter_embed_batch_size: int = 64

    # Ollama
    ollama_api_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_s: float = 120.0

    # Mock
    mock_llm_model: str = MOCK_LLM_MODEL
    mock_embed_model: str = MOCK_EMBED_MODEL
    mock_rank_model: str = MOCK_RANK_MODEL
    mock_cost_usd_per_call: float = 0.0  # lets tests exercise the budget
    mock_latency_ms: float = 0.0

    # Client behaviour
    budget_usd: float = 5.0
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None  # None: no writable cache (frozen layers only)
    frozen_dirs: tuple[Path, ...] = ()
    cache_only: bool = False
    embed_dim: int = 768
    max_concurrency: int = 8
    max_retries: int = 5
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    timeout_s: float = 60.0
    schema_retries: int = 2

    def __post_init__(self) -> None:
        if self.llm_provider not in GENERATE_PROVIDERS:
            raise ConfigError(f"unknown LLM_PROVIDER {self.llm_provider!r}")
        if self.embed_provider not in EMBED_PROVIDERS:
            raise ConfigError(f"unknown EMBED_PROVIDER {self.embed_provider!r}")
        if self.rank_provider not in RANK_PROVIDERS:
            raise ConfigError(f"unknown RANK_PROVIDER {self.rank_provider!r}")
        if self.budget_usd < 0:
            raise ConfigError("budget must be >= 0")

    def with_(self, **changes) -> LLMConfig:
        return replace(self, **changes)

    # Default model id per capability for the selected provider
    def default_model(self, capability: str, provider: Optional[str] = None) -> str:
        provider = provider or self.provider_for(capability)
        table = {
            ("generate", "vertex"): self.vertex_llm_model,
            ("generate", "openrouter"): self.openrouter_llm_model,
            ("generate", "ollama"): self.ollama_model,
            ("generate", "mock"): self.mock_llm_model,
            ("embed", "vertex"): self.vertex_embed_model,
            ("embed", "openrouter"): self.openrouter_embed_model,
            ("embed", "mock"): self.mock_embed_model,
            ("rank", "vertex"): self.vertex_rank_model,
            ("rank", "openrouter"): self.openrouter_rank_model,
            ("rank", "mock"): self.mock_rank_model,
        }
        try:
            return table[(capability, provider)]
        except KeyError:
            raise ConfigError(f"provider {provider!r} does not support {capability}") from None

    def provider_for(self, capability: str) -> str:
        return {
            "generate": self.llm_provider,
            "embed": self.embed_provider,
            "rank": self.rank_provider,
        }[capability]

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> LLMConfig:
        if settings is None:
            from app.settings import get_settings

            settings = get_settings()
        return cls(
            llm_provider=settings.llm_provider,
            embed_provider=settings.resolved_embed_provider,
            rank_provider=settings.resolved_rank_provider,
            google_cloud_project=settings.google_cloud_project,
            google_cloud_location=settings.google_cloud_location,
            vertex_llm_location=settings.vertex_llm_location,
            vertex_rank_location=settings.vertex_rank_location,
            vertex_llm_model=settings.vertex_llm_model,
            vertex_embed_model=settings.vertex_embed_model,
            vertex_rank_model=settings.vertex_rank_model,
            llm_thinking_budget=settings.llm_thinking_budget,
            vertex_embed_batch_size=settings.vertex_embed_batch_size,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_base_url=settings.openrouter_base_url,
            openrouter_llm_model=settings.openrouter_llm_model,
            openrouter_embed_model=settings.openrouter_embed_model,
            openrouter_rank_model=settings.openrouter_rank_model,
            openrouter_embed_batch_size=settings.openrouter_embed_batch_size,
            ollama_api_url=settings.ollama_api_url,
            ollama_model=settings.ollama_model,
            ollama_timeout_s=float(settings.ollama_timeout),
            budget_usd=settings.llm_budget_usd_per_run,
            cache_enabled=settings.llm_cache_enabled,
            cache_dir=settings.resolved_llm_cache_dir,
            frozen_dirs=tuple(settings.resolved_frozen_dirs),
            cache_only=settings.llm_cache_only,
            embed_dim=settings.embed_dim,
            max_concurrency=settings.llm_max_concurrency,
            max_retries=settings.llm_max_retries,
            timeout_s=settings.llm_timeout_s,
            schema_retries=settings.llm_schema_retries,
        )
