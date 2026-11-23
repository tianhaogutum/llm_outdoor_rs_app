from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../backend
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "app.db"
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_URL = f"sqlite+aiosqlite:///{DEFAULT_DB_PATH.as_posix()}"
DEFAULT_LLM_CACHE_DIR = REPO_ROOT / ".cache" / "llm"

LLMProviderName = Literal["vertex", "openrouter", "ollama", "mock"]
EmbedProviderName = Literal["vertex", "openrouter", "mock"]
RankProviderName = Literal["openrouter", "vertex", "mock"]


class Settings(BaseSettings):
    """
    Runtime configuration for the FastAPI service.

    Every field is read from the environment variable of the same name
    (case-insensitive), e.g. ``database_url`` <- ``DATABASE_URL``, and from
    ``backend/.env`` (or ``.env`` in the working directory). Real environment
    variables take precedence over ``.env``. All variables are documented in
    ``backend/env.example``.
    """

    app_name: str = Field(default="Rec Lab API")
    version: str = Field(default="0.1.0")

    # Database configuration (SQLite by default)
    database_url: str = Field(
        default=DEFAULT_DB_URL,
        description="Database connection string. Defaults to sqlite+aiosqlite:///.../data/app.db",
    )

    # ------------------------------------------------------------------
    # LLM / embedding / ranking client (app.llm). See docs/LLM_CLIENT.md.
    # ------------------------------------------------------------------
    llm_provider: LLMProviderName = Field(
        default="vertex", description="Provider for text generation."
    )
    embed_provider: Optional[EmbedProviderName] = Field(
        default=None,
        description="Provider for embeddings. Unset: same as LLM_PROVIDER (ollama -> mock).",
    )
    rank_provider: Optional[RankProviderName] = Field(
        default=None,
        description="Provider for reranking. Unset: openrouter (mock if LLM_PROVIDER=mock).",
    )

    # Vertex AI (Application Default Credentials; no keys here)
    google_cloud_project: Optional[str] = Field(default=None)
    google_cloud_location: Optional[str] = Field(default=None, description="e.g. europe-west4")
    vertex_llm_location: Optional[str] = Field(
        default=None,
        description="Override location for generation only (e.g. 'global' for Gemini 3.x). "
        "Unset: GOOGLE_CLOUD_LOCATION.",
    )
    vertex_rank_location: str = Field(
        default="global", description="Location of the Discovery Engine ranking config."
    )
    vertex_llm_model: str = Field(default="gemini-2.5-flash")
    vertex_embed_model: str = Field(default="gemini-embedding-001")
    vertex_rank_model: str = Field(default="semantic-ranker-default@latest")
    llm_thinking_budget: int = Field(
        default=0,
        description="Default thinking budget (tokens) for vertex/openrouter generate calls; "
        "0 disables thinking, -1 leaves the provider default.",
    )
    vertex_embed_batch_size: int = Field(
        default=250,
        description="Max texts per embedContent request (Vertex rejects > 250; a ~20k-token cap is also applied).",
    )

    # OpenRouter (key is a secret: never printed, logged or cached)
    openrouter_api_key: Optional[SecretStr] = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_llm_model: str = Field(default="google/gemini-2.5-flash")
    openrouter_embed_model: str = Field(default="google/gemini-embedding-001")
    openrouter_rank_model: str = Field(default="cohere/rerank-v3.5")
    openrouter_embed_batch_size: int = Field(default=64)

    # Ollama LLM configuration (existing local path; generation only)
    ollama_api_url: str = Field(
        default="http://127.0.0.1:11434/api/generate",
        description="Ollama API endpoint URL",
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name (llama3.1:8b recommended for best results)",
    )
    ollama_timeout: int = Field(
        default=120,
        description="Timeout in seconds for Ollama API calls (increased for batch story generation)",
    )

    # Budget, cache and robustness
    llm_budget_usd_per_run: float = Field(
        default=5.0, description="Hard stop: real USD spend per client (process / run)."
    )
    llm_cache_enabled: bool = Field(default=True)
    llm_cache_dir: Optional[Path] = Field(
        default=None, description="Writable cache. Unset: <repo>/.cache/llm"
    )
    llm_cache_frozen_dirs: str = Field(
        default="",
        description="Comma-separated read-only frozen cache directories (relative to repo root or absolute).",
    )
    llm_cache_only: bool = Field(
        default=False,
        description="Replay mode: a cache miss raises instead of calling a provider.",
    )
    embed_dim: int = Field(default=768, description="Default embedding dimensionality.")
    llm_max_concurrency: int = Field(default=8)
    llm_max_retries: int = Field(default=5)
    llm_timeout_s: float = Field(default=60.0)
    llm_schema_retries: int = Field(
        default=2, description="Extra attempts after a structured-output schema violation."
    )

    # Chat preference elicitation (app/chat, docs/CHAT.md). Its own client (model, location, budget),
    # so the chat model can differ from the narrative model. With LLM_PROVIDER=mock the chat is always
    # mock (tests and CI never reach the network, whatever CHAT_LLM_PROVIDER says).
    chat_llm_provider: Optional[LLMProviderName] = Field(
        default=None, description="Provider for chat slot extraction. Unset: LLM_PROVIDER."
    )
    chat_llm_model: Optional[str] = Field(
        default=None, description="Model id for chat extraction. Unset: the chat default for the provider."
    )
    chat_vertex_location: Optional[str] = Field(
        default=None, description="Vertex location for the chat model. Unset: the chat default's location."
    )
    chat_thinking: Optional[str] = Field(
        default=None,
        description="Thinking for chat calls: an int budget ('0' = off) or a level ('minimal', 'low'). "
        "Unset: the chat default.",
    )
    chat_budget_usd: float = Field(default=1.0, description="Hard stop: real USD per process for chat calls.")

    # Serving the recommender (docs/SERVING.md). RECSYS_SYSTEM picks what the recommendations endpoint
    # serves: a registered pipeline (default recsys_serve, see app/recsys/systems/pipelines.py and D25) or the old
    # rule scorer v0_rules (the DB path in recommendation_service, unchanged).
    recsys_system: str = Field(
        default="recsys_serve", description="recsys_serve (default), recsys_v1, v7_mmr, ..., or v0_rules."
    )
    recsys_timeout_s: float = Field(
        default=2.0, description="Deadline for one pipeline call; on expiry the no-LLM path serves the request."
    )
    recsys_embed_retries: int = Field(default=1, description="Retries of the query-embedding call while serving.")
    recsys_budget_usd: float = Field(
        default=1.0, description="Hard stop: real USD per process for serving-time embedding/rank calls."
    )
    recsys_catalog_dir: Optional[Path] = Field(
        default=None, description="Catalogue the pipeline serves (unset: backend/data/catalog). Tests use a subset."
    )

    # Demo protections (docs/SERVING.md). Off / generous by default for development.
    rate_limit_enabled: bool = Field(default=False, description="Per-IP rate limit on endpoints that can spend.")
    rate_limit_window_s: float = Field(default=60.0, description="Sliding window of the rate limit, seconds.")
    rate_limit_chat: int = Field(default=20, description="Chat turns per IP per window (over: HTTP 429).")
    rate_limit_recommendations: int = Field(
        default=60, description="Live-provider recommendation requests per IP per window (over: no-LLM path)."
    )
    rate_limit_narrative: int = Field(
        default=10, description="LLM narratives per IP per window (over: template text)."
    )
    rate_limit_trust_forwarded: bool = Field(
        default=False, description="Take the client IP from X-Forwarded-For (only behind a trusted proxy)."
    )
    daily_budget_usd: float = Field(
        default=5.0,
        description="Real USD per UTC day over all LLM clients of the process; then the app serves the no-LLM "
        "paths (recommendations without embeddings, keyword chat extraction, template narratives). 0 = off.",
    )

    # Narratives (welcome text, post-route summary; docs/DEPLOY.md). Order: live LLM (only when this is on
    # and LLM_PROVIDER has credentials) -> pre-generated text (backend/data/narratives) -> template.
    narratives_live: bool = Field(
        default=True,
        description="Call the LLM for narratives. false: pre-generated text or template only (no spend). "
        "The Docker image sets false.",
    )
    narratives_dir: Optional[Path] = Field(
        default=None, description="Pre-generated narrative store (unset: backend/data/narratives)."
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_enable_file: bool = Field(default=True, description="Enable file logging")
    log_enable_console: bool = Field(default=True, description="Enable console logging")
    log_detailed_format: bool = Field(
        default=True,
        description="Use detailed log format with function names and line numbers",
    )

    model_config = SettingsConfigDict(
        # backend/.env regardless of the working directory; a .env in the
        # working directory (the historical behaviour) overrides it.
        env_file=(str(PROJECT_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Derived values -----------------------------------------------------
    @property
    def resolved_embed_provider(self) -> str:
        if self.embed_provider:
            return self.embed_provider
        if self.llm_provider == "ollama":
            return "mock"
        return self.llm_provider

    @property
    def resolved_rank_provider(self) -> str:
        if self.rank_provider:
            return self.rank_provider
        return "mock" if self.llm_provider == "mock" else "openrouter"

    @property
    def resolved_llm_cache_dir(self) -> Path:
        return Path(self.llm_cache_dir) if self.llm_cache_dir else DEFAULT_LLM_CACHE_DIR

    @property
    def resolved_frozen_dirs(self) -> list[Path]:
        out = []
        for part in self.llm_cache_frozen_dirs.split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part).expanduser()
            out.append(p if p.is_absolute() else REPO_ROOT / p)
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor for application settings.
    """
    return Settings()
