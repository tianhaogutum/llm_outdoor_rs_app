"""Error types raised by the LLM client."""

from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """Base class for every error raised by ``app.llm``."""


class ConfigError(LLMError):
    """The client or a provider is misconfigured (missing project, key, unknown provider)."""


class ProviderError(LLMError):
    """A provider call failed.

    ``status_code`` is the upstream HTTP status when there is one; ``timeout``
    marks timeouts; ``retryable`` says whether the client may retry it (429,
    5xx, timeouts, connection errors). Messages are redacted: they never
    contain API keys or the Google Cloud project id.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
        timeout: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.timeout = timeout

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = super().__str__()
        code = f" status={self.status_code}" if self.status_code is not None else ""
        return f"[{self.provider}{code}] {base}"


class BudgetExceeded(LLMError):
    """The per-client real-USD budget is exhausted; no further provider calls are made."""

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        super().__init__(
            f"LLM budget exhausted: spent {spent_usd:.4f} USD of {budget_usd:.4f} USD (LLM_BUDGET_USD_PER_RUN)"
        )
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd


class SchemaValidationError(LLMError):
    """Structured output still violated the schema after all allowed attempts."""

    def __init__(self, message: str, *, attempts: int, last_text: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_text = last_text


class CacheMiss(LLMError):
    """Raised in cache-only (replay) mode when an entry is not cached."""

    def __init__(self, capability: str, key: str) -> None:
        super().__init__(f"cache miss for {capability} key {key[:16]}... in cache-only mode (LLM_CACHE_ONLY)")
        self.capability = capability
        self.key = key
