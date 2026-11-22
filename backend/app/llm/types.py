"""Request and result types shared by the client and providers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np

EmbedTask = Literal["query", "document"]
Capability = Literal["generate", "embed", "rank"]
# Thinking control: None = provider default from config, int = token budget
# (0 disables thinking where the model allows it), str = level ("minimal", "low", ...)
Thinking = Union[None, int, str]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens


# ---------------------------------------------------------------- requests
@dataclass(frozen=True)
class GenerateRequest:
    """What a provider needs to produce one completion.

    ``examples`` are few-shot (user, assistant) turns placed between the system
    text and the final user prompt. Each provider renders them in its own
    format (Llama 3.1 chat template for Ollama, multi-turn contents for Vertex,
    chat messages for OpenRouter).
    """

    model: str
    prompt: str
    system: Optional[str] = None
    examples: tuple[tuple[str, str], ...] = ()
    max_tokens: int = 512
    temperature: float = 0.0
    json_schema: Optional[dict[str, Any]] = None
    schema_name: Optional[str] = None
    thinking: Thinking = None
    tag: str = "default"
    attempt: int = 0  # schema-retry attempt (0 = first)


@dataclass
class ProviderGeneration:
    text: str
    model: str  # model id/version reported by the provider (falls back to the requested id)
    usage: Usage = field(default_factory=Usage)
    cost_usd: Optional[float] = None  # None: computed from the price table
    finish_reason: Optional[str] = None


@dataclass
class ProviderEmbedding:
    vectors: np.ndarray  # n x d, not necessarily normalised
    model: str
    input_tokens: int = 0
    cost_usd: Optional[float] = None


@dataclass
class ProviderRanking:
    scores: list[float]  # aligned with the input documents
    model: str
    search_units: int = 1
    cost_usd: Optional[float] = None


# ----------------------------------------------------------------- results
@dataclass
class GenerateResult:
    text: str
    parsed: Any  # validated pydantic instance when a schema was given, else None
    model: str
    provider: str
    usage: Usage
    cost_usd: float  # real spend of this call (0 on a cache hit)
    nominal_cost_usd: float  # what the call cost when it was first made
    latency_ms: float  # real wall time of this call
    nominal_latency_ms: float  # provider latency of the original call
    cached: bool
    cache_key: str
    schema_attempts: int = 1

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens


@dataclass
class EmbedResult:
    vectors: np.ndarray  # float32, L2-normalised, shape (n, dim)
    model: str
    provider: str
    task: EmbedTask
    dim: int
    input_tokens: int
    cost_usd: float
    nominal_cost_usd: float
    latency_ms: float
    nominal_latency_ms: float
    n_cached: int
    cache_keys: list[str]

    @property
    def cached(self) -> bool:
        """True when every text came from the cache."""
        return self.n_cached == len(self.cache_keys)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])


@dataclass
class RankResult:
    scores: list[float]  # one per input document, input order
    model: str
    provider: str
    top_n: Optional[int]
    cost_usd: float
    nominal_cost_usd: float
    latency_ms: float
    nominal_latency_ms: float
    cached: bool
    cache_key: str

    @property
    def order(self) -> list[int]:
        """Document indices by descending score (stable), cut to ``top_n``."""
        idx = sorted(range(len(self.scores)), key=lambda i: (-self.scores[i], i))
        return idx[: self.top_n] if self.top_n else idx

    def top(self, documents: Sequence[Any]) -> list[tuple[int, float, Any]]:
        return [(i, self.scores[i], documents[i]) for i in self.order]
