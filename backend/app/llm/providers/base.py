"""Provider interface."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Optional

from ..errors import ProviderError
from ..types import (
    EmbedTask,
    GenerateRequest,
    ProviderEmbedding,
    ProviderGeneration,
    ProviderRanking,
)


class Provider:
    """One backend. Providers do no caching, budgeting or retrying; the client does.

    Implementations must raise :class:`ProviderError` (with ``retryable`` set
    correctly) for upstream failures and must never put secrets in messages.
    """

    name: str = "base"
    supports_native_schema: bool = False

    async def generate(self, req: GenerateRequest) -> ProviderGeneration:
        raise ProviderError("generate not supported", provider=self.name)

    async def embed(self, texts: Sequence[str], *, model: str, task: EmbedTask, dim: int) -> ProviderEmbedding:
        raise ProviderError("embed not supported", provider=self.name)

    async def rank(self, query: str, documents: Sequence[str], *, model: str) -> ProviderRanking:
        raise ProviderError("rank not supported", provider=self.name)

    def embed_batches(self, texts: Sequence[str]) -> list[list[int]]:
        """Split text indices into request-sized batches."""
        return [list(range(len(texts)))] if texts else []

    async def aclose(self) -> None:
        return None


def is_retryable_status(status: Optional[int]) -> bool:
    return status is not None and (status == 429 or status == 408 or 500 <= status < 600)


def redact(message: str, secrets: Iterable[Optional[str]]) -> str:
    """Remove secret-ish values (keys, project id) from an error message."""
    out = message
    for s in secrets:
        if s:
            out = out.replace(s, "<redacted>")
    # Bearer tokens / API-key-looking strings, defensively
    out = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", out)
    out = re.sub(r"sk-ant-[A-Za-z0-9_\-]+", "<redacted>", out)
    out = re.sub(r"sk-or-[A-Za-z0-9_\-]+", "<redacted>", out)
    return out[:800]


def batch_by_count_and_tokens(
    texts: Sequence[str], max_count: int, max_tokens: int, chars_per_token: float = 3.0
) -> list[list[int]]:
    """Greedy batches with at most ``max_count`` texts and an estimated token cap."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_tok = 0
    for i, t in enumerate(texts):
        est = max(1, int(len(t) / chars_per_token) + 1)
        if cur and (len(cur) >= max_count or cur_tok + est > max_tokens):
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += est
    if cur:
        batches.append(cur)
    return batches
