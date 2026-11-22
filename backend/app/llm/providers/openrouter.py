"""OpenRouter over plain httpx: chat completions, /embeddings, /rerank.

OpenRouter always returns usage accounting (``usage.cost`` in USD, the real
charge), which the client records as the real cost.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Sequence
from typing import Any, Optional

import httpx
import numpy as np
from pydantic import SecretStr

from ..errors import ConfigError, ProviderError
from ..prompts import json_instruction
from ..types import (
    EmbedTask,
    GenerateRequest,
    ProviderEmbedding,
    ProviderGeneration,
    ProviderRanking,
    Usage,
)
from .base import Provider, is_retryable_status, redact


class OpenRouterProvider(Provider):
    name = "openrouter"
    supports_native_schema = True

    def __init__(
        self,
        *,
        api_key: Optional[SecretStr],
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 60.0,
        embed_batch_size: int = 64,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.embed_batch_size = embed_batch_size
        self._transport = transport
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        # models that rejected response_format=json_schema; use prompt-only JSON for them
        self.no_native_schema: set[str] = set()

    def __repr__(self) -> str:
        return f"OpenRouterProvider(base_url={self.base_url!r}, api_key=<{'set' if self._api_key else 'unset'}>)"

    # ------------------------------------------------------------ plumbing
    def _client(self) -> httpx.AsyncClient:
        if self._api_key is None or not self._api_key.get_secret_value():
            raise ConfigError("OPENROUTER_API_KEY is not set")
        loop = asyncio.get_running_loop()
        c = self._clients.get(loop)
        if c is None or c.is_closed:
            c = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_s,
                transport=self._transport,
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
            )
            self._clients[loop] = c
        return c

    def _redact(self, msg: str) -> str:
        return redact(msg, [self._api_key.get_secret_value() if self._api_key else None])

    async def _post(self, path: str, body: dict) -> dict:
        try:
            resp = await self._client().post(path, json=body)
        except httpx.TimeoutException:
            raise ProviderError(f"timeout on {path}", provider=self.name, retryable=True, timeout=True) from None
        except httpx.HTTPError as e:
            raise ProviderError(
                f"transport error on {path}: {type(e).__name__}", provider=self.name, retryable=True
            ) from None
        if resp.status_code >= 400:
            raise ProviderError(
                self._redact(f"{path}: {resp.text[:500]}"),
                provider=self.name,
                status_code=resp.status_code,
                retryable=is_retryable_status(resp.status_code),
            )
        try:
            data = resp.json()
        except ValueError:
            raise ProviderError(f"{path}: non-JSON response", provider=self.name, retryable=True) from None
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            code = err.get("code") if isinstance(err, dict) else None
            code = code if isinstance(code, int) else None
            raise ProviderError(
                self._redact(f"{path}: {str(err)[:500]}"),
                provider=self.name,
                status_code=code,
                retryable=code is None or is_retryable_status(code),
            )
        return data

    async def aclose(self) -> None:
        for c in list(self._clients.values()):
            await c.aclose()
        self._clients.clear()

    # ------------------------------------------------------------ generate
    def build_chat_body(self, req: GenerateRequest, *, native_schema: bool) -> dict:
        system = req.system
        if req.json_schema is not None and not native_schema:
            system = (system + "\n\n" if system else "") + json_instruction(req.json_schema)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for user, assistant in req.examples:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": req.prompt})
        body: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.json_schema is not None and native_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": (req.schema_name or "response")[:64],
                    "strict": False,
                    "schema": req.json_schema,
                },
            }
        if isinstance(req.thinking, int) and not isinstance(req.thinking, bool):
            if req.thinking >= 0:
                body["reasoning"] = {"enabled": False} if req.thinking == 0 else {"max_tokens": req.thinking}
        elif isinstance(req.thinking, str):
            body["reasoning"] = {"effort": req.thinking}
        return body

    async def generate(self, req: GenerateRequest) -> ProviderGeneration:
        native = req.json_schema is not None and req.model not in self.no_native_schema
        try:
            data = await self._post("/chat/completions", self.build_chat_body(req, native_schema=native))
        except ProviderError as e:
            if native and e.status_code in (400, 404, 422):
                # model/provider does not support json_schema: fall back to validated JSON via prompt
                self.no_native_schema.add(req.model)
                data = await self._post("/chat/completions", self.build_chat_body(req, native_schema=False))
            else:
                raise
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError("malformed chat completion response", provider=self.name, retryable=True) from None
        usage = data.get("usage") or {}
        completion = int(usage.get("completion_tokens") or 0)
        reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        return ProviderGeneration(
            text=text.strip(),
            model=data.get("model", req.model),
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=max(completion - reasoning, 0),
                thinking_tokens=reasoning,
            ),
            cost_usd=float(usage["cost"]) if usage.get("cost") is not None else None,
            finish_reason=choice.get("finish_reason"),
        )

    # --------------------------------------------------------------- embed
    def embed_batches(self, texts: Sequence[str]) -> list[list[int]]:
        n = max(1, self.embed_batch_size)
        return [list(range(i, min(i + n, len(texts)))) for i in range(0, len(texts), n)]

    def build_embed_body(self, texts: Sequence[str], *, model: str, dim: int) -> dict:
        # Task type is not forwarded: OpenRouter lists no supported parameters for the Gemini
        # embedding models. The task still keys the cache.
        return {"model": model, "input": list(texts), "dimensions": dim, "encoding_format": "float"}

    async def embed(self, texts: Sequence[str], *, model: str, task: EmbedTask, dim: int) -> ProviderEmbedding:
        data = await self._post("/embeddings", self.build_embed_body(texts, model=model, dim=dim))
        rows = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        if len(rows) != len(texts):
            raise ProviderError(
                f"embeddings: expected {len(texts)} vectors, got {len(rows)}", provider=self.name, retryable=True
            )
        mat = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        usage = data.get("usage") or {}
        return ProviderEmbedding(
            vectors=mat,
            model=data.get("model", model),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            cost_usd=float(usage["cost"]) if usage.get("cost") is not None else None,
        )

    # ---------------------------------------------------------------- rank
    def build_rank_body(self, query: str, documents: Sequence[str], *, model: str) -> dict:
        # Ask for every document so each input gets a score; top_n is applied client-side.
        return {"model": model, "query": query, "documents": list(documents), "top_n": len(documents)}

    async def rank(self, query: str, documents: Sequence[str], *, model: str) -> ProviderRanking:
        data = await self._post("/rerank", self.build_rank_body(query, documents, model=model))
        scores = [0.0] * len(documents)
        seen = 0
        for r in data.get("results") or []:
            i = int(r["index"])
            if 0 <= i < len(scores):
                scores[i] = float(r["relevance_score"])
                seen += 1
        if seen != len(documents):
            raise ProviderError(
                f"rerank: expected {len(documents)} scores, got {seen}", provider=self.name, retryable=False
            )
        usage = data.get("usage") or {}
        return ProviderRanking(
            scores=scores,
            model=data.get("model", model),
            search_units=int(usage.get("search_units") or 1),
            cost_usd=float(usage["cost"]) if usage.get("cost") is not None else None,
        )
