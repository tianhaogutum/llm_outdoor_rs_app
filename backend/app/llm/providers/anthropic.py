"""Anthropic Messages API over plain httpx (generation only).

Anthropic has no embeddings or rerank endpoints; those capabilities stay on
Vertex / OpenRouter (or the no-LLM fallback). Cost is computed from the token
price table in ``pricing.anthropic_generate_cost``.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import Any, Optional

import httpx
from pydantic import SecretStr

from ..errors import ConfigError, ProviderError
from ..pricing import anthropic_generate_cost
from ..prompts import json_instruction
from ..types import GenerateRequest, ProviderGeneration, Usage
from .base import Provider, is_retryable_status, redact

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"
    supports_native_schema = False  # prompt-only JSON; the client validates and retries

    def __init__(
        self,
        *,
        api_key: Optional[SecretStr],
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 60.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )

    def __repr__(self) -> str:
        return f"AnthropicProvider(base_url={self.base_url!r}, api_key=<{'set' if self._api_key else 'unset'}>)"

    def _client(self) -> httpx.AsyncClient:
        if self._api_key is None or not self._api_key.get_secret_value():
            raise ConfigError("ANTHROPIC_API_KEY is not set")
        loop = asyncio.get_running_loop()
        c = self._clients.get(loop)
        if c is None or c.is_closed:
            c = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_s,
                transport=self._transport,
                headers={
                    "x-api-key": self._api_key.get_secret_value(),
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
            )
            self._clients[loop] = c
        return c

    def _redact(self, msg: str) -> str:
        return redact(msg, [self._api_key.get_secret_value() if self._api_key else None])

    def build_body(self, req: GenerateRequest) -> dict:
        system = req.system
        if req.json_schema is not None:
            system = (system + "\n\n" if system else "") + json_instruction(req.json_schema)
        messages: list[dict[str, str]] = []
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
        if system:
            body["system"] = system
        return body

    async def generate(self, req: GenerateRequest) -> ProviderGeneration:
        body = self.build_body(req)
        path = "/v1/messages"
        try:
            resp = await self._client().post(path, json=body)
        except httpx.TimeoutException:
            raise ProviderError(f"timeout on {path}", provider=self.name, retryable=True, timeout=True) from None
        except httpx.HTTPError as e:
            raise ProviderError(
                f"transport error on {path}: {type(e).__name__}", provider=self.name, retryable=True
            ) from None
        # 529 = Anthropic "overloaded": retry
        if resp.status_code >= 400:
            raise ProviderError(
                self._redact(f"{path}: {resp.text[:500]}"),
                provider=self.name,
                status_code=resp.status_code,
                retryable=is_retryable_status(resp.status_code) or resp.status_code == 529,
            )
        try:
            data = resp.json()
        except ValueError:
            raise ProviderError(f"{path}: non-JSON response", provider=self.name, retryable=True) from None
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        if not text:
            raise ProviderError("Empty response from Anthropic", provider=self.name, retryable=True)
        u = data.get("usage") or {}
        usage = Usage(input_tokens=int(u.get("input_tokens") or 0), output_tokens=int(u.get("output_tokens") or 0))
        model = data.get("model") or req.model
        return ProviderGeneration(
            text=text,
            model=model,
            usage=usage,
            cost_usd=anthropic_generate_cost(model, usage.input_tokens, usage.output_tokens),
            finish_reason=data.get("stop_reason"),
        )

    async def aclose(self) -> None:
        for c in list(self._clients.values()):
            await c.aclose()
        self._clients.clear()
