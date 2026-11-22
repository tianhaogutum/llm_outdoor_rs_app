"""Local Ollama (generation only): the pre-existing path.

Sends the same ``/api/generate`` payload as the old ``call_ollama``: the
prompt rendered in the Llama 3.1 chat template, ``stream: false`` and
``options.temperature`` / ``options.num_predict``.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..errors import ProviderError
from ..prompts import render_llama3
from ..types import GenerateRequest, ProviderGeneration, Usage
from .base import Provider, is_retryable_status, redact


class OllamaProvider(Provider):
    name = "ollama"
    supports_native_schema = True  # Ollama "format": <json schema>

    def __init__(
        self,
        *,
        api_url: str,
        timeout_s: float = 120.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.api_url = api_url
        self.timeout_s = timeout_s
        self._transport = transport

    def build_payload(self, req: GenerateRequest) -> dict:
        payload = {
            "model": req.model,
            "prompt": render_llama3(req.system, req.examples, req.prompt),
            "stream": False,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
        }
        if req.json_schema is not None:
            payload["format"] = req.json_schema
        return payload

    async def generate(self, req: GenerateRequest) -> ProviderGeneration:
        payload = self.build_payload(req)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, transport=self._transport) as client:
                resp = await client.post(self.api_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            raise ProviderError(
                redact(f"Ollama API error: {e.response.text}", []),
                provider=self.name,
                status_code=code,
                retryable=is_retryable_status(code),
            ) from None
        except httpx.TimeoutException:
            raise ProviderError(
                f"Ollama API timeout after {self.timeout_s}s", provider=self.name, retryable=True, timeout=True
            ) from None
        except httpx.HTTPError as e:
            # Connection refused etc.: Ollama not running. Not retried (fail fast to the fallback).
            raise ProviderError(f"Ollama unreachable: {type(e).__name__}", provider=self.name) from None

        if "response" in data and data.get("done", False):
            return ProviderGeneration(
                text=data["response"].strip(),
                model=data.get("model", req.model),
                usage=Usage(
                    input_tokens=int(data.get("prompt_eval_count") or 0),
                    output_tokens=int(data.get("eval_count") or 0),
                ),
                cost_usd=0.0,
                finish_reason=data.get("done_reason"),
            )
        raise ProviderError("Empty response from Ollama", provider=self.name)
