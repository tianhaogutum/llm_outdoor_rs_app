"""Vertex AI through the Google Gen AI SDK (Vertex mode, Application Default Credentials)
and the Discovery Engine Ranking API.

SDK clients are created lazily (first provider call) and per event loop, so a
fully cached replay never needs credentials and the sync helpers
(``asyncio.run`` per call) keep working.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np

from ..errors import ConfigError, ProviderError
from ..pricing import vertex_embed_cost, vertex_generate_cost, vertex_rank_cost
from ..types import (
    EmbedTask,
    GenerateRequest,
    ProviderEmbedding,
    ProviderGeneration,
    ProviderRanking,
    Usage,
)
from .base import Provider, batch_by_count_and_tokens, is_retryable_status, redact

TASK_TYPES = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}
RANK_MAX_RECORDS_PER_REQUEST = 1000


class VertexProvider(Provider):
    name = "vertex"
    supports_native_schema = True

    def __init__(
        self,
        *,
        project: Optional[str],
        location: Optional[str],
        llm_location: Optional[str] = None,
        rank_location: str = "global",
        timeout_s: float = 60.0,
        embed_batch_size: int = 250,
        embed_max_tokens_per_request: int = 18_000,
        genai_client_factory: Optional[Callable[[str], Any]] = None,
        rank_client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.project = project
        self.location = location
        self.llm_location = llm_location or location
        self.rank_location = rank_location
        self.timeout_s = timeout_s
        self.embed_batch_size = max(1, min(embed_batch_size, 250))
        self.embed_max_tokens = embed_max_tokens_per_request
        self._genai_factory = genai_client_factory or self._default_genai_client
        self._rank_factory = rank_client_factory or self._default_rank_client
        self._genai: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, Any]] = weakref.WeakKeyDictionary()
        self._rank: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = weakref.WeakKeyDictionary()

    def __repr__(self) -> str:
        return f"VertexProvider(location={self.location!r}, project=<{'set' if self.project else 'unset'}>)"

    # ------------------------------------------------------------ plumbing
    def _require_project(self) -> None:
        if not self.project or not self.location:
            raise ConfigError("Vertex AI needs GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION")

    def _default_genai_client(self, location: str) -> Any:
        from google import genai
        from google.genai import types

        return genai.Client(
            vertexai=True,
            project=self.project,
            location=location,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )

    def _default_rank_client(self) -> Any:
        from google.cloud import discoveryengine_v1

        return discoveryengine_v1.RankServiceAsyncClient()

    def _client_for(self, location: str) -> Any:
        self._require_project()
        loop = asyncio.get_running_loop()
        per_loop = self._genai.setdefault(loop, {})
        if location not in per_loop:
            try:
                per_loop[location] = self._genai_factory(location)
            except Exception as e:  # DefaultCredentialsError and friends
                raise ConfigError(self._redact(f"cannot create Vertex client: {type(e).__name__}: {e}")) from None
        return per_loop[location]

    def _rank_client(self) -> Any:
        self._require_project()
        loop = asyncio.get_running_loop()
        c = self._rank.get(loop)
        if c is None:
            try:
                c = self._rank_factory()
            except Exception as e:
                raise ConfigError(self._redact(f"cannot create ranking client: {type(e).__name__}: {e}")) from None
            self._rank[loop] = c
        return c

    def _redact(self, msg: str) -> str:
        return redact(msg, [self.project])

    def _map_error(self, e: BaseException, what: str) -> ProviderError:
        """Map google-genai ``APIError`` (``.code`` = HTTP status) and google.api_core
        exceptions (``.code`` = HTTP status) to ProviderError."""
        code = getattr(e, "code", None)
        code_int = code if isinstance(code, int) and not isinstance(code, bool) else None
        name = type(e).__name__
        timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError)) or "Timeout" in name or "DeadlineExceeded" in name
        transport = any(n in name for n in ("ConnectError", "ReadError", "WriteError", "RemoteProtocolError"))
        retryable = is_retryable_status(code_int) or timeout or transport
        return ProviderError(
            self._redact(f"{what}: {name}: {e}"),
            provider=self.name,
            status_code=code_int,
            retryable=retryable,
            timeout=timeout,
        )

    # ------------------------------------------------------------ generate
    def build_generate_args(self, req: GenerateRequest) -> dict[str, Any]:
        from google.genai import types

        contents: list[Any] = []
        for user, assistant in req.examples:
            contents.append(types.Content(role="user", parts=[types.Part(text=user)]))
            contents.append(types.Content(role="model", parts=[types.Part(text=assistant)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=req.prompt)]))

        cfg: dict[str, Any] = {
            "temperature": req.temperature,
            "max_output_tokens": req.max_tokens,
        }
        if req.system:
            cfg["system_instruction"] = req.system
        if isinstance(req.thinking, int) and not isinstance(req.thinking, bool):
            if req.thinking >= 0:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=req.thinking)
        elif isinstance(req.thinking, str):
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=req.thinking)
        if req.json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = req.json_schema
        return {"model": req.model, "contents": contents, "config": types.GenerateContentConfig(**cfg)}

    async def generate(self, req: GenerateRequest) -> ProviderGeneration:
        client = self._client_for(self.llm_location)
        args = self.build_generate_args(req)
        try:
            resp = await client.aio.models.generate_content(**args)
        except ConfigError:
            raise
        except Exception as e:
            raise self._map_error(e, "generate_content") from None
        try:
            text = resp.text or ""
        except Exception:  # blocked / no candidates
            text = ""
        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(um, "thoughts_token_count", 0) or 0),
        )
        finish = None
        try:
            finish = str(resp.candidates[0].finish_reason) if resp.candidates else None
        except Exception:
            pass
        cost = vertex_generate_cost(
            req.model, self.llm_location, usage.input_tokens, usage.output_tokens, usage.thinking_tokens
        )
        return ProviderGeneration(
            text=text.strip(),
            model=getattr(resp, "model_version", None) or req.model,
            usage=usage,
            cost_usd=cost,  # None -> unpriced model
            finish_reason=finish,
        )

    # --------------------------------------------------------------- embed
    def embed_batches(self, texts: Sequence[str]) -> list[list[int]]:
        return batch_by_count_and_tokens(texts, self.embed_batch_size, self.embed_max_tokens)

    def build_embed_args(self, texts: Sequence[str], *, model: str, task: EmbedTask, dim: int) -> dict[str, Any]:
        from google.genai import types

        return {
            "model": model,
            "contents": list(texts),
            "config": types.EmbedContentConfig(
                task_type=TASK_TYPES[task], output_dimensionality=dim, auto_truncate=True
            ),
        }

    async def embed(self, texts: Sequence[str], *, model: str, task: EmbedTask, dim: int) -> ProviderEmbedding:
        client = self._client_for(self.location)
        args = self.build_embed_args(texts, model=model, task=task, dim=dim)
        try:
            resp = await client.aio.models.embed_content(**args)
        except ConfigError:
            raise
        except Exception as e:
            raise self._map_error(e, "embed_content") from None
        embs = resp.embeddings or []
        if len(embs) != len(texts):
            raise ProviderError(
                f"embed_content: expected {len(texts)} vectors, got {len(embs)}", provider=self.name, retryable=True
            )
        mat = np.asarray([e.values for e in embs], dtype=np.float32)
        tokens = 0
        um = getattr(resp, "usage_metadata", None)
        if um is not None and getattr(um, "prompt_token_count", None):
            tokens = int(um.prompt_token_count)
        else:
            tokens = sum(int(getattr(getattr(e, "statistics", None), "token_count", 0) or 0) for e in embs)
        return ProviderEmbedding(
            vectors=mat, model=model, input_tokens=tokens, cost_usd=vertex_embed_cost(model, tokens)
        )

    # ---------------------------------------------------------------- rank
    def ranking_config(self) -> str:
        return f"projects/{self.project}/locations/{self.rank_location}/rankingConfigs/default_ranking_config"

    def build_rank_request(self, query: str, documents: Sequence[str], *, model: str, offset: int = 0) -> Any:
        from google.cloud import discoveryengine_v1 as de

        return de.RankRequest(
            ranking_config=self.ranking_config(),
            model=model,
            top_n=len(documents),
            query=query,
            records=[de.RankingRecord(id=str(offset + i), content=d) for i, d in enumerate(documents)],
            ignore_record_details_in_response=True,
        )

    async def rank(self, query: str, documents: Sequence[str], *, model: str) -> ProviderRanking:
        client = self._rank_client()
        scores = [0.0] * len(documents)
        seen = 0
        cost = 0.0
        for start in range(0, len(documents), RANK_MAX_RECORDS_PER_REQUEST):
            chunk = documents[start : start + RANK_MAX_RECORDS_PER_REQUEST]
            request = self.build_rank_request(query, chunk, model=model, offset=start)
            try:
                resp = await client.rank(request=request, timeout=self.timeout_s)
            except ConfigError:
                raise
            except Exception as e:
                raise self._map_error(e, "rank") from None
            for rec in resp.records:
                i = int(rec.id)
                if 0 <= i < len(scores):
                    scores[i] = float(rec.score)
                    seen += 1
            cost += vertex_rank_cost(len(chunk))
        if seen != len(documents):
            raise ProviderError(f"rank: expected {len(documents)} scores, got {seen}", provider=self.name)
        return ProviderRanking(
            scores=scores, model=model, search_units=max(1, -(-len(documents) // 100)), cost_usd=cost
        )
