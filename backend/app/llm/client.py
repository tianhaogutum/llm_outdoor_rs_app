"""The single client for every generate / embed / rank call in the project.

    from app.llm import get_client
    client = get_client()
    res = await client.generate("...", system="...", schema=MyModel, tag="judge")
    emb = await client.embed(texts, task="document", tag="routes")
    rk = await client.rank(query, docs, top_n=10, tag="rerank")
    report = client.usage_report()

Responsibilities: provider selection per capability, on-disk cache (local +
frozen layers), per-run budget (hard stop), ledger, bounded concurrency,
retries with exponential backoff and jitter, structured-output validation.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from .cache import Entry, TieredCache, embed_key, export_frozen, generate_key, rank_key
from .config import LLMConfig
from .errors import BudgetExceeded, CacheMiss, ConfigError, ProviderError, SchemaValidationError
from .ledger import Ledger, Stats
from .prompts import extract_json_text
from .providers.base import Provider
from .types import (
    EmbedResult,
    EmbedTask,
    GenerateRequest,
    GenerateResult,
    ProviderGeneration,
    RankResult,
    Thinking,
    Usage,
)

logger = logging.getLogger("app.llm")
T = TypeVar("T")
_FREE_PROVIDERS = {"ollama"}


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class LLMClient:
    def __init__(self, config: LLMConfig, *, providers: Optional[dict[str, Provider]] = None) -> None:
        self.config = config
        self._providers: dict[str, Provider] = dict(providers or {})
        self._provider_lock = threading.Lock()
        local_dir = config.cache_dir if config.cache_enabled else None
        frozen = config.frozen_dirs if config.cache_enabled else ()
        self.cache = TieredCache(local_dir, frozen)
        self.ledger = Ledger()
        self._spend_lock = threading.Lock()
        self._spent_usd = 0.0
        self._touched: dict[str, None] = {}
        self._touched_lock = threading.Lock()
        self._sems: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def __repr__(self) -> str:
        c = self.config
        return (
            f"LLMClient(generate={c.llm_provider}, embed={c.embed_provider}, rank={c.rank_provider}, "
            f"budget_usd={c.budget_usd})"
        )

    # ------------------------------------------------------------ providers
    def provider(self, name: str) -> Provider:
        with self._provider_lock:
            p = self._providers.get(name)
            if p is None:
                p = self._build_provider(name)
                self._providers[name] = p
            return p

    @property
    def mock(self):
        """The mock provider instance (to register canned responses in tests)."""
        from .providers.mock import MockProvider

        p = self.provider("mock")
        assert isinstance(p, MockProvider)
        return p

    def _build_provider(self, name: str) -> Provider:
        c = self.config
        if name == "mock":
            from .providers.mock import MockProvider

            return MockProvider(cost_usd_per_call=c.mock_cost_usd_per_call, latency_ms=c.mock_latency_ms)
        if name == "ollama":
            from .providers.ollama import OllamaProvider

            return OllamaProvider(api_url=c.ollama_api_url, timeout_s=c.ollama_timeout_s)
        if name == "openrouter":
            from .providers.openrouter import OpenRouterProvider

            return OpenRouterProvider(
                api_key=c.openrouter_api_key,
                base_url=c.openrouter_base_url,
                timeout_s=c.timeout_s,
                embed_batch_size=c.openrouter_embed_batch_size,
            )
        if name == "vertex":
            from .providers.vertex import VertexProvider

            return VertexProvider(
                project=c.google_cloud_project,
                location=c.google_cloud_location,
                llm_location=c.vertex_llm_location,
                rank_location=c.vertex_rank_location,
                timeout_s=c.timeout_s,
                embed_batch_size=c.vertex_embed_batch_size,
                embed_max_tokens_per_request=c.vertex_embed_max_tokens_per_request,
            )
        raise ConfigError(f"unknown provider {name!r}")

    def _is_free(self, provider_name: str) -> bool:
        if provider_name in _FREE_PROVIDERS:
            return True
        return provider_name == "mock" and self.config.mock_cost_usd_per_call == 0

    # ---------------------------------------------------------------- budget
    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    def _check_budget(self, provider_name: str) -> None:
        if self._is_free(provider_name):
            return
        if self._spent_usd >= self.config.budget_usd:
            raise BudgetExceeded(self._spent_usd, self.config.budget_usd)

    def _charge(self, usd: float) -> None:
        with self._spend_lock:
            self._spent_usd += usd

    # ------------------------------------------------------------- plumbing
    def _sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        s = self._sems.get(loop)
        if s is None:
            s = asyncio.Semaphore(max(1, self.config.max_concurrency))
            self._sems[loop] = s
        return s

    def _touch(self, keys: Iterable[str]) -> None:
        with self._touched_lock:
            for k in keys:
                self._touched[k] = None

    def _use_cache(self, cache: Optional[bool]) -> bool:
        return self.config.cache_enabled and (True if cache is None else bool(cache))

    def _backoff(self, attempt: int) -> float:
        d = min(self.config.retry_max_delay_s, self.config.retry_base_delay_s * (2**attempt))
        return d / 2 + random.uniform(0, d / 2)  # "equal jitter"

    async def _call(
        self,
        provider_name: str,
        fn: Callable[[], Awaitable[T]],
        stats: Stats,
        *,
        retries: Optional[int] = None,
    ) -> tuple[T, float]:
        """Run one provider request with budget check, concurrency bound and retries.

        Returns (result, latency_ms of the successful attempt)."""
        max_retries = self.config.max_retries if retries is None else retries
        attempt = 0
        while True:
            try:
                self._check_budget(provider_name)
            except BudgetExceeded:
                stats.budget_blocked += 1
                raise
            async with self._sem():
                t0 = time.perf_counter()
                stats.provider_requests += 1
                try:
                    res = await fn()
                    return res, (time.perf_counter() - t0) * 1000
                except ProviderError as e:
                    err = e
                except (ConfigError, BudgetExceeded):
                    stats.errors += 1
                    raise
            if err.retryable and attempt < max_retries:
                delay = self._backoff(attempt)
                logger.warning(
                    "llm %s retry %d/%d after %.2fs: %s", provider_name, attempt + 1, max_retries, delay, err
                )
                stats.retries += 1
                attempt += 1
                await asyncio.sleep(delay)
                continue
            stats.errors += 1
            raise err

    # -------------------------------------------------------------- generate
    def _resolve_thinking(self, provider_name: str, thinking: Thinking) -> Thinking:
        if provider_name not in ("vertex", "openrouter"):
            return None
        if thinking is None:
            thinking = self.config.llm_thinking_budget
        if isinstance(thinking, int) and not isinstance(thinking, bool) and thinking < 0:
            return None
        return thinking

    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        examples: Sequence[tuple[str, str]] = (),
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        schema: Optional[type[BaseModel]] = None,
        thinking: Thinking = None,
        tag: str = "default",
        cache: Optional[bool] = None,
        retries: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> GenerateResult:
        """Generate text (or a validated ``schema`` instance in ``.parsed``).

        ``examples``: few-shot (user, assistant) turns, rendered per provider.
        ``thinking``: None = config default (``LLM_THINKING_BUDGET``, 0 = off), int budget, or level str.
        ``cache``: None/True = use the cache, False = always call the provider and do not store.
        """
        t_start = time.perf_counter()
        pname = provider or self.config.llm_provider
        model = model or self.config.default_model("generate", pname)
        thinking = self._resolve_thinking(pname, thinking)
        json_schema = schema.model_json_schema() if schema is not None else None
        examples = tuple((str(u), str(a)) for u, a in examples)
        key = generate_key(
            provider=pname,
            model=model,
            prompt=prompt,
            system=system,
            examples=examples,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            thinking=thinking,
        )
        stats = Stats(calls=1, items=1)
        use_cache = self._use_cache(cache)

        if use_cache:
            self._touch([key])
            entry = self.cache.get(key)
            if entry is not None:
                p = entry.payload
                parsed = None
                ok = True
                if schema is not None:
                    try:
                        parsed = schema.model_validate_json(extract_json_text(p["text"]))
                    except ValidationError:
                        ok = False  # should not happen: only valid outputs are stored
                if ok:
                    usage = Usage(**p.get("usage", {}))
                    stats.cache_hits = 1
                    stats.nominal_usd = float(p.get("cost_usd", 0.0))
                    stats.nominal_latency_ms = float(p.get("latency_ms", 0.0))
                    stats.real_latency_ms = (time.perf_counter() - t_start) * 1000
                    self.ledger.record(
                        "generate", tag, stats, provider=pname, model=model, reported_model=p.get("model")
                    )
                    return GenerateResult(
                        text=p["text"],
                        parsed=parsed,
                        model=p.get("model", model),
                        provider=pname,
                        usage=usage,
                        cost_usd=0.0,
                        nominal_cost_usd=stats.nominal_usd,
                        latency_ms=stats.real_latency_ms,
                        nominal_latency_ms=stats.nominal_latency_ms,
                        cached=True,
                        cache_key=key,
                        schema_attempts=int(p.get("schema_attempts", 1)),
                    )
        if self.config.cache_only:
            self.ledger.record("generate", tag, Stats(calls=1, items=1, errors=1), provider=pname, model=model)
            raise CacheMiss("generate", key)

        prov = self.provider(pname)
        attempts_allowed = 1 + (max(0, self.config.schema_retries) if schema is not None else 0)
        total = Usage()
        cost_total = 0.0
        latency_total = 0.0
        last_text = ""
        last_err = ""
        gen: Optional[ProviderGeneration] = None
        parsed = None
        attempt = 0
        failed = True
        try:
            for attempt in range(attempts_allowed):
                p_text = prompt
                if attempt > 0:
                    p_text = (
                        f"{prompt}\n\nYour previous reply did not validate against the required JSON schema "
                        f"({last_err}). Reply again with one valid JSON object only."
                    )
                req = GenerateRequest(
                    model=model,
                    prompt=p_text,
                    system=system,
                    examples=examples,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_schema=json_schema,
                    schema_name=schema.__name__ if schema is not None else None,
                    thinking=thinking,
                    tag=tag,
                    attempt=attempt,
                )
                gen, lat = await self._call(pname, lambda req=req: prov.generate(req), stats, retries=retries)
                c = gen.cost_usd
                if c is None:
                    stats.unpriced_requests += 1
                    c = 0.0
                self._charge(c)
                cost_total += c
                latency_total += lat
                total = Usage(
                    total.input_tokens + gen.usage.input_tokens,
                    total.output_tokens + gen.usage.output_tokens,
                    total.thinking_tokens + gen.usage.thinking_tokens,
                )
                last_text = gen.text
                if schema is None:
                    break
                try:
                    parsed = schema.model_validate_json(extract_json_text(gen.text))
                    break
                except ValidationError as ve:
                    stats.schema_violations += 1
                    last_err = f"{ve.error_count()} error(s): " + "; ".join(
                        f"{'.'.join(map(str, er['loc'])) or '<root>'}: {er['msg']}" for er in ve.errors()[:3]
                    )
                    parsed = None
            failed = False
        finally:
            stats.input_tokens += total.input_tokens
            stats.output_tokens += total.output_tokens
            stats.thinking_tokens += total.thinking_tokens
            stats.real_usd += cost_total
            stats.nominal_usd += cost_total
            stats.real_latency_ms += (time.perf_counter() - t_start) * 1000
            stats.nominal_latency_ms += latency_total
            if failed:
                self.ledger.record("generate", tag, stats, provider=pname, model=model)

        assert gen is not None
        if schema is not None and parsed is None:
            stats.schema_failures += 1
            self.ledger.record("generate", tag, stats, provider=pname, model=model, reported_model=gen.model)
            raise SchemaValidationError(
                f"output of {model} violated schema {schema.__name__} in {attempt + 1} attempt(s): {last_err}",
                attempts=attempt + 1,
                last_text=last_text,
            )
        self.ledger.record("generate", tag, stats, provider=pname, model=model, reported_model=gen.model)

        if use_cache and gen.text:
            self.cache.put_many(
                [
                    Entry(
                        key,
                        "generate",
                        {
                            "text": gen.text,
                            "model": gen.model,
                            "provider": pname,
                            "usage": {
                                "input_tokens": total.input_tokens,
                                "output_tokens": total.output_tokens,
                                "thinking_tokens": total.thinking_tokens,
                            },
                            "cost_usd": cost_total,
                            "latency_ms": latency_total,
                            "schema_attempts": attempt + 1,
                            "finish_reason": gen.finish_reason,
                        },
                    )
                ]
            )
        logger.debug(
            "llm generate tag=%s provider=%s model=%s cost=%.6f latency=%.0fms",
            tag,
            pname,
            model,
            cost_total,
            latency_total,
        )
        return GenerateResult(
            text=gen.text,
            parsed=parsed,
            model=gen.model,
            provider=pname,
            usage=total,
            cost_usd=cost_total,
            nominal_cost_usd=cost_total,
            latency_ms=stats.real_latency_ms,
            nominal_latency_ms=latency_total,
            cached=False,
            cache_key=key,
            schema_attempts=attempt + 1,
        )

    # ----------------------------------------------------------------- embed
    async def embed(
        self,
        texts: Sequence[str],
        *,
        task: EmbedTask = "document",
        model: Optional[str] = None,
        dim: Optional[int] = None,
        tag: str = "default",
        cache: Optional[bool] = None,
        retries: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> EmbedResult:
        """Embed ``texts``; returns float32 L2-normalised vectors (n x dim) in input order.

        Cached per text: only unknown texts go to the provider."""
        if task not in ("query", "document"):
            raise ValueError("task must be 'query' or 'document'")
        if isinstance(texts, str):
            raise TypeError("embed() takes a list of strings")
        t_start = time.perf_counter()
        texts = [str(t) for t in texts]
        pname = provider or self.config.embed_provider
        model = model or self.config.default_model("embed", pname)
        dim = int(dim or self.config.embed_dim)
        keys = [embed_key(provider=pname, model=model, text=t, task=task, dim=dim) for t in texts]
        stats = Stats(calls=1, items=len(texts))
        use_cache = self._use_cache(cache)
        vectors: dict[str, np.ndarray] = {}
        nominal_cost = 0.0
        nominal_lat = 0.0
        tokens = 0  # nominal input tokens (cached + fresh)
        real_tokens = 0
        reported = None

        if use_cache and keys:
            self._touch(keys)
            for k, e in self.cache.get_many(keys).items():
                if e.vector is not None and e.vector.shape[0] == dim:
                    vectors[k] = e.vector
                    nominal_cost += float(e.payload.get("cost_usd", 0.0))
                    nominal_lat += float(e.payload.get("latency_ms", 0.0))
                    tokens += int(e.payload.get("input_tokens", 0))
                    reported = reported or e.payload.get("model")
        n_cached = sum(1 for k in keys if k in vectors)
        stats.cache_hits = n_cached

        # unique missing texts, first occurrence order
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in vectors and k not in missing:
                missing[k] = t

        real_cost = 0.0
        if missing:
            if self.config.cache_only:
                stats.errors += 1
                self.ledger.record("embed", tag, stats, provider=pname, model=model)
                raise CacheMiss("embed", next(iter(missing)))
            prov = self.provider(pname)
            mkeys = list(missing)
            mtexts = [missing[k] for k in mkeys]

            async def run_batch(idx: list[int]) -> tuple[float, float, int, Optional[str]]:
                btexts = [mtexts[i] for i in idx]
                res, lat = await self._call(
                    pname, lambda: prov.embed(btexts, model=model, task=task, dim=dim), stats, retries=retries
                )
                mat = np.asarray(res.vectors, dtype=np.float32)
                if mat.ndim != 2 or mat.shape[0] != len(btexts):
                    raise ProviderError("embedding batch shape mismatch", provider=pname)
                if mat.shape[1] < dim:
                    raise ProviderError(
                        f"provider returned {mat.shape[1]}-dim vectors, {dim} requested", provider=pname
                    )
                mat = _l2_normalise(mat[:, :dim])  # MRL truncation if the provider ignored dim
                c = res.cost_usd
                if c is None:
                    stats.unpriced_requests += 1
                    c = 0.0
                self._charge(c)
                weights = np.array([max(len(t), 1) for t in btexts], dtype=np.float64)
                weights /= weights.sum()
                entries = []
                for j, i in enumerate(idx):
                    k = mkeys[i]
                    vectors[k] = mat[j]
                    entries.append(
                        Entry(
                            k,
                            "embed",
                            {
                                "model": res.model,
                                "provider": pname,
                                "task": task,
                                "dim": dim,
                                "input_tokens": int(round(res.input_tokens * weights[j])),
                                "cost_usd": float(c * weights[j]),
                                "latency_ms": float(lat / len(idx)),
                            },
                            mat[j],
                        )
                    )
                if use_cache:
                    self.cache.put_many(entries)
                return c, lat, res.input_tokens, res.model

            batches = prov.embed_batches(mtexts)
            results = await asyncio.gather(*(run_batch(b) for b in batches), return_exceptions=True)
            errs = [r for r in results if isinstance(r, BaseException)]
            for r in results:
                if not isinstance(r, BaseException):
                    c, lat, tok, rep = r
                    real_cost += c
                    nominal_cost += c
                    nominal_lat += lat
                    tokens += tok
                    real_tokens += tok
                    reported = rep or reported
            if errs:
                stats.real_usd += real_cost
                stats.nominal_usd += nominal_cost
                stats.input_tokens += real_tokens
                self.ledger.record("embed", tag, stats, provider=pname, model=model, reported_model=reported)
                raise errs[0]

        mat = np.stack([vectors[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, dim), np.float32)
        stats.real_usd += real_cost
        stats.nominal_usd += nominal_cost
        stats.input_tokens += real_tokens
        stats.real_latency_ms += (time.perf_counter() - t_start) * 1000
        stats.nominal_latency_ms += nominal_lat
        self.ledger.record("embed", tag, stats, provider=pname, model=model, reported_model=reported)
        return EmbedResult(
            vectors=mat,
            model=reported or model,
            provider=pname,
            task=task,
            dim=dim,
            input_tokens=tokens,
            cost_usd=real_cost,
            nominal_cost_usd=nominal_cost,
            latency_ms=stats.real_latency_ms,
            nominal_latency_ms=nominal_lat,
            n_cached=n_cached,
            cache_keys=keys,
        )

    # ------------------------------------------------------------------ rank
    async def rank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        tag: str = "default",
        cache: Optional[bool] = None,
        retries: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> RankResult:
        """Score every document against ``query``. ``scores`` is aligned with the input order;
        ``top_n`` only cuts ``RankResult.order`` (all documents are always scored)."""
        t_start = time.perf_counter()
        documents = [str(d) for d in documents]
        pname = provider or self.config.rank_provider
        model = model or self.config.default_model("rank", pname)
        key = rank_key(provider=pname, model=model, query=query, documents=documents)
        stats = Stats(calls=1, items=1)
        use_cache = self._use_cache(cache)

        def result(scores, rep_model, cost, nominal, lat, nominal_lat, cached) -> RankResult:
            return RankResult(
                scores=list(scores),
                model=rep_model or model,
                provider=pname,
                top_n=top_n,
                cost_usd=cost,
                nominal_cost_usd=nominal,
                latency_ms=lat,
                nominal_latency_ms=nominal_lat,
                cached=cached,
                cache_key=key,
            )

        if not documents:
            return result([], None, 0.0, 0.0, 0.0, 0.0, False)

        if use_cache:
            self._touch([key])
            e = self.cache.get(key)
            if e is not None:
                p = e.payload
                stats.cache_hits = 1
                stats.nominal_usd = float(p.get("cost_usd", 0.0))
                stats.nominal_latency_ms = float(p.get("latency_ms", 0.0))
                stats.real_latency_ms = (time.perf_counter() - t_start) * 1000
                self.ledger.record("rank", tag, stats, provider=pname, model=model, reported_model=p.get("model"))
                return result(
                    p["scores"],
                    p.get("model"),
                    0.0,
                    stats.nominal_usd,
                    stats.real_latency_ms,
                    stats.nominal_latency_ms,
                    True,
                )
        if self.config.cache_only:
            self.ledger.record("rank", tag, Stats(calls=1, items=1, errors=1), provider=pname, model=model)
            raise CacheMiss("rank", key)

        prov = self.provider(pname)
        try:
            res, lat = await self._call(pname, lambda: prov.rank(query, documents, model=model), stats, retries=retries)
        except BaseException:
            self.ledger.record("rank", tag, stats, provider=pname, model=model)
            raise
        if len(res.scores) != len(documents):
            raise ProviderError("rank: score count mismatch", provider=pname)
        c = res.cost_usd
        if c is None:
            stats.unpriced_requests += 1
            c = 0.0
        self._charge(c)
        stats.real_usd = stats.nominal_usd = c
        stats.real_latency_ms = (time.perf_counter() - t_start) * 1000
        stats.nominal_latency_ms = lat
        self.ledger.record("rank", tag, stats, provider=pname, model=model, reported_model=res.model)
        if use_cache:
            self.cache.put_many(
                [
                    Entry(
                        key,
                        "rank",
                        {
                            "scores": [float(s) for s in res.scores],
                            "model": res.model,
                            "provider": pname,
                            "search_units": res.search_units,
                            "cost_usd": c,
                            "latency_ms": lat,
                        },
                    )
                ]
            )
        return result(res.scores, res.model, c, c, stats.real_latency_ms, lat, False)

    # ------------------------------------------------------- sync helpers
    def generate_sync(self, prompt: str, **kwargs: Any) -> GenerateResult:
        return run_sync(self.generate(prompt, **kwargs))

    def embed_sync(self, texts: Sequence[str], **kwargs: Any) -> EmbedResult:
        return run_sync(self.embed(texts, **kwargs))

    def rank_sync(self, query: str, documents: Sequence[str], **kwargs: Any) -> RankResult:
        return run_sync(self.rank(query, documents, **kwargs))

    # ------------------------------------------------- touched keys / export
    def touched_keys(self) -> list[str]:
        """Cache keys looked up or written by this client, in first-touch order."""
        with self._touched_lock:
            return list(self._touched)

    def reset_touched(self) -> None:
        with self._touched_lock:
            self._touched.clear()

    def export_frozen(self, dest_dir: Path | str, keys: Optional[Iterable[str]] = None) -> dict[str, Any]:
        """Export cache entries (default: every key this client touched) as a frozen shard.

        Returns the manifest plus ``missing`` (keys that were not in any cache layer)."""
        keys = list(dict.fromkeys(keys if keys is not None else self.touched_keys()))
        found = self.cache.get_many(keys)
        manifest = export_frozen(found.values(), Path(dest_dir))
        manifest["missing"] = len([k for k in keys if k not in found])
        return manifest

    # --------------------------------------------------------------- report
    def usage_report(self) -> dict[str, Any]:
        """Plain-dict ledger for results files. Contains no credentials, project id or paths."""
        c = self.config
        totals = {cap: s.as_dict() for cap, s in sorted(self.ledger.totals().items())}
        real = sum(s["real_usd"] for s in totals.values())
        nominal = sum(s["nominal_usd"] for s in totals.values())
        return {
            "schema": "app.llm.usage/v1",
            "budget_usd": c.budget_usd,
            "real_usd": round(real, 8),
            "nominal_usd": round(nominal, 8),
            "budget_remaining_usd": round(max(c.budget_usd - self._spent_usd, 0.0), 8),
            "providers": {
                "generate": {
                    "provider": c.llm_provider,
                    "model": c.default_model("generate"),
                    **(
                        {"location": c.vertex_llm_location or c.google_cloud_location}
                        if c.llm_provider == "vertex"
                        else {}
                    ),
                    **(
                        {"thinking_budget": c.llm_thinking_budget} if c.llm_provider in ("vertex", "openrouter") else {}
                    ),
                },
                "embed": {
                    "provider": c.embed_provider,
                    "model": c.default_model("embed"),
                    "dim": c.embed_dim,
                    **({"location": c.google_cloud_location} if c.embed_provider == "vertex" else {}),
                },
                "rank": {
                    "provider": c.rank_provider,
                    "model": c.default_model("rank"),
                    **({"location": c.vertex_rank_location} if c.rank_provider == "vertex" else {}),
                },
            },
            "models_used": self.ledger.models_used(),
            "totals": totals,
            "by_tag": self.ledger.by_tag(),
            "cache": {
                "enabled": c.cache_enabled,
                "cache_only": c.cache_only,
                "frozen_layers": len(self.cache.frozen),
                "keys_touched": len(self._touched),
            },
        }

    async def aclose(self) -> None:
        for p in list(self._providers.values()):
            await p.aclose()

    def close(self) -> None:
        self.cache.close()


def run_sync[R](coro: Awaitable[R]) -> R:
    """Run a coroutine from synchronous code (scripts). Not usable inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    if asyncio.iscoroutine(coro):
        coro.close()
    raise RuntimeError("run_sync() called from a running event loop; use 'await' instead")


# ------------------------------------------------------- process-wide client
_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def get_client() -> LLMClient:
    """Process-wide client built from settings (created on first use)."""
    global _client
    with _client_lock:
        if _client is None:
            _client = LLMClient(LLMConfig.from_settings())
        return _client


def set_client(client: Optional[LLMClient]) -> None:
    """Replace the process-wide client (tests, scripts with explicit config)."""
    global _client
    with _client_lock:
        _client = client


def reset_client() -> None:
    set_client(None)
