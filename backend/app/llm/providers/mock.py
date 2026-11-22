"""Deterministic, offline provider for tests and CI.

- ``generate``: deterministic text derived from the request; with a schema, a
  schema-valid JSON object. Tests can register canned responders by tag or
  predicate (:meth:`MockProvider.register`).
- ``embed``: feature-hashed bag of word tokens, L2-normalised, so texts that
  share words have higher cosine similarity. Same vector for query/document.
- ``rank``: lexical overlap, ``|q ∩ d| / |q|`` over word-token sets, plus a
  small Jaccard tie-breaker.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from ..errors import ProviderError
from ..types import (
    EmbedTask,
    GenerateRequest,
    ProviderEmbedding,
    ProviderGeneration,
    ProviderRanking,
    Usage,
)
from .base import Provider

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _h(s: str, size: int = 8) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=size).digest(), "big")


def hashed_embedding(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    counts = Counter(tokenize(text))
    for tok, tf in counts.items():
        h = _h(tok)
        idx = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        v[idx] += sign * (1.0 + math.log(tf))
    n = np.linalg.norm(v)
    if n == 0:
        v[0] = 1.0
        n = 1.0
    return (v / n).astype(np.float32)


def lexical_score(query: str, doc: str) -> float:
    q, d = set(tokenize(query)), set(tokenize(doc))
    if not q:
        return 0.0
    inter = len(q & d)
    jac = inter / len(q | d) if (q | d) else 0.0
    return round(min(1.0, 0.99 * inter / len(q) + 0.01 * jac), 6)


# ------------------------------------------------- schema-valid fake objects
def fake_from_schema(schema: dict, defs: Optional[dict] = None, seed: str = "x") -> Any:
    """Build a deterministic value that validates against a (Pydantic) JSON schema."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return fake_from_schema(defs[name], defs, seed)
    if "default" in schema:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    for comb in ("anyOf", "oneOf"):
        if comb in schema:
            opts = [o for o in schema[comb] if o.get("type") != "null"] or schema[comb]
            return fake_from_schema(opts[0], defs, seed)
    if "allOf" in schema:
        return fake_from_schema(schema["allOf"][0], defs, seed)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "null")
    if t == "object" or (t is None and "properties" in schema):
        props = schema.get("properties", {})
        return {k: fake_from_schema(v, defs, f"{seed}.{k}") for k, v in props.items()}
    if t == "array":
        n = max(int(schema.get("minItems", 1)), 1)
        if "maxItems" in schema:
            n = min(n, int(schema["maxItems"]))
        if "prefixItems" in schema:
            return [fake_from_schema(s, defs, f"{seed}[{i}]") for i, s in enumerate(schema["prefixItems"])]
        item = schema.get("items", {})
        return [fake_from_schema(item, defs, f"{seed}[{i}]") for i in range(n)]
    if t == "string":
        s = f"mock-{_h(seed) % 100000:05d}"
        lo, hi = int(schema.get("minLength", 0)), schema.get("maxLength")
        if len(s) < lo:
            s = s + "x" * (lo - len(s))
        if hi is not None:
            s = s[: int(hi)]
        return s
    if t in ("integer", "number"):
        lo = schema.get("minimum", schema.get("exclusiveMinimum"))
        hi = schema.get("maximum", schema.get("exclusiveMaximum"))
        if lo is not None and hi is not None:
            val = (lo + hi) / 2
        elif lo is not None:
            val = lo + 1 if "exclusiveMinimum" in schema else lo
        elif hi is not None:
            val = hi - 1 if "exclusiveMaximum" in schema else hi
        else:
            val = 0
        return int(math.floor(val)) if t == "integer" else float(val)
    if t == "boolean":
        return False
    if t == "null":
        return None
    return None


# ----------------------------------------------------------------- responders
Responder = Any  # str | dict | list | BaseModel | callable(req) -> those


@dataclass
class _Registered:
    response: Responder
    tag: Optional[str]
    predicate: Optional[Callable[[GenerateRequest], bool]]
    times: Optional[int]
    used: int = 0

    def matches(self, req: GenerateRequest) -> bool:
        if self.times is not None and self.used >= self.times:
            return False
        if self.tag is not None and self.tag != req.tag:
            return False
        if self.predicate is not None and not self.predicate(req):
            return False
        return True


def _render(resp: Any, req: GenerateRequest) -> str:
    if callable(resp) and not isinstance(resp, type):
        resp = resp(req)
    if isinstance(resp, BaseModel):
        return resp.model_dump_json()
    if isinstance(resp, (dict, list)):
        return json.dumps(resp)
    return str(resp)


class MockProvider(Provider):
    name = "mock"
    supports_native_schema = True

    def __init__(self, *, cost_usd_per_call: float = 0.0, latency_ms: float = 0.0) -> None:
        self.cost_usd_per_call = cost_usd_per_call
        self.latency_ms = latency_ms
        self._responders: list[_Registered] = []
        self._failures: dict[str, list[ProviderError]] = {"generate": [], "embed": [], "rank": []}
        self.calls: Counter = Counter()
        self.requests: list[Any] = []
        self.embed_batch_size = 64

    # -- test hooks
    def register(
        self,
        response: Responder,
        *,
        tag: Optional[str] = None,
        predicate: Optional[Callable[[GenerateRequest], bool]] = None,
        times: Optional[int] = None,
    ) -> None:
        """Register a canned response. Later registrations win; ``times`` limits uses."""
        self._responders.insert(0, _Registered(response, tag, predicate, times))

    def fail(self, capability: str = "generate", *, times: int = 1, status_code: int = 429) -> None:
        """Make the next ``times`` calls of ``capability`` fail with ``status_code``."""
        from .base import is_retryable_status

        for _ in range(times):
            self._failures[capability].append(
                ProviderError(
                    f"mock failure {status_code}",
                    provider=self.name,
                    status_code=status_code,
                    retryable=is_retryable_status(status_code),
                )
            )

    def clear(self) -> None:
        self._responders.clear()
        for v in self._failures.values():
            v.clear()

    async def _tick(self, capability: str) -> None:
        self.calls[capability] += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if self._failures[capability]:
            raise self._failures[capability].pop(0)

    # -- capabilities
    async def generate(self, req: GenerateRequest) -> ProviderGeneration:
        self.requests.append(req)
        await self._tick("generate")
        text = None
        for r in self._responders:
            if r.matches(req):
                r.used += 1
                text = _render(r.response, req)
                break
        if text is None:
            seed = json.dumps(
                [req.model, req.system, list(req.examples), req.prompt, req.temperature, req.json_schema],
                sort_keys=True,
                default=str,
            )
            if req.json_schema is not None:
                text = json.dumps(fake_from_schema(req.json_schema, seed=seed))
            else:
                words = req.prompt.split()[:12]
                text = f"[mock {_h(seed) % 10**8:08d}] " + " ".join(words)
        in_tok = sum(len(x.split()) for x in [req.system or "", req.prompt] + [u + " " + a for u, a in req.examples])
        return ProviderGeneration(
            text=text,
            model=req.model,
            usage=Usage(input_tokens=in_tok, output_tokens=len(text.split())),
            cost_usd=self.cost_usd_per_call,
            finish_reason="stop",
        )

    def embed_batches(self, texts: Sequence[str]) -> list[list[int]]:
        n = self.embed_batch_size
        return [list(range(i, min(i + n, len(texts)))) for i in range(0, len(texts), n)]

    async def embed(self, texts: Sequence[str], *, model: str, task: EmbedTask, dim: int) -> ProviderEmbedding:
        self.requests.append(("embed", list(texts), task, dim))
        await self._tick("embed")
        vecs = np.stack([hashed_embedding(t, dim) for t in texts]) if texts else np.zeros((0, dim), np.float32)
        return ProviderEmbedding(
            vectors=vecs,
            model=model,
            input_tokens=sum(len(tokenize(t)) for t in texts),
            cost_usd=self.cost_usd_per_call,
        )

    async def rank(self, query: str, documents: Sequence[str], *, model: str) -> ProviderRanking:
        self.requests.append(("rank", query, list(documents)))
        await self._tick("rank")
        return ProviderRanking(
            scores=[lexical_score(query, d) for d in documents],
            model=model,
            cost_usd=self.cost_usd_per_call,
        )
