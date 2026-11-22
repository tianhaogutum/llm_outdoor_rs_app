"""Per-client usage ledger, keyed by (capability, tag)."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Stats:
    calls: int = 0  # client calls (one embed() call counts once, whatever its batch size)
    items: int = 0  # prompts / texts / rank lists handled
    cache_hits: int = 0  # items served from cache
    provider_requests: int = 0  # requests actually sent (incl. retries and schema re-asks)
    retries: int = 0  # transient-error retries
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    real_usd: float = 0.0
    nominal_usd: float = 0.0  # cost as if nothing had been cached
    real_latency_ms: float = 0.0
    nominal_latency_ms: float = 0.0
    schema_violations: int = 0
    schema_failures: int = 0  # calls that raised SchemaValidationError
    errors: int = 0
    budget_blocked: int = 0
    unpriced_requests: int = 0  # provider returned no cost and the model is not in the price table

    def add(self, other: Stats) -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("real_usd", "nominal_usd"):
            d[k] = round(d[k], 8)
        for k in ("real_latency_ms", "nominal_latency_ms"):
            d[k] = round(d[k], 1)
        return d


@dataclass
class ModelUse:
    capability: str
    provider: str
    model: str
    reported_models: set[str] = field(default_factory=set)


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str], Stats] = {}
        self._models: dict[tuple[str, str, str], ModelUse] = {}
        self.real_usd = 0.0

    def record(
        self,
        capability: str,
        tag: str,
        delta: Stats,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        reported_model: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._rows.setdefault((capability, tag), Stats()).add(delta)
            self.real_usd += delta.real_usd
            if provider and model:
                mu = self._models.setdefault((capability, provider, model), ModelUse(capability, provider, model))
                if reported_model:
                    mu.reported_models.add(reported_model)

    def totals(self) -> dict[str, Stats]:
        out: dict[str, Stats] = {}
        with self._lock:
            for (cap, _tag), s in self._rows.items():
                out.setdefault(cap, Stats()).add(s)
        return out

    def by_tag(self) -> dict[str, dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, dict[str, Any]]] = {}
        with self._lock:
            for (cap, tag), s in sorted(self._rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                out.setdefault(tag, {})[cap] = s.as_dict()
        return out

    def models_used(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "capability": m.capability,
                    "provider": m.provider,
                    "model": m.model,
                    "reported_models": sorted(m.reported_models),
                }
                for m in sorted(self._models.values(), key=lambda m: (m.capability, m.provider, m.model))
            ]
