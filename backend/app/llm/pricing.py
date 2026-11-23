"""Price table for providers that do not return the cost (Vertex AI).

OpenRouter returns the real charge in ``usage.cost``; that value is used as is.
Prices below were read from the public pricing pages on 2026-11-22; see
docs/LLM_CLIENT.md for URLs. USD. Update when prices change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenPrice:
    input_per_m: float  # USD per 1M input tokens
    output_per_m: float  # USD per 1M output tokens (thinking tokens bill as output)


# Standard (on-demand) text prices, <= 200k input tokens.
# Gemini 3.x has a regional ("non-global") price 10 % above the global endpoint.
_VERTEX_GENERATE: dict[str, dict[str, TokenPrice]] = {
    "gemini-2.5-flash": {"global": TokenPrice(0.30, 2.50), "regional": TokenPrice(0.30, 2.50)},
    "gemini-2.5-flash-lite": {"global": TokenPrice(0.10, 0.40), "regional": TokenPrice(0.10, 0.40)},
    "gemini-3.5-flash": {"global": TokenPrice(1.50, 9.00), "regional": TokenPrice(1.65, 9.90)},
    "gemini-3.5-flash-lite": {"global": TokenPrice(0.30, 2.50), "regional": TokenPrice(0.33, 2.75)},
    # 3.8 Flash: introductory 0.75 / 3.75 until 2026-12-31; the regular price is used (overestimates, safe for budget)
    "gemini-3.8-flash": {"global": TokenPrice(1.50, 7.50), "regional": TokenPrice(1.65, 8.25)},
}

# Gemini Embedding (gemini-embedding-001): 0.00015 USD per 1,000 input tokens.
_VERTEX_EMBED_PER_M: dict[str, float] = {
    "gemini-embedding-001": 0.15,
}

# Vertex AI Ranking API: 1.00 USD per 1,000 queries; one query = up to 100 records.
VERTEX_RANK_PER_QUERY = 1.00 / 1000
VERTEX_RANK_RECORDS_PER_QUERY = 100


def _base_model(model: str) -> str:
    # "publishers/google/models/gemini-2.5-flash" / "gemini-2.5-flash-001" -> table key
    m = model.rsplit("/", 1)[-1]
    if m in _VERTEX_GENERATE or m in _VERTEX_EMBED_PER_M:
        return m
    for known in sorted(list(_VERTEX_GENERATE) + list(_VERTEX_EMBED_PER_M), key=len, reverse=True):
        if m.startswith(known + "-") or m.startswith(known + "@"):
            return known
    return m


def vertex_generate_cost(
    model: str, location: Optional[str], input_tokens: int, output_tokens: int, thinking_tokens: int
) -> Optional[float]:
    """USD for one generate call, or None if the model is not in the table."""
    prices = _VERTEX_GENERATE.get(_base_model(model))
    if prices is None:
        return None
    p = prices["global" if (location or "global") == "global" else "regional"]
    return (input_tokens * p.input_per_m + (output_tokens + thinking_tokens) * p.output_per_m) / 1e6


def vertex_embed_cost(model: str, input_tokens: int) -> Optional[float]:
    price = _VERTEX_EMBED_PER_M.get(_base_model(model))
    if price is None:
        return None
    return input_tokens * price / 1e6


def vertex_rank_cost(n_records: int) -> float:
    return math.ceil(max(n_records, 1) / VERTEX_RANK_RECORDS_PER_QUERY) * VERTEX_RANK_PER_QUERY


# Anthropic Messages API (no cost in the response). Conservative Haiku-class price; update from
# https://platform.claude.com/docs/en/about-claude/pricing when it changes.
_ANTHROPIC_DEFAULT = TokenPrice(1.00, 5.00)
_ANTHROPIC_GENERATE: dict[str, TokenPrice] = {
    "claude-haiku-5-5": TokenPrice(1.00, 5.00),
}


def anthropic_generate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for one Anthropic call; unknown models use the conservative default so the budget stays safe."""
    base = model.split("@")[0]
    p = _ANTHROPIC_GENERATE.get(base) or next(
        (v for k, v in _ANTHROPIC_GENERATE.items() if base.startswith(k)), _ANTHROPIC_DEFAULT
    )
    return (input_tokens * p.input_per_m + output_tokens * p.output_per_m) / 1e6
