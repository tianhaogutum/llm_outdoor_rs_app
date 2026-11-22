"""Single LLM / embedding / ranking client with cache, budget and ledger.

See docs/LLM_CLIENT.md.
"""

from .cache import KEY_VERSION, canonical_json, embed_key, generate_key, make_key, rank_key
from .client import LLMClient, get_client, reset_client, run_sync, set_client
from .config import LLMConfig
from .errors import (
    BudgetExceeded,
    CacheMiss,
    ConfigError,
    LLMError,
    ProviderError,
    SchemaValidationError,
)
from .prompts import UNTRUSTED_DATA_RULE, render_llama3, wrap_untrusted
from .types import EmbedResult, GenerateResult, RankResult, Usage

__all__ = [
    "KEY_VERSION",
    "BudgetExceeded",
    "CacheMiss",
    "ConfigError",
    "EmbedResult",
    "GenerateResult",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "ProviderError",
    "RankResult",
    "SchemaValidationError",
    "UNTRUSTED_DATA_RULE",
    "Usage",
    "canonical_json",
    "embed_key",
    "generate_key",
    "get_client",
    "make_key",
    "rank_key",
    "render_llama3",
    "reset_client",
    "run_sync",
    "set_client",
    "wrap_untrusted",
]
