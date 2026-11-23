"""The chat's own LLM client (model, location, thinking and budget separate from the narratives).

Defaults per provider are in :data:`CHAT_DEFAULTS` (chosen by the scripted comparison, docs/CHAT.md
section 7); ``CHAT_LLM_*`` settings override them. With ``LLM_PROVIDER=mock`` the chat is always mock and
the keyword extractor (:mod:`app.chat.rules`) answers, so tests and CI stay offline.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional, Union

from app.llm import LLMClient, LLMConfig

Thinking = Union[int, str, None]


@dataclass(frozen=True)
class ChatModel:
    provider: str
    model: str
    location: Optional[str] = None  # vertex only; None = the configured region
    thinking: Thinking = 0


# Default chat model per provider (docs/CHAT.md section 7 and the DECISIONS entry it proposes). The
# overall default is the OpenRouter entry (Claude Haiku 5.5): same slot quality as gemini-3.5-flash on the
# scripted set at a tenth of the cost and lower latency; Vertex gemini-3.5-flash is the fallback.
CHAT_DEFAULTS: dict[str, ChatModel] = {
    "vertex": ChatModel("vertex", "gemini-3.5-flash", location="global", thinking="minimal"),
    "openrouter": ChatModel("openrouter", "anthropic/claude-haiku-5.5", thinking=0),
    "ollama": ChatModel("ollama", "llama3.1:8b", thinking=None),
    "mock": ChatModel("mock", "mock-llm-1", thinking=None),
}


def _parse_thinking(value: Optional[str]) -> Thinking:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def default_chat_provider(settings: Any) -> str:
    """mock under LLM_PROVIDER=mock; else CHAT_LLM_PROVIDER; else OpenRouter (the chosen default model,
    docs/CHAT.md section 7) when a key is configured and the app uses a cloud provider; else LLM_PROVIDER."""
    if settings.llm_provider == "mock":
        return "mock"
    if settings.chat_llm_provider:
        return settings.chat_llm_provider
    if settings.llm_provider in ("vertex", "openrouter") and settings.openrouter_api_key is not None:
        return "openrouter"
    return settings.llm_provider


def resolve_chat_model(settings: Any = None) -> ChatModel:
    if settings is None:
        from app.settings import get_settings

        settings = get_settings()
    provider = default_chat_provider(settings)
    base = CHAT_DEFAULTS[provider]
    if provider == "mock":
        return base
    thinking = _parse_thinking(settings.chat_thinking)
    model = settings.chat_llm_model or base.model
    location = settings.chat_vertex_location or (base.location if model == base.model else None)
    return ChatModel(provider, model, location, base.thinking if thinking is None else thinking)


def build_chat_client(chat_model: ChatModel, *, config: Optional[LLMConfig] = None, budget_usd: float = 1.0,
                      **overrides: Any) -> LLMClient:
    """A client whose generate default is ``chat_model`` (plus ``overrides`` on the config)."""
    cfg = config or LLMConfig.from_settings()
    o: dict[str, Any] = {"llm_provider": chat_model.provider, "budget_usd": budget_usd}
    if chat_model.provider == "vertex":
        o["vertex_llm_model"] = chat_model.model
        if chat_model.location:
            o["vertex_llm_location"] = chat_model.location
    elif chat_model.provider == "openrouter":
        o["openrouter_llm_model"] = chat_model.model
    elif chat_model.provider == "ollama":
        o["ollama_model"] = chat_model.model
    o.update(overrides)
    client = LLMClient(cfg.with_(**o))
    client._chat_model = chat_model  # type: ignore[attr-defined]  # read by chat_call_options
    install_mock_responder(client)
    return client


def install_mock_responder(client: LLMClient) -> None:
    """Make the mock provider answer chat extraction calls with the keyword extractor (idempotent)."""
    from .extract import EXTRACT_TAG
    from .rules import mock_responder

    if client.config.llm_provider != "mock" or getattr(client, "_chat_mock_installed", False):
        return
    client.mock.register(mock_responder, tag=EXTRACT_TAG)
    client._chat_mock_installed = True  # type: ignore[attr-defined]


def chat_call_options(client: LLMClient, settings: Any = None) -> dict[str, Any]:
    """Per-call options for ``client.generate`` (thinking of the chat model)."""
    cm: Optional[ChatModel] = getattr(client, "_chat_model", None)
    if cm is None:
        install_mock_responder(client)
        return {}
    return {"thinking": cm.thinking} if cm.thinking is not None else {}


_client: Optional[LLMClient] = None
_lock = threading.Lock()


def get_chat_client() -> LLMClient:
    global _client
    with _lock:
        if _client is None:
            from app.settings import get_settings

            s = get_settings()
            _client = build_chat_client(resolve_chat_model(s), budget_usd=s.chat_budget_usd)
        return _client


def set_chat_client(client: Optional[LLMClient]) -> None:
    global _client
    with _lock:
        _client = client
        if client is not None:
            install_mock_responder(client)
