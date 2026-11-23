"""Is a provider usable without making a call? (docs/DEPLOY.md, "Running without credentials").

A cheap, local check used to skip LLM calls that cannot succeed (no Vertex project, no OpenRouter key), so a
container started without credentials serves its offline paths straight away instead of failing a call,
logging a stack trace and then falling back. It does not prove that credentials work: a configured provider
can still fail at call time (expired ADC, network), and every caller keeps its own fallback for that.
"""

from __future__ import annotations

from typing import Any, Optional


def provider_configured(provider: str, settings: Any, *, location: Optional[str] = None) -> bool:
    """True if ``provider`` has what it needs to attempt a call.

    - ``mock``: always (offline). ``ollama``: always (a local server; unreachable shows up as a call error).
    - ``openrouter``: ``OPENROUTER_API_KEY`` is set.
    - ``anthropic``: ``ANTHROPIC_API_KEY`` is set.
    - ``vertex``: ``GOOGLE_CLOUD_PROJECT`` and a location (``location`` if the caller has its own, else
      ``VERTEX_LLM_LOCATION`` or ``GOOGLE_CLOUD_LOCATION``). Application Default Credentials are not checked
      here (that would need the Google auth library and, on some hosts, a metadata-server round trip).
    """
    if provider in ("mock", "ollama"):
        return True
    if provider == "openrouter":
        key = getattr(settings, "openrouter_api_key", None)
        if key is None:
            return False
        value = key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
        return bool(value.strip())
    if provider == "anthropic":
        key = getattr(settings, "anthropic_api_key", None)
        if key is None:
            return False
        value = key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
        return bool(value.strip())
    if provider == "vertex":
        loc = location or getattr(settings, "vertex_llm_location", None) or getattr(
            settings, "google_cloud_location", None
        )
        return bool(getattr(settings, "google_cloud_project", None)) and bool(loc)
    return False


def generation_configured(settings: Any = None) -> bool:
    """Can the narrative client (``LLM_PROVIDER``) attempt a generate call?"""
    if settings is None:
        from app.settings import get_settings

        settings = get_settings()
    return provider_configured(settings.llm_provider, settings)


def chat_configured(settings: Any = None) -> bool:
    """Can the chat client attempt an extraction call? (mock answers with the keyword rules anyway.)"""
    if settings is None:
        from app.settings import get_settings

        settings = get_settings()
    from app.chat.llm import resolve_chat_model

    cm = resolve_chat_model(settings)
    return provider_configured(cm.provider, settings, location=cm.location)
