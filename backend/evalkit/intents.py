"""Free-text intents written by an LLM from the persona's latent profile.

The writer is a different vendor from the relevance judge (OpenAI via OpenRouter writes, Google
judges), so the judge does not grade text phrased by itself. Calls are cached (temperature > 0 but
the cache stores the first answer), so ``personas.jsonl`` is deterministic from cache.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from app.llm import UNTRUSTED_DATA_RULE, LLMClient, wrap_untrusted
from evalkit.cards import latent_lines
from evalkit.personas import Persona

WRITER = {
    "provider": "openrouter",
    "model": "openai/gpt-6-luna",
    "temperature": 0.9,
    "max_tokens": 300,
    "thinking": 0,
    "prompt_version": "intent/v2",
}

STYLE_TEXT = {
    "vague": "Very short and vague, 3 to 9 words, like a lazy search query. Mention at most one wish. "
    "Leave out numbers.",
    "short": "One short sentence (8 to 18 words). Mention the one or two things that matter most.",
    "medium": "One or two sentences (15 to 35 words). Mention the main wishes and one practical constraint "
    "(who is coming, how far or how hard).",
    "detailed": "Two sentences (25 to 50 words). Mention the main wishes and one or two constraints, "
    "the way someone explains what they want to a friend who knows the area.",
}

AVOID = (
    "Refreshment stops available",
    "Round trip",
    "Linear route",
    "Scenic",
    "Family-friendly",
    "Cultural/historical interest",
    "Summit route",
    "Public transport access",
    "Mostly paved",
)

VOICE_TEXT = {
    "casual": "relaxed and informal, like a quick message",
    "practical": "matter-of-fact, focused on logistics",
    "enthusiastic": "keen and a little excited",
    "terse": "clipped, fragments are fine, no greeting",
    "polite": "polite and slightly formal",
}

SYSTEM = (
    "You write the text a real person types into the search box of an app that recommends hiking and "
    "cycling routes in Bavaria. You are given a private profile of that person. Write what THIS person "
    "would type, in English, in their own words. Rules: mention only what is on their mind (usually the "
    "one to three main wishes and perhaps one constraint) and leave the rest out; paraphrase instead of "
    "copying phrases or lists from the profile (pick one concrete thing, e.g. 'a castle', rather than "
    "listing alternatives); people round numbers ('about 10 km', 'half a day', 'not too much climbing') "
    "and never quote exact figures from the profile; no technical labels such as 'SAC' or 'T2'; do not "
    "start with 'I'm looking for'; never use these catalogue phrases: " + "; ".join(AVOID) + ". "
    "Do not invent wishes that are not in the profile. " + UNTRUSTED_DATA_RULE
)


class Intent(BaseModel):
    intent: str = Field(description="the text the person types, nothing else")


def writer_config() -> dict[str, Any]:
    return dict(WRITER)


def intent_prompt(p: Persona) -> str:
    meta = p.intent_meta or {}
    style = meta.get("style", "medium")
    home = p.latent["home"]["town"]
    home_rule = (
        f"You may mention that they start from {home} or its area."
        if meta.get("mention_home")
        else "Do not mention where they live."
    )
    return (
        "Private profile of the person:\n" + wrap_untrusted("\n".join(latent_lines(p)), "profile") + "\n\n"
        f"Length: {STYLE_TEXT[style]}\nVoice: {VOICE_TEXT[meta.get('voice', 'casual')]}\n{home_rule}\n"
        'Answer as JSON: {"intent": "..."}'
    )


def make_writer_client(**overrides: Any) -> LLMClient:
    """OpenRouter client for the writer model; stays on the mock provider when LLM_PROVIDER=mock (tests, CI)."""
    from app.llm import LLMConfig
    from evalkit.llm import make_client

    if LLMConfig.from_settings().llm_provider == "mock":
        return make_client("intent-writer", **overrides)
    return make_client(
        "intent-writer", llm_provider=WRITER["provider"], openrouter_llm_model=WRITER["model"], **overrides
    )


async def _write_all(personas: list[Persona], client: LLMClient) -> None:
    # de-duplicate identical prompts (the client does not de-duplicate in-flight calls)
    prompts: dict[str, list[Persona]] = {}
    for p in personas:
        prompts.setdefault(intent_prompt(p), []).append(p)

    async def one(prompt: str) -> tuple[str, Any]:
        res = await client.generate(
            prompt,
            system=SYSTEM,
            schema=Intent,
            temperature=WRITER["temperature"],
            max_tokens=WRITER["max_tokens"],
            thinking=WRITER["thinking"],
            tag="persona-intent",
        )
        return prompt, res

    results = await asyncio.gather(*(one(pr) for pr in prompts))
    for prompt, res in results:
        text = " ".join(res.parsed.intent.split())
        for p in prompts[prompt]:
            p.intent = text
            p.intent_meta = {
                **(p.intent_meta or {}),
                "model": res.model,
                "provider": res.provider,
                "prompt_version": WRITER["prompt_version"],
            }


def add_intents(personas: list[Persona], client: LLMClient | None = None) -> None:
    from app.llm import run_sync

    client = client or make_writer_client()
    run_sync(_write_all(personas, client))
