"""genai_service narrative functions against the mock provider, prompt parity and error mapping."""

from __future__ import annotations

import re

import pytest
from fastapi import HTTPException

from app.api.schemas import ProfileCreate
from app.llm import render_llama3
from app.services import genai_service as gs
from tests.llm.legacy_prompts import legacy_post_run_prompt, legacy_welcome_prompt


def _norm(s: str) -> str:
    # the legacy f-strings contained indentation-only lines; nothing else may differ
    return re.sub(r"\n[ \t]+\n", "\n\n", s)


@pytest.mark.parametrize(
    "q",
    [
        ProfileCreate(fitness="beginner", type=["family-fun"], narrative="playful"),
        ProfileCreate(fitness="advanced", type=[], narrative="unknown-style"),
    ],
)
def test_welcome_prompt_llama_rendering_matches_legacy(q):
    assert render_llama3(*gs.build_welcome_prompt(q)) == _norm(legacy_welcome_prompt(q))


def test_post_run_prompt_llama_rendering_matches_legacy():
    args = ("Herzogstand Loop", 9.3, 2, 3, 4)
    assert render_llama3(*gs.build_post_run_prompt(*args)) == _norm(legacy_post_run_prompt(*args))
    args0 = ("No Quests", 1.0, 0, 0, 1)
    assert render_llama3(*gs.build_post_run_prompt(*args0)) == _norm(legacy_post_run_prompt(*args0))


def test_clean_prompts_have_no_template_tokens():
    q = ProfileCreate(fitness="beginner", type=["family-fun"], narrative="playful")
    system, examples, user = gs.build_welcome_prompt(q)
    for text in [system, user, *[x for pair in examples for x in pair]]:
        assert "<|" not in text


async def test_generate_welcome_summary_mock(global_client):
    global_client.mock.register("Welcome, Trail Seeker! Mock text.", tag="narrative.welcome")
    q = ProfileCreate(fitness="intermediate", type=["natural-scenery"], narrative="mystery")
    text = await gs.generate_welcome_summary(q)
    assert text == "Welcome, Trail Seeker! Mock text."
    req = global_client.mock.requests[-1]
    assert req.system.startswith("You are the TrailSaga")
    assert len(req.examples) == 3
    assert req.prompt == "Profile:\n- Fitness: intermediate\n- Type: natural-scenery\n- Narrative: mystery"
    assert (req.max_tokens, req.temperature) == (170, 0.6)
    # narratives are not cached
    await gs.generate_welcome_summary(q)
    assert global_client.mock.calls["generate"] == 2
    rep = global_client.usage_report()
    assert rep["by_tag"]["narrative.welcome"]["generate"]["calls"] == 2


async def test_generate_welcome_summary_strips_markdown(global_client):
    global_client.mock.register("**Welcome, Hero!**\n# Ready", tag="narrative.welcome")
    q = ProfileCreate(fitness="beginner", type=["family-fun"], narrative="playful")
    assert await gs.generate_welcome_summary(q) == "Welcome, Hero! Ready"


async def test_generate_post_run_summary_mock(global_client):
    text = await gs.generate_post_run_summary("Lake Loop", 7.5, 2, 3, 4)
    assert text.startswith("[mock ")
    req = global_client.mock.requests[-1]
    assert req.prompt == "Route: Lake Loop\nDistance: 7.5 km\nQuests: 2/3 (67%)\nLevel: 4"
    assert (req.max_tokens, req.temperature, len(req.examples)) == (150, 0.75, 2)


async def test_call_llm_http_error_mapping(make_client):
    from app.llm import set_client

    c = make_client(max_retries=0)
    set_client(c)
    try:
        c.mock.fail("generate", times=2, status_code=503)  # call_llm retries once
        with pytest.raises(HTTPException) as ei:
            await gs.call_llm("x")
        assert ei.value.status_code == 503

        c.mock.register("")  # empty response -> 500
        with pytest.raises(HTTPException) as ei:
            await gs.call_llm("y")
        assert ei.value.status_code == 500
        c.mock.clear()

        from app.llm import ProviderError

        for _ in range(2):
            c.mock._failures["generate"].append(ProviderError("t", provider="mock", timeout=True, retryable=True))
        with pytest.raises(HTTPException) as ei:
            await gs.call_llm("z")
        assert ei.value.status_code == 504
    finally:
        set_client(None)


async def test_call_llm_budget_exhausted_maps_to_503(make_client):
    from app.llm import set_client

    c = make_client(budget_usd=0.001, mock_cost_usd_per_call=0.01)
    set_client(c)
    try:
        await gs.call_llm("first")
        with pytest.raises(HTTPException) as ei:
            await gs.call_llm("second")
        assert ei.value.status_code == 503
    finally:
        set_client(None)


async def test_call_ollama_compat_wrapper(global_client):
    out = await gs.call_ollama("plain prompt", max_tokens=10, temperature=0.1)
    assert out
    assert global_client.mock.requests[-1].tag == "narrative.raw"
