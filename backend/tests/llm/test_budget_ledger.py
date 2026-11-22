"""Budget hard stop, ledger contents, usage report, retries, concurrency bound."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import SecretStr

from app.llm import BudgetExceeded, LLMConfig, ProviderError
from app.llm.providers.mock import MockProvider


async def test_budget_hard_stop(make_client):
    c = make_client(budget_usd=0.025, mock_cost_usd_per_call=0.01)
    await c.generate("one")
    await c.generate("two")
    await c.generate("three")  # starts at 0.02 < 0.025: allowed
    assert c.spent_usd == pytest.approx(0.03)
    with pytest.raises(BudgetExceeded):
        await c.generate("four")
    # cache hits cost nothing and still work after the budget is gone
    r = await c.generate("one")
    assert r.cached and r.cost_usd == 0.0
    with pytest.raises(BudgetExceeded):
        await c.embed(["new text"])
    with pytest.raises(BudgetExceeded):
        await c.rank("q", ["d"])
    rep = c.usage_report()
    assert rep["real_usd"] == pytest.approx(0.03)
    assert rep["budget_remaining_usd"] == 0.0
    assert rep["totals"]["generate"]["budget_blocked"] == 1


async def test_free_providers_ignore_exhausted_budget(make_client):
    c = make_client(budget_usd=0.0)  # mock with zero cost is free
    r = await c.generate("free")
    assert r.text


async def test_ledger_per_tag_and_capability(make_client):
    c = make_client(mock_cost_usd_per_call=0.001)
    await c.generate("a", tag="judge")
    await c.generate("a", tag="judge")  # hit
    await c.generate("b", tag="chat")
    await c.embed(["x", "y", "z"], tag="docs", dim=16)
    await c.embed(["x", "w"], tag="queries", task="query", dim=16)
    await c.rank("q", ["d1", "d2"], tag="rerank")
    rep = c.usage_report()
    j = rep["by_tag"]["judge"]["generate"]
    assert (j["calls"], j["cache_hits"], j["provider_requests"]) == (2, 1, 1)
    assert j["real_usd"] == pytest.approx(0.001) and j["nominal_usd"] == pytest.approx(0.002)
    assert j["input_tokens"] > 0 and j["output_tokens"] > 0
    d = rep["by_tag"]["docs"]["embed"]
    assert (d["calls"], d["items"], d["cache_hits"]) == (1, 3, 0)
    q = rep["by_tag"]["queries"]["embed"]
    assert (q["items"], q["cache_hits"]) == (2, 0)  # task=query: separate entries
    assert rep["totals"]["generate"]["calls"] == 3
    assert rep["totals"]["rank"]["real_usd"] == pytest.approx(0.001)
    models = {(m["capability"], m["provider"], m["model"]) for m in rep["models_used"]}
    assert models == {
        ("generate", "mock", "mock-llm-1"),
        ("embed", "mock", "mock-embed-1"),
        ("rank", "mock", "mock-rank-1"),
    }
    assert rep["providers"]["embed"] == {"provider": "mock", "model": "mock-embed-1", "dim": 768}
    json.dumps(rep)  # plain, serialisable


def test_report_and_reprs_contain_no_secrets(make_client):
    secret = "sk-or-v1-THIS-IS-NOT-A-REAL-KEY"
    c = make_client(
        openrouter_api_key=SecretStr(secret),
        google_cloud_project="my-secret-project-123",
        google_cloud_location="europe-west4",
        rank_provider="openrouter",
    )
    blob = json.dumps(c.usage_report()) + repr(c) + repr(c.config) + repr(c.provider("openrouter"))
    assert secret not in blob
    assert "my-secret-project-123" not in json.dumps(c.usage_report())


async def test_retry_on_429_then_success(client):
    client.mock.fail("generate", times=2, status_code=429)
    r = await client.generate("retry me")
    assert r.text
    t = client.usage_report()["totals"]["generate"]
    assert t["retries"] == 2 and t["provider_requests"] == 3


async def test_retry_exhaustion_raises_provider_error(make_client):
    c = make_client(max_retries=2)
    c.mock.fail("embed", times=5, status_code=503)
    with pytest.raises(ProviderError) as ei:
        await c.embed(["x"])
    assert ei.value.status_code == 503 and ei.value.retryable
    assert c.usage_report()["totals"]["embed"]["retries"] == 2


async def test_no_retry_on_400(client):
    client.mock.fail("rank", times=1, status_code=400)
    with pytest.raises(ProviderError):
        await client.rank("q", ["d"])
    assert client.usage_report()["totals"]["rank"]["retries"] == 0


async def test_per_call_retry_override(client):
    client.mock.fail("generate", times=3, status_code=500)
    with pytest.raises(ProviderError):
        await client.generate("fast fail", retries=1)
    assert client.mock.calls["generate"] == 2


async def test_bounded_concurrency(make_client):
    class Probe(MockProvider):
        def __init__(self):
            super().__init__()
            self.inflight = 0
            self.peak = 0

        async def generate(self, req):
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            await asyncio.sleep(0.005)
            self.inflight -= 1
            return await super().generate(req)

    p = Probe()
    c = make_client(max_concurrency=3, providers={"mock": p})
    await asyncio.gather(*(c.generate(f"p{i}") for i in range(20)))
    assert p.peak == 3


def test_config_validation():
    from app.llm import ConfigError

    with pytest.raises(ConfigError):
        LLMConfig(llm_provider="nope")
    with pytest.raises(ConfigError):
        LLMConfig(embed_provider="ollama")
