"""OpenRouter and Ollama providers against httpx.MockTransport (no network)."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from app.llm import ProviderError, render_llama3
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openrouter import OpenRouterProvider

KEY = "sk-or-v1-FAKEFAKEFAKE"


class Answer(BaseModel):
    answer: str


class Recorder:
    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.chat_status = [200]
        self.reject_schema = False

    def bodies(self, path):
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith(path)]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        path = request.url.path
        if path.endswith("/chat/completions"):
            status = self.chat_status.pop(0) if len(self.chat_status) > 1 else self.chat_status[0]
            if status != 200:
                return httpx.Response(status, json={"error": {"message": f"upstream {status}", "code": status}})
            if self.reject_schema and "response_format" in body:
                return httpx.Response(400, json={"error": {"message": "response_format unsupported", "code": 400}})
            content = '{"answer": "Tegernsee"}' if ("response_format" in body or "JSON" in str(body)) else "Hi!"
            return httpx.Response(
                200,
                json={
                    "model": "google/gemini-2.5-flash",
                    "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                        "cost": 0.0000123,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            )
        if path.endswith("/embeddings"):
            n, d = len(body["input"]), body["dimensions"] + 2  # return more dims than asked (MRL truncation)
            data = [
                {"object": "embedding", "index": i, "embedding": [float(i + 1)] + [1.0] * (d - 1)} for i in range(n)
            ]
            data.reverse()  # out of order on purpose
            return httpx.Response(
                200,
                json={
                    "model": "gemini-embedding-001",
                    "data": data,
                    "usage": {"prompt_tokens": 4 * n, "total_tokens": 4 * n, "cost": 6e-7 * n},
                },
            )
        if path.endswith("/rerank"):
            docs = body["documents"]
            results = [{"index": i, "relevance_score": 1.0 / (i + 1)} for i in range(len(docs))][::-1]
            return httpx.Response(
                200,
                json={"model": "cohere.rerank-v3-5:0", "results": results, "usage": {"search_units": 1, "cost": 0.002}},
            )
        return httpx.Response(404, json={"error": {"message": "nope", "code": 404}})


@pytest.fixture
def rec():
    return Recorder()


@pytest.fixture
def orclient(make_client, rec):
    prov = OpenRouterProvider(
        api_key=SecretStr(KEY),
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(rec),
        embed_batch_size=2,
    )
    return make_client(
        llm_provider="openrouter",
        embed_provider="openrouter",
        rank_provider="openrouter",
        openrouter_api_key=SecretStr(KEY),
        providers={"openrouter": prov},
    )


async def test_chat_request_and_cost(orclient, rec):
    r = await orclient.generate(
        "Which lake?", system="Be terse.", examples=[("Q", "A")], max_tokens=40, temperature=0.3
    )
    req = rec.requests[-1]
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert str(req.url) == "https://openrouter.test/api/v1/chat/completions"
    body = rec.bodies("/chat/completions")[-1]
    assert body["model"] == "google/gemini-2.5-flash"
    assert body["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "Which lake?"},
    ]
    assert body["max_tokens"] == 40 and body["temperature"] == 0.3
    assert body["reasoning"] == {"enabled": False}
    assert "response_format" not in body
    assert r.text == "Hi!" and r.cost_usd == pytest.approx(0.0000123)
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.thinking_tokens) == (12, 3, 2)


async def test_structured_output_native(orclient, rec):
    r = await orclient.generate("x", schema=Answer)
    body = rec.bodies("/chat/completions")[-1]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "Answer"
    assert rf["json_schema"]["schema"] == Answer.model_json_schema()
    assert r.parsed == Answer(answer="Tegernsee")


async def test_structured_output_fallback_to_prompt_json(orclient, rec):
    rec.reject_schema = True
    r = await orclient.generate("y", schema=Answer)
    bodies = rec.bodies("/chat/completions")
    assert "response_format" in bodies[-2] and "response_format" not in bodies[-1]
    assert "JSON Schema" in bodies[-1]["messages"][0]["content"]
    assert r.parsed == Answer(answer="Tegernsee")
    await orclient.generate("z", schema=Answer)  # remembered: no second rejected attempt
    assert "response_format" not in rec.bodies("/chat/completions")[-1]


async def test_retry_on_429_and_error_redaction(orclient, rec):
    rec.chat_status = [429, 200]
    r = await orclient.generate("retry")
    assert r.text == "Hi!"
    assert orclient.usage_report()["totals"]["generate"]["retries"] == 1
    rec.chat_status = [401]
    with pytest.raises(ProviderError) as ei:
        await orclient.generate("unauthorised")
    assert ei.value.status_code == 401 and KEY not in str(ei.value)


async def test_embeddings_request_order_and_truncation(orclient, rec):
    r = await orclient.embed(["a", "b", "c"], task="query", dim=4)
    bodies = rec.bodies("/embeddings")
    assert [b["input"] for b in bodies] == [["a", "b"], ["c"]]
    assert all(b["dimensions"] == 4 and b["model"] == "google/gemini-embedding-001" for b in bodies)
    assert r.vectors.shape == (3, 4)
    assert r.cost_usd == pytest.approx(1.8e-6)
    # rows came back reversed; vectors must follow input order (row i of a batch = [i+1, 1, 1, 1] normalised)
    assert r.vectors[0][0] == pytest.approx(0.5)
    assert r.vectors[1][0] == pytest.approx(2 / 7**0.5)
    assert r.vectors[2][0] == pytest.approx(0.5)


async def test_rerank_request_and_alignment(orclient, rec):
    docs = ["d0", "d1", "d2", "d3"]
    r = await orclient.rank("lake", docs, top_n=2)
    body = rec.bodies("/rerank")[-1]
    assert body == {"model": "cohere/rerank-v3.5", "query": "lake", "documents": docs, "top_n": 4}
    assert r.scores == pytest.approx([1.0, 0.5, 1 / 3, 0.25])
    assert r.order == [0, 1]
    assert r.cost_usd == pytest.approx(0.002)
    assert r.model == "cohere.rerank-v3-5:0"


async def test_missing_key_is_config_error(make_client):
    from app.llm import ConfigError

    c = make_client(rank_provider="openrouter", openrouter_api_key=None)
    with pytest.raises(ConfigError):
        await c.rank("q", ["d"])


# ---------------------------------------------------------------- ollama


async def test_ollama_payload_matches_legacy_shape(make_client):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "llama3.1:8b",
                "response": "  Welcome!  ",
                "done": True,
                "prompt_eval_count": 50,
                "eval_count": 3,
            },
        )

    prov = OllamaProvider(api_url="http://ollama.test/api/generate", transport=httpx.MockTransport(handler))
    c = make_client(llm_provider="ollama", providers={"ollama": prov})
    r = await c.generate("Profile: x", system="SYS", examples=[("u", "a")], max_tokens=170, temperature=0.6)
    assert r.text == "Welcome!" and r.cost_usd == 0.0
    assert seen[0] == {
        "model": "llama3.1:8b",
        "prompt": render_llama3("SYS", [("u", "a")], "Profile: x"),
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 170},
    }
    assert seen[0]["prompt"].startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSYS<|eot_id|>")
    assert seen[0]["prompt"].endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


async def test_ollama_errors(make_client):
    def handler(request):
        return httpx.Response(503, text="model loading")

    prov = OllamaProvider(api_url="http://ollama.test/api/generate", transport=httpx.MockTransport(handler))
    c = make_client(llm_provider="ollama", providers={"ollama": prov})
    with pytest.raises(ProviderError) as ei:
        await c.generate("x", retries=0)
    assert ei.value.status_code == 503

    def not_done(request):
        return httpx.Response(200, json={"response": "", "done": False})

    prov2 = OllamaProvider(api_url="http://ollama.test/api/generate", transport=httpx.MockTransport(not_done))
    c2 = make_client(llm_provider="ollama", providers={"ollama": prov2})
    with pytest.raises(ProviderError, match="Empty response"):
        await c2.generate("x")
