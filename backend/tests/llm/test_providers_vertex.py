"""Vertex provider request construction against fake SDK clients (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import BaseModel

from app.llm import ProviderError
from app.llm.providers.vertex import VertexProvider

PROJECT = "fake-project-42"


class FakeModels:
    def __init__(self):
        self.generate_calls = []
        self.embed_calls = []
        self.fail_with = None

    async def generate_content(self, *, model, contents, config):
        self.generate_calls.append(dict(model=model, contents=contents, config=config))
        if self.fail_with:
            raise self.fail_with
        text = '{"answer": "Tegernsee"}' if config.response_json_schema else "Hello there"
        return SimpleNamespace(
            text=text,
            model_version="gemini-2.5-flash-001",
            usage_metadata=SimpleNamespace(prompt_token_count=1000, candidates_token_count=200, thoughts_token_count=0),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

    async def embed_content(self, *, model, contents, config):
        self.embed_calls.append(dict(model=model, contents=list(contents), config=config))
        dim = config.output_dimensionality
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=list(np.arange(1, dim + 1, dtype=float) * (i + 1)), statistics=None)
                for i in range(len(contents))
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=10 * len(contents)),
        )


class FakeGenai:
    def __init__(self):
        self.models = FakeModels()
        self.aio = SimpleNamespace(models=self.models)


class FakeRank:
    def __init__(self):
        self.requests = []

    async def rank(self, *, request, timeout=None):
        self.requests.append(request)
        # return records in score order (not input order), like the API
        recs = [SimpleNamespace(id=r.id, score=1.0 / (1 + int(r.id))) for r in request.records]
        return SimpleNamespace(records=sorted(recs, key=lambda r: -r.score))


@pytest.fixture
def fakes():
    return {"genai": FakeGenai(), "rank": FakeRank(), "locations": []}


@pytest.fixture
def vertex(fakes):
    def factory(location):
        fakes["locations"].append(location)
        return fakes["genai"]

    return VertexProvider(
        project=PROJECT,
        location="europe-west4",
        genai_client_factory=factory,
        rank_client_factory=lambda: fakes["rank"],
        embed_batch_size=3,
        embed_max_tokens_per_request=1000,
    )


@pytest.fixture
def vclient(make_client, vertex):
    return make_client(
        llm_provider="vertex",
        embed_provider="vertex",
        rank_provider="vertex",
        google_cloud_project=PROJECT,
        google_cloud_location="europe-west4",
        providers={"vertex": vertex},
    )


class Answer(BaseModel):
    answer: str


async def test_generate_request_shape(vclient, fakes):
    r = await vclient.generate(
        "Which lake?", system="You are terse.", examples=[("Q1", "A1")], max_tokens=77, temperature=0.2, tag="t"
    )
    call = fakes["genai"].models.generate_calls[-1]
    assert call["model"] == "gemini-2.5-flash"
    roles = [(c.role, c.parts[0].text) for c in call["contents"]]
    assert roles == [("user", "Q1"), ("model", "A1"), ("user", "Which lake?")]
    cfg = call["config"]
    assert cfg.system_instruction == "You are terse."
    assert cfg.max_output_tokens == 77 and cfg.temperature == 0.2
    assert cfg.thinking_config.thinking_budget == 0  # default: thinking off
    assert cfg.response_mime_type is None
    assert fakes["locations"] == ["europe-west4"]
    # cost from the price table: 1000 in * 0.30/M + 200 out * 2.50/M
    assert r.cost_usd == pytest.approx(1000 * 0.30e-6 + 200 * 2.50e-6)
    assert r.model == "gemini-2.5-flash-001"
    assert vclient.usage_report()["models_used"][0]["reported_models"] == ["gemini-2.5-flash-001"]


async def test_generate_structured_and_thinking_level(vclient, fakes):
    r = await vclient.generate("x", schema=Answer, thinking="low")
    cfg = fakes["genai"].models.generate_calls[-1]["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == Answer.model_json_schema()
    assert str(cfg.thinking_config.thinking_level).lower().endswith("low")
    assert r.parsed == Answer(answer="Tegernsee")


async def test_llm_location_override(make_client, fakes):
    v = VertexProvider(
        project=PROJECT,
        location="europe-west4",
        llm_location="global",
        genai_client_factory=lambda loc: fakes["locations"].append(loc) or fakes["genai"],
    )
    c = make_client(llm_provider="vertex", providers={"vertex": v})
    await c.generate("x")
    assert fakes["locations"] == ["global"]


async def test_embed_batching_task_type_and_dim(vclient, fakes):
    texts = [f"text {i}" for i in range(7)]
    r = await vclient.embed(texts, task="query", dim=4)
    calls = fakes["genai"].models.embed_calls
    assert [len(c["contents"]) for c in calls] == [3, 3, 1]  # batch size 3
    assert all(c["config"].task_type == "RETRIEVAL_QUERY" for c in calls)
    assert all(c["config"].output_dimensionality == 4 for c in calls)
    assert all(c["config"].auto_truncate for c in calls)
    assert r.vectors.shape == (7, 4)
    np.testing.assert_allclose(np.linalg.norm(r.vectors, axis=1), 1.0, rtol=1e-6)
    assert r.cost_usd == pytest.approx(70 * 0.15e-6)
    await vclient.embed(["doc"], task="document", dim=4)
    assert fakes["genai"].models.embed_calls[-1]["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_embed_batches_respect_token_cap(vertex):
    long = "x" * 2400  # ~800 estimated tokens
    batches = vertex.embed_batches([long, long, "short", long])
    assert all(len(b) <= 3 for b in batches)
    assert batches[0] == [0]  # second long text would exceed the 1000-token cap


async def test_rank_request_building(vclient, fakes):
    docs = ["first doc", "second doc", "third doc"]
    r = await vclient.rank("lake hike", docs, top_n=2)
    req = fakes["rank"].requests[-1]
    assert req.ranking_config == f"projects/{PROJECT}/locations/global/rankingConfigs/default_ranking_config"
    assert req.model == "semantic-ranker-default@latest"
    assert req.query == "lake hike"
    assert req.top_n == 3  # always score every record
    assert [(x.id, x.content) for x in req.records] == [("0", "first doc"), ("1", "second doc"), ("2", "third doc")]
    assert req.ignore_record_details_in_response
    assert r.scores == pytest.approx([1.0, 0.5, 1 / 3])  # aligned with input order
    assert r.order == [0, 1]
    assert r.cost_usd == pytest.approx(0.001)


async def test_rank_chunks_over_1000_records(vertex, fakes):
    docs = [f"d{i}" for i in range(1500)]
    res = await vertex.rank("q", docs, model="semantic-ranker-default@latest")
    assert [len(r.records) for r in fakes["rank"].requests] == [1000, 500]
    assert fakes["rank"].requests[1].records[0].id == "1000"
    assert len(res.scores) == 1500
    assert res.cost_usd == pytest.approx(0.015)


async def test_error_mapping_redacts_project(vclient, fakes):
    class APIError(Exception):
        def __init__(self, code, msg):
            super().__init__(msg)
            self.code = code

    fakes["genai"].models.fail_with = APIError(
        404, f"Publisher model projects/{PROJECT}/locations/europe-west4/publishers/google/models/x not found"
    )
    with pytest.raises(ProviderError) as ei:
        await vclient.generate("x")
    assert ei.value.status_code == 404 and not ei.value.retryable
    assert PROJECT not in str(ei.value)
    fakes["genai"].models.fail_with = APIError(429, "Resource exhausted")
    with pytest.raises(ProviderError) as ei:
        await vclient.generate("y", retries=0)
    assert ei.value.retryable


async def test_missing_project_is_config_error(make_client):
    from app.llm import ConfigError

    c = make_client(llm_provider="vertex", google_cloud_project=None)
    with pytest.raises(ConfigError):
        await c.generate("x")
