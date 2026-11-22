"""Cache keys, hit/miss, per-text embedding cache, frozen layers, concurrency."""

from __future__ import annotations

import asyncio
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

from app.llm import CacheMiss, LLMClient, LLMConfig, embed_key, generate_key, rank_key
from app.llm.cache import LocalCache
from app.llm.providers.base import Provider

# ------------------------------------------------------------------ keys


def test_generate_key_is_hard_coded_stable():
    key = generate_key(
        provider="vertex",
        model="gemini-2.5-flash",
        prompt="Name one lake in Bavaria.",
        system="Be brief.",
        examples=[("Hi", "Hello")],
        temperature=0.0,
        max_tokens=64,
        json_schema=None,
        thinking=0,
    )
    assert key == "66b28ff2b253fbe711b2be4a2d9c3d94bd370b1f81db6964b5722c9ac726bcbf"
    # and it is literally sha256 of the documented canonical JSON
    doc = (
        '{"cap":"generate","examples":[["Hi","Hello"]],"max_tokens":64,"model":"gemini-2.5-flash",'
        '"prompt":"Name one lake in Bavaria.","provider":"vertex","schema":null,"system":"Be brief.",'
        '"temperature":0.0,"thinking":0,"v":1}'
    )
    assert hashlib.sha256(doc.encode()).hexdigest() == key


def test_embed_and_rank_keys_hard_coded():
    assert (
        embed_key(
            provider="vertex",
            model="gemini-embedding-001",
            text="Lakeside loop near Tegernsee",
            task="document",
            dim=768,
        )
        == "6542d3f65694814a6597a60d7ae7fb5fece4a228c7adbf94c94b5873ab758c92"
    )
    assert (
        rank_key(provider="openrouter", model="cohere/rerank-v3.5", query="lake hike", documents=["a", "b"])
        == "7d7d38ec3da7c584f8bb73f91837f815e4d261e3170ab93b8ffaf599ab8c1787"
    )


def test_key_depends_on_output_relevant_fields_only():
    base = dict(
        provider="mock",
        model="m",
        prompt="p",
        system=None,
        examples=(),
        temperature=0.0,
        max_tokens=10,
        json_schema=None,
        thinking=None,
    )
    k = generate_key(**base)
    assert generate_key(**{**base, "temperature": 0}) == k  # int vs float normalised
    for change in [
        {"prompt": "q"},
        {"model": "n"},
        {"temperature": 0.5},
        {"max_tokens": 11},
        {"system": "s"},
        {"json_schema": {"type": "object"}},
        {"thinking": 0},
        {"provider": "vertex"},
    ]:
        assert generate_key(**{**base, **change}) != k
    assert embed_key(provider="mock", model="m", text="t", task="query", dim=8) != embed_key(
        provider="mock", model="m", text="t", task="document", dim=8
    )


async def test_tag_does_not_change_key(client):
    a = await client.generate("hello", tag="a")
    b = await client.generate("hello", tag="b")
    assert a.cache_key == b.cache_key and b.cached


# ------------------------------------------------------------ hit / miss


async def test_generate_hit_miss_and_determinism(client):
    r1 = await client.generate("Describe a lake walk", max_tokens=50)
    r2 = await client.generate("Describe a lake walk", max_tokens=50)
    assert not r1.cached and r2.cached
    assert r1.text == r2.text
    assert client.mock.calls["generate"] == 1
    r3 = await client.generate("Describe a lake walk", max_tokens=51)
    assert not r3.cached


async def test_cache_survives_new_client(make_client):
    a = make_client()
    r1 = await a.generate("persist me")
    b = make_client()
    r2 = await b.generate("persist me")
    assert r2.cached and r2.text == r1.text
    assert b.mock.calls["generate"] == 0


async def test_cache_false_bypasses_and_does_not_store(client):
    await client.generate("fresh", cache=False)
    await client.generate("fresh", cache=False)
    assert client.mock.calls["generate"] == 2
    r = await client.generate("fresh")
    assert not r.cached


async def test_cache_disabled_globally(make_client):
    c = make_client(cache_enabled=False)
    await c.generate("x")
    r = await c.generate("x")
    assert not r.cached and c.mock.calls["generate"] == 2


async def test_cached_replay_reports_nominal_cost_and_latency(make_client):
    c = make_client(mock_cost_usd_per_call=0.01, mock_latency_ms=20)
    r1 = await c.generate("priced")
    r2 = await c.generate("priced")
    assert r1.cost_usd == pytest.approx(0.01)
    assert r2.cost_usd == 0.0 and r2.nominal_cost_usd == pytest.approx(0.01)
    assert r2.nominal_latency_ms == pytest.approx(r1.nominal_latency_ms)
    assert r1.nominal_latency_ms >= 15


async def test_rank_cache(client):
    docs = ["lake walk", "museum", "hike by the lake"]
    a = await client.rank("lake hike", docs, top_n=2)
    b = await client.rank("lake hike", docs, top_n=1)  # top_n does not change the key
    assert b.cached and a.scores == b.scores
    assert b.order == [2]


# ---------------------------------------------------------- embeddings


async def test_embed_per_text_cache(client):
    r1 = await client.embed(["alpha lake", "beta forest"], task="document", dim=64)
    assert r1.n_cached == 0 and r1.vectors.shape == (2, 64) and r1.vectors.dtype == np.float32
    r2 = await client.embed(["beta forest", "gamma ridge", "alpha lake"], task="document", dim=64)
    assert r2.n_cached == 2
    sent = [req for req in client.mock.requests if req[0] == "embed"]
    assert sent[-1][1] == ["gamma ridge"]  # only the unknown text went to the provider
    np.testing.assert_allclose(r2.vectors[0], r1.vectors[1])
    np.testing.assert_allclose(np.linalg.norm(r2.vectors, axis=1), 1.0, rtol=1e-5)
    # different task or dim: different entries
    r3 = await client.embed(["alpha lake"], task="query", dim=64)
    assert r3.n_cached == 0


async def test_embed_duplicate_texts_in_one_batch(client):
    r = await client.embed(["same", "same", "other"], dim=32)
    assert r.vectors.shape == (3, 32)
    sent = [req for req in client.mock.requests if req[0] == "embed"]
    assert sorted(sent[-1][1]) == ["other", "same"]


async def test_embed_empty_list(client):
    r = await client.embed([], dim=16)
    assert r.vectors.shape == (0, 16)


# -------------------------------------------------------------- frozen


class ExplodingProvider(Provider):
    name = "vertex"

    async def generate(self, req):  # pragma: no cover - must not be called
        raise AssertionError("provider called during replay")

    async def embed(self, *a, **k):  # pragma: no cover
        raise AssertionError("provider called during replay")

    async def rank(self, *a, **k):  # pragma: no cover
        raise AssertionError("provider called during replay")


async def _record_run(c: LLMClient):
    g = await c.generate("judge this", tag="judge")
    e = await c.embed(["route one by the lake", "route two in the hills"], task="document", dim=48, tag="docs")
    r = await c.rank("lake", ["route one by the lake", "route two in the hills"], tag="rerank")
    return g, e, r


async def test_frozen_export_and_fresh_replay_without_provider(make_client, tmp_path):
    rec = make_client(cache_dir=tmp_path / "rec_cache")
    g, e, r = await _record_run(rec)
    await rec.generate("not part of the export", cache=False)
    keys = rec.touched_keys()
    assert len(keys) == 1 + 2 + 1
    manifest = rec.export_frozen(tmp_path / "frozen")
    assert manifest["entries"] == 4 and manifest["missing"] == 0
    assert manifest["by_capability"] == {"embed": 2, "generate": 1, "rank": 1}
    files = sorted(p.name for p in (tmp_path / "frozen").iterdir())
    assert any(f.startswith("entries-") and f.endswith(".jsonl.gz") for f in files)
    assert any(f.startswith("vectors-") and f.endswith(".npz") for f in files)

    # fresh "clone": empty local cache, frozen layer only, a provider that explodes, cache-only mode
    fresh = make_client(
        cache_dir=tmp_path / "empty_cache",
        frozen_dirs=(tmp_path / "frozen",),
        cache_only=True,
        providers={"mock": ExplodingProvider()},
    )
    g2, e2, r2 = await _record_run(fresh)
    assert g2.cached and g2.text == g.text
    assert e2.cached and np.array_equal(e2.vectors, e.vectors)
    assert r2.cached and r2.scores == r.scores
    rep = fresh.usage_report()
    assert rep["real_usd"] == 0.0
    assert rep["totals"]["generate"]["cache_hits"] == 1
    with pytest.raises(CacheMiss):
        await fresh.generate("never recorded")


async def test_export_is_byte_deterministic(make_client, tmp_path):
    c = make_client()
    await _record_run(c)
    c.export_frozen(tmp_path / "a")
    c.export_frozen(tmp_path / "b")
    for f in sorted((tmp_path / "a").iterdir()):
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes(), f.name


async def test_frozen_layer_used_after_local(make_client, tmp_path):
    rec = make_client(cache_dir=tmp_path / "c1")
    await rec.generate("shared")
    rec.export_frozen(tmp_path / "frozen")
    c = make_client(cache_dir=tmp_path / "c2", frozen_dirs=(tmp_path / "frozen",))
    r = await c.generate("shared")
    assert r.cached and c.mock.calls["generate"] == 0
    assert len(list((tmp_path / "frozen").glob("manifest-*.json"))) == 1


# --------------------------------------------------------- concurrency


async def test_concurrent_async_use(make_client):
    c = make_client(mock_latency_ms=2, max_concurrency=4)
    prompts = [f"prompt {i % 10}" for i in range(60)]
    res = await asyncio.gather(*(c.generate(p) for p in prompts))
    assert len(res) == 60
    again = await asyncio.gather(*(c.generate(f"prompt {i}") for i in range(10)))
    assert all(r.cached for r in again)


def _proc_writer(cache_dir: str, start: int) -> None:
    cfg = LLMConfig(cache_dir=Path(cache_dir))
    c = LLMClient(cfg)
    c.embed_sync([f"text {i}" for i in range(start, start + 40)], dim=16)
    c.generate_sync(f"proc {start}")
    c.close()


def test_parallel_processes_share_cache(tmp_path):
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_proc_writer, args=(str(tmp_path), s)) for s in (0, 20, 40)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert LocalCache(tmp_path).count() == 80 + 3


def test_sync_helpers(client):
    r = client.generate_sync("sync please")
    assert r.text and client.generate_sync("sync please").cached
    assert client.embed_sync(["a b"], dim=8).vectors.shape == (1, 8)
    assert len(client.rank_sync("a", ["a", "b"]).scores) == 2
