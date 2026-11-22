# LLM / embedding / ranking client

Every generate, embed and rank call in the project goes through one client in `backend/app/llm/`. It handles provider selection per capability, an on-disk cache, a per-run budget with a hard stop, a usage ledger, bounded concurrency, retries, and structured-output validation. Tests and CI use the `mock` provider and make no network calls.

Related: [BUILD_SPEC.md](BUILD_SPEC.md) §0 (cost control, no secrets) and §3, [DECISIONS.md](DECISIONS.md) D2 (providers).

## 1. Usage

```python
from app.llm import get_client, wrap_untrusted, UNTRUSTED_DATA_RULE

client = get_client()                      # process-wide, built from settings / backend/.env

res = await client.generate(
    "Grade this route for the request.\n" + wrap_untrusted(route_text, "route"),
    system="You are a strict grader. " + UNTRUSTED_DATA_RULE,
    schema=Grade,                          # pydantic model -> res.parsed is a validated Grade
    max_tokens=200, temperature=0.0, tag="judge",
)
emb = await client.embed(texts, task="document", tag="routes")   # emb.vectors: float32 (n, 768), L2-normalised
q   = await client.embed([query], task="query", tag="queries")
rk  = await client.rank(query, docs, top_n=10, tag="rerank")     # rk.scores aligned with docs; rk.order = top-10 indices

report = client.usage_report()             # plain dict; embed it in results files
client.export_frozen("eval/cache/frozen")  # every cache key this run touched
```

Scripts can use the sync helpers `client.generate_sync(...)`, `client.embed_sync(...)`, `client.rank_sync(...)` (they call `asyncio.run`, so not from inside an event loop). Tests build isolated clients with `LLMClient(LLMConfig(cache_dir=tmp, budget_usd=0.05, ...))` and can install one process-wide with `set_client(...)`.

### Public API

```python
class LLMClient:
    def __init__(self, config: LLMConfig, *, providers: dict[str, Provider] | None = None)

    async def generate(self, prompt: str, *, system: str | None = None,
                       examples: Sequence[tuple[str, str]] = (),       # few-shot (user, assistant) turns
                       model: str | None = None, max_tokens: int = 512, temperature: float = 0.0,
                       schema: type[BaseModel] | None = None,
                       thinking: int | str | None = None,              # None = LLM_THINKING_BUDGET (default 0 = off)
                       tag: str = "default", cache: bool | None = None,
                       retries: int | None = None, provider: str | None = None) -> GenerateResult

    async def embed(self, texts: Sequence[str], *, task: Literal["query", "document"] = "document",
                    model: str | None = None, dim: int | None = None, tag: str = "default",
                    cache: bool | None = None, retries: int | None = None,
                    provider: str | None = None) -> EmbedResult

    async def rank(self, query: str, documents: Sequence[str], *, model: str | None = None,
                   top_n: int | None = None, tag: str = "default", cache: bool | None = None,
                   retries: int | None = None, provider: str | None = None) -> RankResult

    def generate_sync(...); def embed_sync(...); def rank_sync(...)
    def usage_report(self) -> dict
    def touched_keys(self) -> list[str]; def reset_touched(self) -> None
    def export_frozen(self, dest_dir, keys: Iterable[str] | None = None) -> dict   # manifest + "missing"
    @property spent_usd: float
    @property mock: MockProvider          # register canned responses in tests

get_client() -> LLMClient; set_client(client | None); reset_client(); run_sync(coro)
wrap_untrusted(text, label="data") -> str; UNTRUSTED_DATA_RULE: str; render_llama3(system, examples, prompt) -> str
```

Results:

| Type | Fields |
|---|---|
| `GenerateResult` | `text`, `parsed`, `model` (as reported by the provider), `provider`, `usage` (`input_tokens`, `output_tokens`, `thinking_tokens`), `cost_usd` (real; 0 on a cache hit), `nominal_cost_usd`, `latency_ms`, `nominal_latency_ms`, `cached`, `cache_key`, `schema_attempts` |
| `EmbedResult` | `vectors` (float32, n×dim, L2-normalised, input order), `model`, `provider`, `task`, `dim`, `input_tokens`, `cost_usd`, `nominal_cost_usd`, `latency_ms`, `nominal_latency_ms`, `n_cached`, `cached` (all texts from cache), `cache_keys` |
| `RankResult` | `scores` (one per input document, input order), `order` (indices by descending score, cut to `top_n`), `top(documents)`, `model`, `provider`, `top_n`, cost/latency fields, `cached`, `cache_key` |

Errors (all subclass `LLMError`): `ProviderError` (`status_code`, `retryable`, `timeout`; messages are redacted), `BudgetExceeded`, `SchemaValidationError` (`attempts`, `last_text`), `CacheMiss` (cache-only mode), `ConfigError`.

## 2. Layout

```
backend/app/llm/
  __init__.py       public exports
  client.py         LLMClient, get_client/set_client, run_sync; cache, budget, retries, concurrency, schema loop
  config.py         LLMConfig (frozen dataclass); LLMConfig.from_settings()
  cache.py          key functions, LocalCache (SQLite), FrozenLayer, TieredCache, export_frozen()
  ledger.py         Stats per (capability, tag), models used
  pricing.py        Vertex price table (OpenRouter returns its own cost)
  prompts.py        wrap_untrusted, UNTRUSTED_DATA_RULE, render_llama3, JSON fallback helpers
  types.py          requests / results
  errors.py
  providers/        base.py, mock.py, vertex.py, openrouter.py, ollama.py
```

## 3. Providers

Selected per capability (D2):

| Variable | Default | Values |
|---|---|---|
| `LLM_PROVIDER` | `vertex` | `vertex`, `openrouter`, `ollama`, `mock` |
| `EMBED_PROVIDER` | `LLM_PROVIDER` (`ollama` → `mock`) | `vertex`, `openrouter`, `mock` |
| `RANK_PROVIDER` | `openrouter` (`mock` if `LLM_PROVIDER=mock`) | `openrouter`, `vertex`, `mock` |

- **vertex**: Google Gen AI SDK (`google-genai`) in Vertex mode with Application Default Credentials. Few-shot examples become alternating `user` / `model` contents and `system` becomes `system_instruction`. Structured output uses `response_mime_type="application/json"` plus `response_json_schema` (the Pydantic JSON schema); the client validates the result anyway. Thinking: an int goes to `thinking_budget` (default 0 = off for Gemini 2.5), a string goes to `thinking_level` (Gemini 3.x). Embeddings send `task_type` `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`, `output_dimensionality` and `auto_truncate=True`. Batches hold at most 250 texts and about 18k estimated tokens. Ranking calls the Discovery Engine `RankService.rank` on `projects/<p>/locations/global/rankingConfigs/default_ranking_config` with `top_n = len(records)`, ids = input positions and `ignore_record_details_in_response`, in chunks of at most 1,000 records. It can't be tested live yet (the API is not enabled in the project), so request building is covered with a fake. Cost comes from `pricing.py`. SDK clients are created lazily and per event loop, so a cached replay never needs credentials.
- **openrouter**: plain `httpx`. `POST /chat/completions`: messages are system + few-shot turns + user. Structured output uses `response_format: {type: json_schema, strict: false, schema}`. If a model rejects that (400/404/422), the client retries once with the schema as a prompt instruction and remembers the model. Either way the reply is validated. Thinking 0 → `reasoning: {enabled: false}`. `POST /embeddings` with `dimensions`. If more dimensions come back, the vector is truncated (Matryoshka) and renormalised. `POST /rerank` with `top_n = len(documents)`. OpenRouter always returns `usage.cost`, the real charge, and the client records it. The task type is **not** forwarded for embeddings: OpenRouter lists no supported parameters for `google/gemini-embedding-001`, so query and document vectors are identical there. The task still keys the cache.
- **ollama**: generation only, the original path. Sends the same `/api/generate` payload as before: the Llama 3.1 chat template rendered by `render_llama3`, `stream: false`, `options.temperature` / `options.num_predict`. If a schema is given, it is passed as `format`. Free, so it is never blocked by the budget.
- **mock**: deterministic and offline.
  - Generate returns `"[mock <hash>] <first prompt words>"`. With a schema it returns a schema-valid object built from the JSON schema.
  - Tests can register responders with `client.mock.register(response, tag=..., predicate=..., times=...)`. A response may be a str, dict, BaseModel or callable. `client.mock.fail(capability, times, status_code)` injects errors.
  - Embeddings are a signed feature-hashed bag of words (blake2b, `1 + log tf`), L2-normalised, so texts that share words are closer.
  - Rank is lexical overlap, `|q∩d|/|q|`, with a small Jaccard tie-breaker.
  - `mock_cost_usd_per_call` / `mock_latency_ms` let tests exercise the budget and latency.

### Narratives (`app/services/genai_service.py`)

The prompts are now plain `system` + few-shot `examples` + user turn (`build_welcome_prompt`, `build_post_run_prompt`). The provider applies the format. A test renders them with `render_llama3` and compares them with verbatim copies of the old f-strings (`backend/tests/llm/legacy_prompts.py`). They are identical except for two whitespace-only lines that were indentation leftovers in the old f-strings. Unchanged on purpose, for parity: the old Ollama call did not set `raw: true`, so Ollama applies its own template around an already templated prompt. That is still the case.

`call_llm()` maps errors to `HTTPException`: upstream HTTP status → same status, timeout → 504, budget exhausted → 503, anything else or an empty response → 500. `call_ollama(prompt, max_tokens, temperature)` keeps its signature as a thin wrapper. The duplicated response-handling block in the old `call_ollama` is gone. The unused `_create_fallback_pixel_svg` helper was removed.

## 4. Cache

**Key** = SHA-256 hex of a canonical JSON document (`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, UTF-8) with `"v": 1` (key version) plus:

| Capability | Fields |
|---|---|
| generate | `cap`, `provider`, `model`, `system`, `prompt`, `examples` (list of [user, assistant]), `temperature` (as float), `max_tokens`, `schema` (Pydantic `model_json_schema()` or null), `thinking` |
| embed (per text) | `cap`, `provider`, `model`, `task`, `dim`, `text` |
| rank | `cap`, `provider`, `model`, `query`, `documents` (ordered list) |

Nothing else goes into the key: no tag, credentials, project, location, timestamps or `top_n`. `top_n` only cuts `order`, because every document is always scored. Keys are therefore stable across processes and machines. A test pins a hard-coded key. Caveat: `schema` is the JSON schema Pydantic generates, so a Pydantic upgrade that changes schema output would change keys for structured calls.

**Local layer**: `<LLM_CACHE_DIR>/cache.sqlite3` (default `<repo>/.cache/llm/`). It uses SQLite in WAL mode with a 60 s busy timeout and one connection per client behind a lock. It is safe for concurrent async use and for a few parallel processes (tested with 3 processes). The connection opens lazily, so building a client touches no disk. Each entry stores the output plus the original call's `cost_usd` and `latency_ms` (and tokens). A replay therefore reports **nominal** cost and latency as if nothing had been cached, while real cost is 0.

**Embeddings are cached per text.** A batch with some known texts sends only the unknown ones (deduplicated) to the provider. A batch's cost, tokens and latency are split over its texts (cost and tokens by text length, latency evenly).

**Only valid outputs are stored.** Schema violations are never cached, and neither are empty generations.

**Per-call switch**: `cache=False` bypasses the cache completely (no read, no write). `LLM_CACHE_ENABLED=false` disables it globally. **Decision:** the default is *on* for every call, including temperature > 0, because evaluation replays need the exact recorded outputs (for example simulated users). The user-facing narratives in `genai_service` pass `cache=False`, so a user gets fresh text, and they retry a transient error only once so the API falls back quickly.

**Frozen layers** (`LLM_CACHE_FROZEN_DIRS`, comma-separated, relative to the repo root or absolute) are read-only and git-friendly. Lookup order: local, then each frozen layer in order. `client.export_frozen(dir, keys=None)` writes the keys this client touched (or a given set) from any layer as one shard:

- `entries-<digest>.jsonl.gz`: one `{"key", "capability", "payload"}` per line, sorted by key, gzip with mtime 0 and no file name. The payload holds text / scores / metadata plus cost and latency, and no secrets.
- `vectors-<digest>.npz`: per dimension, `d<dim>` (float32, n×dim) and `d<dim>__keys`. The zip is written with fixed timestamps.
- `manifest-<digest>.json`: format name and version, key version, counts, files.

`<digest>` = the first 12 hex digits of the SHA-256 of the sorted keys. Exporting the same set twice gives byte-identical files (tested). For a credential-free replay of a recorded run, set `LLM_CACHE_FROZEN_DIRS=<dir>` and `LLM_CACHE_ONLY=true`. A miss then raises `CacheMiss` instead of calling a provider, and providers are never even constructed. This is tested with a provider that fails if called.

Not done: in-flight de-duplication. Two *concurrent* identical calls that both miss will both call the provider. The evaluation harness should deduplicate inputs itself, as `embed()` already does within one call.

## 5. Budget and ledger

- `LLM_BUDGET_USD_PER_RUN` (default 5) is a per-client running total of **real** spend. Before each provider request, the client raises `BudgetExceeded` if spent ≥ budget. Cache hits cost nothing and keep working after the budget is gone. Free providers (ollama, zero-cost mock) are never blocked. Calls already in flight finish, so concurrency can overshoot by up to `LLM_MAX_CONCURRENCY` requests.
- In the API server the process-wide client lives as long as the process. After 5 USD of narratives the API therefore falls back to its template texts, which acts as a spend cap.
- `usage_report()` returns:
  - `budget_usd`, `real_usd`, `nominal_usd`, `budget_remaining_usd`;
  - `providers`: the configured provider, model id, location (Vertex) and dim per capability;
  - `models_used`: requested model ids and the exact ids the provider reported, e.g. `cohere.rerank-v3-5:0`;
  - `totals`: per capability;
  - `by_tag`: per tag and capability. Fields: `calls`, `items`, `cache_hits`, `provider_requests`, `retries`, `input_tokens`, `output_tokens`, `thinking_tokens`, `real_usd`, `nominal_usd`, `real_latency_ms`, `nominal_latency_ms`, `schema_violations`, `schema_failures`, `errors`, `budget_blocked`, `unpriced_requests`;
  - `cache` (enabled, cache_only, frozen layer count, keys touched).

  It contains no credentials, project id or paths.

## 6. Robustness

- Bounded concurrency: one `asyncio.Semaphore(LLM_MAX_CONCURRENCY)` (default 8) per event loop, shared by all capabilities.
- Retries: up to `LLM_MAX_RETRIES` (default 5) on 429, 408, 5xx, timeouts and transport errors. The backoff is exponential from 1 s, capped at 30 s, with "equal jitter" (d/2 + U(0, d/2)). Other 4xx errors are not retried. `retries=` overrides per call.
- Structured output: validate against the Pydantic schema (code fences and leading prose are tolerated). On a violation, re-ask up to `LLM_SCHEMA_RETRIES` (default 2) times, appending the validation errors to the prompt. Each violation is counted in the ledger, and `SchemaValidationError` is raised at the end.
- Error messages are redacted: API keys, `Bearer` tokens and the Google Cloud project id are removed. Google's 404 bodies contain the full project path.

## 7. Untrusted text

`wrap_untrusted(text, label)` produces:

```
<untrusted_data label="route" id="<10 hex of sha256(text)>">
...text...
</untrusted_data id="...">
```

Any `<untrusted_data` / `</untrusted_data` inside the text is neutralised (`&lt;`). The id is derived from the content, so the wrapper is deterministic and cache keys stay stable. Put `UNTRUSTED_DATA_RULE` in the system prompt of every call that uses it. Later phases must wrap all user and catalogue text this way.

## 8. Verified on 2026-11-22 (Vertex AI docs, pricing, live checks)

Sources (read 2026-11-22):

- Model versions and lifecycle: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
- Text embeddings: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings and https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
- Ranking API: https://docs.cloud.google.com/generative-ai-app-builder/docs/ranking
- Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing and https://cloud.google.com/generative-ai-app-builder/pricing (ranking)
- Locations: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations (the per-region table did not render; availability was checked live instead)
- OpenRouter usage accounting: https://openrouter.ai/docs/use-cases/usage-accounting

### Models

| Model | Lifecycle (docs) | europe-west4, live check | Note |
|---|---|---|---|
| `gemini-2.5-flash` | released 2025-06-17, **retirement date 2026-10-20** (past); replacements listed: `gemini-3.8-flash` or `gemini-3.5-flash` | **works** (model_version `gemini-2.5-flash`) | spec default; kept, see below |
| `gemini-3.5-flash` | GA 2026-05-19, retires ≥ 2027-05-19 | 404 in europe-west4; **works on `global`** | candidate successor |
| `gemini-3.8-flash` | GA 2026-09-02, retires ≥ 2027-09-02 | 404 in europe-west4; on `global` rejects `thinking_level=minimal` | candidate successor |
| `gemini-3.5-flash-lite` | GA 2026-07-21 | 404 in europe-west4 | |
| `gemini-embedding-001` | GA 2025-05-20, retires no sooner than 2028-05-20 | **works** | spec default |
| `gemini-embedding-2` | released 2026-04-22 (priced as "Preview" on the pricing page) | 404 in europe-west4 | |
| `semantic-ranker-default@latest` | alias of `semantic-ranker-default-005` (1024-token context, 25 languages) | not testable: Discovery Engine API not enabled | spec default |

**Decision:** keep `gemini-2.5-flash`. It is past its documented retirement date but still served, and no newer GA Flash model is served in `europe-west4` for this project. Switching means either `VERTEX_LLM_LOCATION=global` with `VERTEX_LLM_MODEL=gemini-3.5-flash` (data leaves the EU region; about 5× / 3.6× the input / output price), or waiting for regional availability. **Pending owner.** Expect `gemini-2.5-flash` to disappear without further notice, and record model ids per results file (the ledger does this).

### Limits

- `gemini-embedding-001`: up to 3072 dimensions (Matryoshka; `output_dimensionality` truncates, so the client renormalises), 2048 tokens per text (excess silently truncated with `auto_truncate`), 20,000 tokens per request. The REST reference says "each request can only include a single input text" (legacy `predict` endpoint). The Gen AI SDK path used here accepted **100 texts per request** live and rejected 251 with "The number of requests must be between 1 and 250". The client therefore batches at most 250 texts and about 18k estimated tokens per request.
- Task types: `RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`, `CODE_RETRIEVAL_QUERY`. Default `RETRIEVAL_QUERY`.
- Ranking API: up to 1000 records per request, `topN` optional, score 0–1, location `global` in all examples.

### Prices (USD, standard on-demand; `pricing.py`)

A model missing from `pricing.py` is recorded with cost 0 and counted in `unpriced_requests`, so the budget does not see it. Add a price before using a new Vertex model.


| Item | Price |
|---|---|
| `gemini-2.5-flash` | 0.30 / 1M input tokens, 2.50 / 1M output tokens (output includes thinking) |
| `gemini-2.5-flash-lite` | 0.10 / 0.40 |
| `gemini-3.5-flash` | global 1.50 / 9.00; non-global 1.65 / 9.90 |
| `gemini-3.5-flash-lite` | global 0.30 / 2.50; non-global 0.33 / 2.75 |
| `gemini-3.8-flash` | 0.75 / 3.75 introductory until 2026-12-31, then 1.50 / 7.50 (non-global +10 %); `pricing.py` uses the regular price, so the budget errs on the safe side |
| Gemini Embedding (`gemini-embedding-001`) | 0.00015 / 1,000 input tokens (= 0.15 / 1M); output free |
| Ranking API | 1.00 / 1,000 queries; one query = up to 100 records |
| OpenRouter | returned per call in `usage.cost`. Observed: `cohere/rerank-v3.5` 0.002 per search; `google/gemini-embedding-001` 0.15 / 1M tokens |

### Live smoke (2026-11-22, `backend/scripts/llm_smoke.py`)

The first run cost 0.0022 USD (Vertex generate and structured generate, Vertex embed, OpenRouter rerank). The second run was all cache hits with 0 real spend, and nominal cost and latency were reported unchanged. One `generate_post_run_summary` call through Vertex cost 0.00026 USD. The structured call returned a valid `RouteFacts` on the first attempt on both Vertex and OpenRouter. With `LLM_PROVIDER=openrouter`, generate (reasoning off, `json_schema` output) and embeddings worked live for 0.00018 USD. With `LLM_PROVIDER=ollama`, `generate_post_run_summary` worked against a local `llama3.1:8b` (336 prompt tokens, 81 output tokens, 9.7 s).

## 9. Operations

```bash
# live smoke (spend capped at 0.10 USD), run twice: the second run is all cache hits
backend/.venv/bin/python backend/scripts/llm_smoke.py [--narrative] [--report]
# offline
LLM_PROVIDER=mock backend/.venv/bin/python backend/scripts/llm_smoke.py
# tests
cd backend && LLM_PROVIDER=mock .venv/bin/python -m pytest tests/llm
```
