# Deploying the demo

How to build and run the demo image, what it needs, what it does without credentials, what is
pre-generated, and what the owner has to decide before anything goes public. **Nothing has been deployed.**
Deployment target, public URL and spend need the owner's approval (BUILD_SPEC Phase 6, DECISIONS D6).

Files: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `backend/requirements-runtime.txt`,
`backend/scripts/container_start.py` (entry point), `backend/scripts/check_runtime_imports.py`,
`backend/scripts/smoke_container.py`, `backend/scripts/narratives.py`, `backend/data/narratives/`,
`backend/app/services/narratives.py`, `backend/app/services/provider_status.py`. CI: `.github/workflows/ci.yml`
(jobs `frontend`, `narratives`, `docker`).

## 1. Build and run locally

```bash
docker build -t trailsaga-demo .                                   # from the repo root
docker run --rm -p 8000:8000 -v trailsaga-data:/data trailsaga-demo
open http://localhost:8000                                         # frontend; API under /api, docs at /docs
python3 backend/scripts/smoke_container.py --expect-offline        # the CI smoke test, against the container
```

or `docker compose up --build`. The image serves the API and the static frontend from one uvicorn process
(one worker) on `$PORT` (8000). It runs as the non-root user `app` (uid 10001). A `HEALTHCHECK` polls
`/healthz` every 30 s.

### Image layout

| Stage | What it does |
|---|---|
| `frontend` | `node:22-bookworm-slim`, pnpm 10.23.0, `pnpm install --frozen-lockfile && pnpm build` → `out/` (static export; Google fonts are fetched and self-hosted at build time) |
| `pydeps` | `python:3.12-slim-bookworm`, a venv with `backend/requirements-runtime.txt` only (binary wheels only) |
| `app` | the backend code the server needs: `app/`, `alembic/`, `scripts/`, `pipeline/{__init__,tags}.py`, the catalogue's `routes.jsonl.gz` + `manifest.json` + `ATTRIBUTION.md`, `data/narratives/`, `data/historical_context/` |
| `seed` | migrates and seeds the **pristine database** from the committed catalogue (`alembic upgrade head` + `scripts/seed_from_catalog.py`), writes the catalogue's sha256 next to it, then runs `scripts/check_runtime_imports.py` |
| `runtime` | slim Python + venv + code + pristine DB + `out/`; no compilers, no node, no `.env`, no caches |

The build context is an allowlist (`.dockerignore` starts with `*`). `backend/.env`, the venv, the 1.3 GB of
raw OSM downloads, LLM caches, local `*.db` files, `node_modules`, `.next`, the evaluation harness and its
data never reach the Docker daemon. Credentials are never part of a layer. They are passed at run time
(section 3).

### Seeding at build time, user data on a volume

The catalogue (9,347 routes, geometries, breakpoints, places) is seeded **once at build time** into a
pristine SQLite file inside the image (47 MB). At start, `container_start.py` does this:

1. If `/data/app.db` is missing, it copies the pristine file there (part of the 0.77 s "database ready" step
   below). It does not seed at every first start.
2. If `/data/app.db` came from an image with a **different catalogue** (the `app.db.catalogue.sha256` marker
   differs), the route ids may no longer match. By default (`RESEED_ON_CATALOGUE_CHANGE=true`) the old file
   is kept as `app.db.<sha>.bak` and a fresh copy is used. With `false` it is kept, and the API reports
   `catalogue_mismatch` and serves the baseline (SERVING.md section 3).
3. It always runs `alembic upgrade head`. This is a no-op on a fresh copy and migrates a database from an older image.
4. It `exec`s uvicorn.

Why this design: start-up stays short (no 9k-route seed on every fresh container), and every container starts
from a database identical to CI's. User data (profiles, feedback, chat state, souvenirs) lives on a writable
volume (`/data`), so it survives restarts and image updates with the same catalogue. Without a volume it
lives in the container layer and is lost when the container is replaced. That is acceptable for a throw-away
demo. The cost is 47 MB of image size.

### Runtime dependency set

`backend/requirements-runtime.txt` (pinned to the versions the tests run on) has fastapi, uvicorn (+ uvloop,
httptools), pydantic(-settings), python-dotenv, SQLAlchemy, aiosqlite, alembic, greenlet, httpx, numpy and
google-genai. It leaves out:

- the data pipeline (`osmium`, `shapely`). The app imports only the pure-Python `pipeline/tags.py`;
- the evaluation harness's training-time libraries (`lightgbm`, `scikit-learn`, `matplotlib`). The ranker
  trees are evaluated in numpy (`app/recsys/ltr.py`);
- `google-cloud-discoveryengine` (grpc/protobuf). Only `RANK_PROVIDER=vertex` uses it (the Vertex Ranking API,
  not enabled in the project, D2), and the served `recsys_serve_v2` calls no ranking API;
- uvicorn's `watchfiles`, `websockets` and PyYAML (reload, websocket routes and YAML log config are unused);
- pytest, ruff, pre-commit.

**Proof.** `scripts/check_runtime_imports.py` parses every `.py` under `app/` and `alembic/` (plus the seeding
scripts) and collects every import, including the lazy ones inside functions (`from google import genai`,
`from scripts.osm_catalog import ...`, `from pipeline.tags import ...`). It imports each one for real, plus the
modules loaded by name (aiosqlite, greenlet, uvicorn, uvloop, httptools). The `seed` stage runs it inside the
image: 100 files, 108 modules imported, only `google.cloud.discoveryengine_v1` missing (expected, listed as
optional). The smoke test then exercises every endpoint the demo uses in the running container.

### Measured (2026-11-23, Docker Desktop on the owner's Mac: Apple silicon, linux/arm64, 10 CPUs, 8 GB VM)

| | |
|---|---|
| Image size | 495 MB unpacked (`docker images`), 114 MB compressed. Layers: venv 144 MB, pristine DB 47 MB, code + catalogue + narratives 21 MB, frontend 2.8 MB, the rest is `python:3.12-slim` |
| Build | first build 42 min, of which 41 min were `pnpm install` downloading 199 packages over a slow link inside the VM. Python deps 9 s, seed + import check 6 s, `next build` 4 s. Cached rebuilds after a backend change take seconds |
| First start (empty volume) → `/healthz` 200 | 3.40 s (3.37 s on an earlier run). Of this, database copy + migrate 0.77 s and recommender resources 1.16 s (catalogue 0.31 s, BM25 index 0.61 s, document matrix 0.05 s, warm-up 0.18 s) |
| Restart (existing volume) → `/healthz` 200 | 3.22 / 3.27 / 3.35 s |
| Memory | 331 MiB right after start (`docker stats`), 333 MiB after 31 full smoke runs. Peak RSS reported by the app 345 MB |
| Latency, no credentials (smoke test, one client) | recommendations with intent (no-LLM path) 13.5 ms, profile creation 21 ms, chat turn 7 ms, route completion 9 ms, static index 54 ms (first hit) |
| Latency, live embeddings (ADC mounted) | first intent request 1.2 s (Vertex client created lazily), then 9.6 ms for a cached intent. SERVING.md section 5 has the full figures |

Plan for about 512 MB of memory per instance and one CPU. These are figures for one machine, not a benchmark.

## 2. Configuration

The image's defaults are the demo configuration. Every value can be overridden with `docker run -e`.
`backend/env.example` documents all variables.

### What the image sets

| Variable | Image value | Code default | Why |
|---|---|---|---|
| `LLM_PROVIDER` | `vertex` | `vertex` | So the served pipeline loads the committed Vertex document matrix. Without `GOOGLE_CLOUD_PROJECT` no call is attempted (section 3) |
| `NARRATIVES_LIVE` | `false` | `true` | Narratives come from the pre-generated store (no spend, grounded texts) even when credentials are given |
| `RATE_LIMIT_ENABLED` | `true` | `false` | Per-IP limits on everything that can spend |
| `DAILY_BUDGET_USD` | `2` | `5` | Cap on real spend per UTC day over all LLM clients of the process |
| `LLM_BUDGET_USD_PER_RUN` / `CHAT_BUDGET_USD` / `RECSYS_BUDGET_USD` | `1` / `1` / `0.5` | `5` / `1` / `1` | Per-client hard stops per process, under the daily cap |
| `DATABASE_URL` | `sqlite+aiosqlite:////data/app.db` | `backend/data/app.db` | The writable volume |
| `LLM_CACHE_DIR` | `/data/llm-cache` | `<repo>/.cache/llm` | Writable. Query embeddings are cached here |
| `FRONTEND_DIST_DIR` | `/app/frontend/out` | `frontend/out` | |
| `LOG_ENABLE_FILE` / `LOG_DETAILED_FORMAT` | `false` / `false` | `true` / `true` | Logs go to stdout only |
| `PORT` | `8000` | – | uvicorn port |

### Variables that matter for the demo, and what each degrades to when unset

| Variable | Default (image) | Unset or no credentials means |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | unset | No Vertex calls. Recommendations with an intent are served by the no-LLM path (`v2_bm25`: same hard filters, BM25F ranking; `served_by.path = "no_llm"`, `reason = "provider_unavailable"`, `degraded = true`, quality in RESULTS.md's `v2_bm25` row). Without an intent the configured pipeline's rule path serves as usual (no provider call). The chat has no Vertex fallback model |
| `GOOGLE_APPLICATION_CREDENTIALS` | unset | Needed with the project for Vertex inside a container (section 3). Project set but ADC missing: the call fails, and the request falls back as above. The chat uses the keyword rules for that turn |
| `OPENROUTER_API_KEY` | unset | The chat does not use Claude Haiku (D18). With neither key nor Vertex project, chat turns are answered by the **keyword extractor** (`model: "keyword-rules"`; slot F1 0.885 vs 0.985 for Haiku on the scripted set, CHAT.md) |
| `EMBED_PROVIDER` | = `LLM_PROVIDER` | Set `vertex` for live query embeddings. Do **not** set `mock` in a demo: the pipeline would then rank with hashed bag-of-words vectors instead of the evaluated embeddings |
| `RECSYS_SYSTEM` | `recsys_serve_v2` (D28) | Any registered pipeline (`recsys_serve`, `recsys_v1`, …), or `v0_rules` |
| `NARRATIVES_LIVE` | `false` | Pre-generated text, else template (section 4) |
| `NARRATIVES_DIR` | `backend/data/narratives` | Missing or unreadable store: templates |
| `RATE_LIMIT_*`, `DAILY_BUDGET_USD` | see above | Section 5 |
| `RATE_LIMIT_TRUST_FORWARDED` | `false` | Behind a reverse proxy every user would share the proxy's IP bucket. Set `true` **only** if the proxy overwrites `X-Forwarded-For` |
| `RESEED_ON_CATALOGUE_CHANGE` | `true` | Section 1 |
| `LOG_LEVEL` | `INFO` | |

**Without any credentials** (the CI smoke test and the measurements above), the container does this:

- serves explained recommendations: per-stage explanation, reason line, filter verdicts;
- answers the chat with the keyword rules;
- gives pre-generated welcome and route texts (template where none exists);
- creates profiles, feedback, souvenirs and achievements as usual;
- makes no outbound network calls.

`provider_status.py` checks locally whether a provider has what it needs (project + location, or key), so no
call is attempted and no stack trace is logged.

## 3. Credentials at run time (optional: live query embeddings)

Only environment variables and a mounted file. Nothing is baked into the image, and nothing here
implements secrets handling beyond that.

```bash
docker run --rm -p 8000:8000 -v trailsaga-data:/data \
  -e GOOGLE_CLOUD_PROJECT=<project> -e GOOGLE_CLOUD_LOCATION=europe-west4 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp.json \
  -v /path/to/service-account.json:/run/secrets/gcp.json:ro \
  trailsaga-demo
```

- **ADC in a container.** `gcloud auth application-default login` writes a user credential on the host that the
  container cannot see. Mount a credentials file and point `GOOGLE_APPLICATION_CREDENTIALS` at it. For anything
  public, use a **service account** with only `roles/aiplatform.user`, not the owner's user credentials. On a
  Google-managed runtime the attached service account replaces the file. Checked locally on 2026-11-23 with
  the owner's ADC file mounted read-only: `recsys_v1` served on the `pipeline` path at 4.5e-7 to 1.05e-6 USD
  per uncached intent.
- **OpenRouter** (chat with Claude Haiku): `-e OPENROUTER_API_KEY=...` (better: the platform's secret store,
  exposed as an env var).
- Narratives stay pre-generated unless `-e NARRATIVES_LIVE=true`.
- **Logs.** `app/main.py` sets the `httpx`, `httpx2` (google-genai's client) and `httpcore` loggers to WARNING. At
  INFO they print every request URL, and Vertex URLs contain the project id. This was found while testing the
  container with live credentials and fixed. The check found the project id 0 times in the container logs
  afterwards.
- Do not commit a key file next to `docker-compose.yml`. `.gitignore` does not cover `*.json` key files.

## 4. Pre-generated narratives

### Which user-visible texts come from an LLM

Established from the code and by calling the endpoints (mock provider: the texts show up as `[mock ...]`).

| Text | Where | Source today |
|---|---|---|
| Welcome summary after the questionnaire | `POST /api/profiles` → `welcome_summary` (stored as `genai_welcome_summary`) | **LLM** (`genai_service.generate_welcome_summary`), template fallback `generate_fallback_welcome` |
| Summary after completing a route | `POST /api/profiles/{id}/souvenirs` → `souvenir.genai_summary` | **LLM** (`genai_service.generate_post_run_summary`), one-sentence template fallback |
| Chat slot extraction | `POST /api/profiles/{id}/chat` | **LLM** (structured output, not prose), keyword-rules fallback. The assistant's replies are templates |
| Route story, chapters, mini quests, quiz | `POST /api/routes/{id}/story` etc. | template (`story_generator.py`, no LLM) |
| Souvenir picture | souvenir `pixel_image_svg` | template (`svg_templates.py`, no LLM) |
| Recommendation reason line, "Why this?" modal | `GET /api/routes/recommendations` | template from the explanation's facts (`recsys_service.reason_sentence`, `frontend/lib/explanation.ts`) |
| Route description | catalogue | generated deterministically from OSM facts by the data pipeline (DATA.md) |

### What is pre-generated

| Store | Scope | Count | Model | Prompt version |
|---|---|---|---|---|
| `backend/data/narratives/welcome.json` | every questionnaire combination: 3 fitness × 8 adventure-type sets (empty set included) × 3 styles | 72 | Vertex `gemini-2.5-flash`, europe-west4, thinking off, temperature 0.7 | `welcome/pregen-v1` |
| `backend/data/narratives/routes.json` | the **demo route set** (`demo_routes.json`): routes in the top 10 of the served pipeline for a grid of demo profiles | 400 | same, temperature 0.4 | `route-complete/pregen-v1` |

**The demo route set** is computed by `narratives.py select` with no provider call. The grid is:

- 24 questionnaire vectors × 3 tabs × {no home, München, Garmisch-Partenkirchen, Tegernsee, Rosenheim, Nürnberg,
  Regensburg at 50 km}, without an intent, through `recsys_v1`'s rule path;
- plus 8 demo intents (the frontend's placeholders and similar) × 6 towns × 3 fitness levels, through the no-LLM
  path, which is what a credential-free container serves.

That is 648 queries, which put 1,173 distinct routes in a top 10. The 400 most frequent cover 78.7% of all
top-10 appearances. The other 8,947 routes get the template.

The set was selected on 2026-11-23 with `recsys_v1`, before D25 made `recsys_serve` the default
(`demo_routes.json` records `"system": "recsys_v1"`). `recsys_serve`'s rule path adds an exploration pick at rank
10, so a re-selection (`python scripts/narratives.py select`, no provider call) changes part of the set; routes
outside the stored set get the template text. Re-run `select` and `make narratives-live` before a deployment
(owner checklist, section 6).

D28 made `recsys_serve_v2` the default. Its stated-need stage reads needs from the intent and the chat slots;
the selection grid's main part has no intent and the intent part runs through the no-LLM path, so re-selecting
with `recsys_serve_v2` gives exactly the same 400 routes as with `recsys_serve` (checked on 2026-11-23 with
`RECSYS_SYSTEM=recsys_serve_v2 python scripts/narratives.py select`; the file was not committed). The narratives
were therefore not regenerated for D28; the difference to the stored `recsys_v1` selection (21 of 400 routes)
is the one described above.

**Provenance.** Each file has `schema`, `llm_generated: true`, a notice and `provenance` (provider, model,
prompt version, system-prompt hash, parameters, generator). Each item has `text`, `llm_generated`, `model`
(as reported by the provider), `prompt_version`, `generated_at`, `cache_key`, `temperature`, `attempt` and the
results of its checks. `routes.json` also records the catalogue's sha256. The API says where each text came
from: `welcome_summary_source` / `summary_source` = `llm` | `pregenerated` | `template`. The frontend does not
label pre-generated text as AI-written yet (owner checklist).

**Grounding.** Route texts get only a facts block built from the catalogue record
(`narratives.route_facts`: name, activity, shape, length, climbing, highest point, time, difficulty, start
locality, region, tags, and the OSM-derived description), wrapped as untrusted data. The prompt forbids
anything not stated there, as well as quests, XP and levels. The user's own numbers are added at request time
by a template sentence ("You completed 2 quests and earned 140 XP."). Every text must pass
`check_route_text`:

- every number must occur in the facts (rounding allowed);
- every capitalised word must occur in the facts or be a common word;
- no game terms unless the facts contain them;
- 20–100 words.

Welcome texts may not contain numbers or the legacy wizard theme, and must be 40–120 words. A failing text
gets one retry at temperature +0.1. If that fails too, the item is left out and the template serves.
Outcome:

- welcome: 70 of 72 passed at the first attempt (2 used "magic"), both retries passed;
- routes: 400 of 400 passed.

**Spot check (2026-11-23).**

- 12 random route texts were read against their catalogue records: no invented name or number. Every named
  lake, summit, castle, gorge and restaurant appears in the record. There are mild, unverifiable embellishments:
  "you journeyed through alpine terrain" (a 1,979 m summit route), "enjoyed the viewpoint", "this gentle
  journey".
- A scan of all 400 for weather, season, wildlife and history words absent from the facts (sun, snow, autumn,
  deer, ancient, medieval, breathtaking, ...) found none.
- One inherited data issue: "Priener-Hütte Runde" ("round") is a one-way route in OSM. The text follows the
  facts ("one-way").
- Welcome texts are generic by design: they name no places, and some say "Bavarian".

**Regenerate.** `backend/data/narratives/cache/` is a frozen cache layer holding every call used, so
`generate` replays byte for byte without credentials or network. CI's `narratives` job checks this with only
the runtime dependencies installed.

```bash
cd backend
python scripts/narratives.py select                 # demo route set (deterministic, no provider call)
python scripts/narratives.py generate               # replay from data/narratives/cache (no credentials)
python scripts/narratives.py generate --live --budget 0.5   # new prompt / route set: provider answers cache misses
python scripts/narratives.py check                  # store vs catalogue, provenance, grounding checks
```

After a catalogue rebuild, run `select` and `generate --live`. The `routes.json` items for routes no longer in
the catalogue fail the tests (`tests/app/test_narratives.py`).

Spend for this task: **0.145 USD** in total. That is 0.0042 for a 16-text pilot, 0.1406 for the full 472-text
run, 0.0005 for the two retries, and about 3e-6 for the live-embedding check in the container.

## 5. Demo protections and their limits

From `backend/app/services/demo_guard.py` (SERVING.md section 6). In the image both are on.

| | Limit (image) | Over the limit |
|---|---|---|
| Chat turns per IP | 20 per 60 s | HTTP 429 with `Retry-After` (seen in the smoke loop: turns 21+ within a minute got 429) |
| Recommendations with a live provider call per IP | 60 per 60 s | no-LLM path, `reason: rate_limited` |
| LLM narratives per IP | 10 per 60 s | pre-generated or template (with `NARRATIVES_LIVE=false` none are counted) |
| Real spend per UTC day, all clients | 2 USD | no-LLM paths everywhere (`daily_budget`), keyword chat, offline narratives |

Limits of this design:

- **The counters are in memory and per process.** A restart resets the day's spend and the windows. Two
  workers or two instances each have their own limits and budgets. The image runs one worker for that
  reason. Scaling out needs a shared store (Redis, the database) or the platform's own limits (max
  instances = 1, request quotas).
- The per-client hard stops (`LLM_BUDGET_USD_PER_RUN`, `CHAT_BUDGET_USD`, `RECSYS_BUDGET_USD`) are also
  per process. A crash loop resets them too, so the **provider-side budget alerts or quotas** are the real
  ceiling.
- The IP is the key. Users behind one NAT share a bucket, and IPv6 users can rotate addresses. Behind a proxy
  see `RATE_LIMIT_TRUST_FORWARDED` above.
- Nothing limits non-spending endpoints (profile creation, feedback, completions). A flood can still fill
  the SQLite volume. The platform's request limits or a WAF are the protection there.

## 6. Owner decisions before any deployment

- [ ] **Target and URL.** Which host or platform, which domain, TLS. The image needs one container with about
  512 MB RAM, a persistent volume for `/data` (or accept losing user data), and a single instance
  (in-memory counters, SQLite).
- [ ] **Spend ceiling.** Daily cap (`DAILY_BUDGET_USD`, image: 2) and per-client caps. Also a budget alert or
  quota at Google Cloud / OpenRouter, because the in-app counters reset on restart.
- [ ] **Which providers are live.** The image default is none: no-LLM recommendations, keyword chat,
  pre-generated narratives. Options:
  - live query embeddings only (Vertex, about 1e-6 USD per new intent);
  - plus the chat model (OpenRouter Haiku, about 0.0007 USD per turn);
  - plus live narratives.

  If credentials are used, it should be a dedicated service account, not the owner's ADC.
- [ ] **EU-region question for the chat** (D18). Chat messages can contain a home town. With OpenRouter or
  Vertex `global` they leave the EU, and OpenRouter is an extra processor. Choose keyword-only chat in public,
  an EU-only model, or accept it and add a privacy notice.
- [ ] **OSM content and attribution.** Route texts and geometry are ODbL. The API returns the attribution and
  the route detail view shows "© OpenStreetMap contributors, ODbL 1.0" (`route-facts.tsx`). Check that it is
  visible wherever routes or maps appear, and add an ODbL notice and a link to `data/catalog/ATTRIBUTION.md`
  on the page. Pre-generated narratives are derived from that data and are labelled LLM-generated in the data,
  but not in the UI yet.
- [ ] **Theme and branding.** The UI title, the story templates, the SVG souvenirs and the live welcome prompt
  use "Hogwarts" and Harry Potter references from the course project. For a public URL that is a
  trademark/copyright question.
- [ ] **Third-party scripts.** `frontend/app/layout.tsx` includes Vercel Analytics. Off Vercel it only requests
  `/_vercel/insights/script.js` (404). On Vercel it collects analytics, which needs a privacy notice or
  removal.
- [ ] **Legacy catalogue.** Not in the image (the OSM catalogue is). D3's Outdooractive question only matters if
  `CATALOG=legacy` is ever deployed.
- [ ] **Data retention.** Profiles, chat text (may contain a town) and feedback stay on the volume
  indefinitely. Decide on a retention or reset policy.
- [ ] **Narrative store.** The pre-generated route texts were selected with `recsys_v1` (section 4). Re-run
  `python scripts/narratives.py select` and `make narratives-live` (live call for the new routes only) so the
  store matches the served `recsys_serve_v2` (whose selection equals `recsys_serve`'s, section 4).

## 7. One neutral example (NOT executed)

This is any single-VM container host behind a TLS-terminating reverse proxy that overwrites
`X-Forwarded-For`. It is an illustration only. Nothing like it has been run, and it needs the decisions above
first.

```bash
# NOT EXECUTED: example only, after owner approval of target, URL and spend
docker build -t trailsaga-demo:2026-11-23 .
docker run -d --name trailsaga --restart unless-stopped \
  -p 127.0.0.1:8000:8000 -v /srv/trailsaga:/data \
  -e RATE_LIMIT_TRUST_FORWARDED=true \
  --memory 768m --cpus 1 \
  trailsaga-demo:2026-11-23
# the reverse proxy forwards https://<approved-domain>/ to 127.0.0.1:8000
```

## 7. Render free tier (public demo, no spend)

`render.yaml` at the repo root is a Blueprint: one Docker web service on the **free** plan, built from the
`recsys-upgrade` branch, no credentials. In the Render dashboard: *New → Blueprint → pick the repo →
Apply*. The first build runs the Dockerfile (several minutes).

What the free plan means for this demo:

- **512 MB RAM.** The app measures about 345 MB peak (section 1), so it fits, with little headroom.
- **Sleeps after 15 min idle**; the next visit waits for a cold start (about a minute with the image pull and the 3 s app start).
- **No persistent disk.** User data lives in the container and is lost on restart or redeploy. Fine for a demo.
- **Spend is zero** because no provider key is set. Setting keys in the dashboard turns on live LLM calls, which are
  limited by the rate limits and `DAILY_BUDGET_USD` (section 5). That is the owner's decision (D6).
- Vercel is not a fit: the backend is a long-lived process with SQLite and a 345 MB in-memory index, not a serverless function.
