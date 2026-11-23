# Serving the recommender

How `GET /api/routes/recommendations` serves the multi-stage pipeline ([RECSYS.md](RECSYS.md)): which
system, how the profile becomes a query, what happens when something is missing or fails, start-up cost,
latency, the explanation payload, and the demo protections. Code: `backend/app/services/recsys_service.py`,
`backend/app/services/demo_guard.py`, `backend/app/api/v1/routes.py`; tests: `backend/tests/app/test_serving.py`.

The figures in sections 4 and 5 were measured by hand on 2026-11-23 on the owner's Mac (Apple silicon,
Python 3.12, uvicorn, one worker, the full 9,347-route catalogue, Vertex `gemini-embedding-001` in
`europe-west4`). They come from requests to a running server, not from `make eval`. Treat them as a snapshot
of one machine and one network, not as results.

## 1. System selection

`RECSYS_SYSTEM` (default `recsys_serve`, the served system chosen in DECISIONS.md D25: the ranker trained without
`length_cap_ratio`, the feedback stage and one exploration slot) names any registered pipeline (`recsys_v1`,
`v7_mmr`, `v4_rrf`, ...) or `v0_rules`. The measurements and the example payload below were taken with
`recsys_v1`, which runs the same stages up to the ranker; `recsys_serve` adds the feedback and exploration stages.
`v0_rules` runs the original DB path (`recommendation_service.get_recommended_routes`) unchanged, so its
golden test still holds. The stated-need variants of iteration 2 (`recsys_serve_needs`,
`recsys_serve_needs_promote`, `recsys_serve_needs_demote`; RECSYS.md section 13, DECISIONS.md D27) are servable
the same way; by D27 none is the default. An unknown name, or a pipeline that fails to load, is logged at start-up and
falls back to the baseline (`reason: pipeline_not_loaded`).

## 2. From the stored profile to a `UserQuery`

| UserQuery field | Source |
|---|---|
| `user_vector` | `demo_profiles.questionnaire_vector_json` (the questionnaire vector as created) with the chat's vector slots projected onto it by the chat's own rules (`app.chat.projection.project_vector`): difficulty band, distance band / duration / cap, wanted and avoided tags. **Feedback is not in it.** |
| `category` | the requested tab; else the chat's activity (mtb → cycling); else every category |
| `home`, `max_travel_km` | `home_lat/lon`, `travel_radius_km` (the radius only applies with a home) |
| `max_distance_km` | `distance_cap_km` |
| `intent` | `intent_text` (from onboarding or chat) |
| `feedback` | every `profile_feedback` row (`too-hard`, `too-easy`, `too-far`, `not-interested`) and every souvenir (`completed`) as `FeedbackEvent(route_id, reason, age_days)`, with real ages from `created_at` / `completed_at` |
| `needs` | needs the chat's slots state (`app.chat.projection.needs_from_slots`): activity `mtb`; wanted tags `Suitable for strollers`, `Accessibility`, `Train access`, `Public transport access`, `Round trip`, `Linear route`, `Multi-stage route`; a stated duration. Read only by a system with a stated-need stage; echoed as `served_by.query.slot_needs` |

**Original vector.** The baseline's feedback endpoint rewrites `user_vector_json` after every feedback, so
feedback compounds (DECISIONS.md D7.1). That behaviour is kept for `v0_rules`. The pipeline never reads
the rewritten vector. Migration `c4d8e2a6f1b9` adds `questionnaire_vector_json`. New profiles store the
questionnaire vector there at creation, and nothing rewrites it afterwards. Existing rows are backfilled
on a best-effort basis:

- Rows without feedback get the chat's pre-chat vector (`chat_state.base_vector`) if a chat happened,
  else the stored vector.
- Rows with feedback get difficulty range and distance band rebuilt from `fitness_level`. Tags stay as
  stored, because a `not-interested` may have removed some and they cannot be recovered. These rows carry
  `"_reconstructed": true`, and `served_by.query.vector_source` says `reconstructed`.
- Rows without any vector stay NULL. The service then uses `user_vector_json`, or `{}`.

`PATCH /api/profiles/{id}` with `user_vector_json` does not touch the questionnaire vector.

## 3. Paths and fallbacks

Every response carries `served_by`: `configured_system`, `system` (what actually ranked), `path`,
`reason`, `degraded`, `request_id`, the query summary, per-stage summaries, `latency_ms` and
`upstream_cost_usd`.

| Situation | Path / system | `reason` |
|---|---|---|
| Profile with intent, provider fine | `pipeline` / the configured system (default `recsys_serve`) | – |
| Profile without intent | `pipeline` / the configured system's rule path: filters → rule scorer → (feedback) → MMR (→ exploration slot) (no provider call) | `no_intent` |
| Profile without home | `pipeline`. The radius filter is not applied, and the verdict says `"applied": false` with `null` distances | – |
| Guest (no `profile_id`) | `random` over the tab, seeded per request (uniform over the whole tab, unlike the baseline's lowest-id bias, D7.3) | `guest` |
| Embedding/rank provider error | `no_llm` / `v2_bm25`: same filters, BM25F candidates ranked by BM25, no provider call | `provider_unavailable` |
| Pipeline exceeds `RECSYS_TIMEOUT_S` (2 s) | `no_llm` | `timeout` |
| Serving client's own budget (`RECSYS_BUDGET_USD`) spent | `no_llm` | `serving_budget` |
| Daily budget spent (section 6) | `no_llm` | `daily_budget` |
| Per-IP limit reached (section 6) | `no_llm` | `rate_limited` |
| DB route ids ≠ catalogue route ids | `baseline` / `v0_rules` (guests: the old DB random) | `catalogue_mismatch` |
| `RECSYS_SYSTEM=v0_rules` | `baseline` | `configured` |

- The no-LLM path is `v2_bm25`, a row in the Phase 3 ablation table (RESULTS.md), so its quality is
  known. MMR is left out on purpose: it reads the document matrix, and if the embedding provider
  changes, the matrix check fails too.
- The serving client has its own budget, 1 retry with sub-second backoff, and a timeout tied to
  `RECSYS_TIMEOUT_S`. The page never waits on the client's default 5 retries.
- **Catalogue/DB check.** On the first request per database URL (also at start-up), the service compares
  the set of route ids in `routes` with the in-memory catalogue. A mismatch (for example a DB seeded with
  the legacy catalogue) logs `recsys: DB routes do not match the in-memory catalogue (db=…, catalogue=…,
  missing in DB=…, not in catalogue=…); serving the baseline v0_rules. Reseed with make seed.` Each
  request also loads the recommended ids from the DB. If any id is missing, that request is served by the
  baseline and the check is redone, so the API never returns an id the DB does not have.
- `limit`: on the intent path the pipeline returns at most its candidate depth (50). The rule paths can
  return more.

## 4. Start-up and memory

Resources load once in the lifespan (`load_resources`, then an intent-free warm-up request) and are
shared by all requests. They are cached per process and configuration.

| Step | ms (3 cold starts) |
|---|---|
| catalogue (`routes.jsonl.gz` → records + numpy arrays) | 258 / 262 / 261 |
| BM25F index | 546 / 553 / 549 |
| learned ranker trees (`ltr_v1_noapi.json`, self-check) | 6 / 6 / 6 |
| document matrix (`doc_embeddings.npz`, float16 → float32) | 33 / 33 / 33 |
| warm-up request (tab memos, MMR) | 157 / 159 / 160 |
| **recsys total in the lifespan** | **1,001 / 1,013 / 1,009** |
| process start → `/healthz` answers | 1.38 / 1.51 / 1.40 s |

Memory: peak RSS 88 MB after imports → 385 MB after loading; resident set size 388 MB with the server
idle. The document matrix alone is 9,347 × 768 float32 = 28.7 MB. Most of the rest is the catalogue
records (with their raw `features` dicts) and the BM25 postings. With `LLM_PROVIDER=mock` the warm-up
also builds the mock document matrix (870 ms).

`GET /api/recsys/status` returns these figures, the DB check (by file name, never a path), the daily
budget state and the rate-limit configuration.

## 5. Request latency (HTTP, sequential, one client on localhost)

| Configuration | n | p50 ms | p95 ms | max ms |
|---|---|---|---|---|
| `recsys_v1`, live Vertex, **new intent each request** (embedding not cached) | 35 | 117 | 186 | 192 |
| … of which the query embedding call | 35 | 106 | 175 | 181 |
| … server-side total (`served_by.latency_ms`) | 35 | 115 | 184 | 190 |
| `recsys_v1`, live, same intent again (embedding from the on-disk cache) | 35 | 9.6 | 9.8 | 10.1 |
| `recsys_v1`, no intent (rule path, no provider call) | 20 | 6.8 | 7.4 | 80 (first) |
| `recsys_v1`, `LLM_PROVIDER=mock`, new intent | 35 | 8.8 | 9.3 | – |
| `v0_rules` (DB path, live chat) | 20 | 160 | 226 | – |

- **Dominant stage: the query embedding.** It takes about 90% of a live uncached request. Local compute
  is ~8 ms: filter ~1 ms, BM25 < 1 ms, learned ranker trees 5–17 ms over 50 candidates, MMR < 1 ms.
  The first live request after start-up took 720 ms, because the Vertex SDK client is created lazily on
  the first call. A second embedding call in the loop took 695 ms (the tail RECSYS.md warns about). The
  2 s deadline covers both.
- These figures sit inside RECSYS.md's no-API budget (p95 ≤ 500 ms) and agree with the offline warm benchmark
  (`make latency`, RESULTS.md "Latency and cost against the budget", which also covers `recsys_serve`).
- `v0_rules` is slower than the pipeline. It loads every route of the tab from SQLite and scores them in
  Python on each request.
- Cost per uncached request: one short intent embedding, about 1e-6 USD.

## 6. Demo protections

Both are configurable (`backend/env.example`) and in memory per process.

**Per-IP rate limit** (`RATE_LIMIT_ENABLED`, default **off**): a sliding window
(`RATE_LIMIT_WINDOW_S` = 60 s), keyed by the client address (`X-Forwarded-For` only with
`RATE_LIMIT_TRUST_FORWARDED=true`, behind a trusted proxy).

| Bucket | Default per window | Over the limit |
|---|---|---|
| chat turns (`POST /api/profiles/{id}/chat`) | 20 | HTTP 429 with `Retry-After` |
| recommendations that would call a live provider (intent present) | 60 | served by the no-LLM path (`reason: rate_limited`) |
| narratives (welcome text, post-route summary) | 10 | template text (the profile and the souvenir are still created) |

Requests without an intent, and story generation (template only), cost nothing and are not counted.

**Daily budget** (`DAILY_BUDGET_USD`, default 5 USD per UTC day, 0 = off): the real spend of every LLM
client in the process (narratives, chat, serving) since the first check of the day. Once spent,
recommendations take the no-LLM path (`reason: daily_budget`), chat uses the keyword extractor
(`model: "keyword-rules"`, same validator, merge rules and replies), and narratives use their templates.
Nothing fails. The per-client hard stops (`LLM_BUDGET_USD_PER_RUN`, `CHAT_BUDGET_USD`, `RECSYS_BUDGET_USD`)
stay in force underneath.

Limits of this design: the counters reset on restart and are per worker. A multi-worker or multi-instance
deployment needs a shared store, or the deployment's own limits.

## 7. Feedback

The pipeline gets every event in `UserQuery.feedback`, from the original vector, applied once. The served
default `recsys_serve` has its own feedback stage (`app/recsys/feedback.py`, D25). For a configured system
*without* one (`recsys_v1`, `v7_mmr`, ...), the serving layer adds an **interim rule** instead: a route the
user rejected or completed in the last 60 days is
left out of the list. Sixty days is two 30-day half-lives of the baseline's decay. Older events no longer
hide it, so an old thumbs-down does not filter a route forever (D7.5). The pipeline is asked for
`limit + hidden` routes, so the list stays full. On the no-intent path the rule scorer also applies its
own feedback adjustment, once, to the original vector.

When the configured system has a feedback stage (`PipelineConfig.feedback` set), the interim rule switches
off. The stage's route effects and re-scoring then appear in `explanation.stages.feedback`, and
`served_by.query.feedback_handled_by` says `pipeline`. Phase 5a (`recsys_v1_fb`, `recsys_v1_fb_explore`) is committed, and serving it is one setting away:
`RECSYS_SYSTEM=recsys_v1_fb_explore`. `tests/app/test_serving.py` checks that path: the interim rule switches off,
the pipeline drops the rejected route itself, and exactly one card per list is labelled "Exploration pick"
with `stages.explore` filled. Since D25 the default is `recsys_serve`, which has both stages, so the interim
rule only applies when a system without a feedback stage is configured.

## 8. Explanations

Each route on a pipeline path carries `explanation` (`RouteExplanationResponse`) and `reason`:

- `stages`: `app.recsys.explain.StageContributions` (`recsys.explanation/v1.1`: v1 plus the optional stage
  `needs`; nothing in v1 changed, so a v1 reader that ignores unknown stages keeps working). These are the trace's
  per-route contributions, typed. A stage that did not run is `null`.
- `stages.needs` (only with a stated-need system and a stated need): one check per need with the catalogue's
  numbers (`fact`, `values`), the rule (`threshold`), whether it was enforced (`applied`, `applied_level`) and met
  (`ok`, `met_level`). `n_eligible` then counts the routes past the hard filter *and* the needs.
- `feedback`: events considered, events about this route, rule penalty, routes hidden, the pipeline's
  feedback-stage contribution if any.
- `exploration`: `is_exploration` plus the exploration stage's contribution. It is always false until a
  configured system has an exploration slot. The UI then labels the card "Exploration pick".
- `n_candidates` (routes the ranker ordered) and `n_eligible` (routes past the filter).
- `reason`: one line built by a template from those fields and the catalogue record only (no LLM, no
  waiting). Stemmed BM25 terms are shown as the user's own word ("hik" → "hiking"). With a stated-need stage it
  names up to two met needs with their numbers ("you asked for stroller-friendly: no steps, 94% of the route on
  stroller-suitable surface, 120 m of climbing") and the first unmet one ("not reachable by train: no train
  station or stop mapped near the start (too few routes met it)").

The baseline path keeps `recommendation_score` / `recommendation_score_breakdown` and gets a template
`reason` from that breakdown.

Real example (live, `recsys_v1`; profile: home Rosenheim, radius 40 km, cap 15 km, intent "easy lake
loop with a hut"; LTR features cut to the top 5 of 38):

```json
{
 "route": {"id": 9379281, "title": "Valepper-Almen-Rundweg", "length_meters": 4268.0, "is_loop": true,
   "reason": "4.3 km loop starting 27 km from Rosenheim, inside your 40 km radius; within your 15 km limit; matches hut, lake, round; ranked 1st of 50 candidates."},
 "explanation": {
  "schema_version": "recsys.explanation/v1", "rank": 1, "score": 0.8,
  "stages": {
   "filter": {"verdict": "pass",
     "category": {"applied": true, "ok": true, "value": "Hiking trail", "allowed": ["Theme trail", "Hiking trail"]},
     "radius": {"applied": true, "ok": true, "distance_km": 27.49, "limit_km": 40.0, "limit_used_km": 40.0, "relaxed": false},
     "distance_cap": {"applied": true, "ok": true, "length_km": 4.27, "limit_km": 15.0}},
   "v0_rules": null,
   "bm25": {"score": 9.907455, "rank": 1, "matched_terms": [
     {"term": "hut", "contribution": 5.1324}, {"term": "lake", "contribution": 2.7543},
     {"term": "round", "contribution": 0.9543}, {"term": "circular", "contribution": 0.6426},
     {"term": "easy", "contribution": 0.4239}]},
   "dense": {"similarity": 0.745242, "rank": 1},
   "fusion": {"rrf_score": 0.064516, "rank": 1, "k": 30, "inputs": {"bm25": 1, "dense": 1}},
   "rank_api": null,
   "ltr": {"score": 1.779112, "rank": 1, "bias": -0.016426, "model": "v1_noapi", "features": [
     {"name": "tag_cover", "value": 1.0, "contribution": 0.73619},
     {"name": "rrf", "value": 0.064516, "contribution": 0.444418},
     {"name": "length_cap_ratio", "value": 0.284533, "contribution": -0.345757},
     {"name": "loop_match", "value": 1.0, "contribution": 0.343148},
     {"name": "fused_rank", "value": 1.0, "contribution": 0.281229}]},
   "mmr": {"lambda": 0.8, "relevance": 1.0, "max_similarity": 0.0, "penalty": 0.0, "mmr_score": 0.8,
     "rank_before": 1, "rank_after": 1}},
  "feedback": {"events_considered": 0, "on_this_route": [], "rule_penalty": null, "hidden_routes": 0,
    "window_days": 60.0, "stage": null},
  "exploration": {"is_exploration": false, "detail": null},
  "n_candidates": 50, "n_eligible": 621
 },
 "served_by": {"configured_system": "recsys_v1", "system": "recsys_v1", "path": "pipeline", "reason": null,
   "degraded": false, "latency_ms": 281.1, "upstream_cost_usd": 9e-07,
   "stages": [["filter", 9347, 621], ["bm25", 621, 100], ["dense", 621, 100, "256 ms upstream"],
              ["fusion", 200, 50], ["ltr", 50, 50], ["mmr", 50, 50]]}
}
```

(`served_by.stages` is abbreviated here. The API returns one object per stage with `n_in`, `n_out`,
`local_ms`, `upstream_latency_ms`, `upstream_cost_usd` and `info`.)

**How the UI renders it** (`frontend/lib/explanation.ts`, `components/recommendation-score-modal.tsx`).
The card shows the `reason` line and a "Why this?" button. The modal shows one plain-language line per
stage; "Show the numbers" expands the raw values under each line:

- *Fits your limits*: "It is a hiking trail, as the tab asks. 27 km from Rosenheim, inside your 40 km
  radius. 4.3 km long, within your 15 km limit."
- *Matches your words*: "Matches: hut, lake, round, circular, easy."
- *Close in meaning*: "The 1st closest in meaning to what you wrote, of 621 routes that fit your limits."
- *Both searches combined*: "1st when the word search and the meaning search are combined."
- *How it ranked*: "Ranked 1st of 50 candidates before diversity. Mostly because it has the features you
  asked for, and it scores high in the combined search. Held back a little because its length is far from what
  suits your limit." These are the top three Saabas contributions, phrased by sign.
- *Keeping the list varied*: "Stayed 1st." Otherwise "Moved up/down from 3rd to 1st …".
- *Your feedback*, when there is any: "1 route you rejected or completed recently is left out of this
  list."
- *What you asked for*, with a stated-need system: "You asked for stroller-friendly: no steps, 94% of the route on
  stroller-suitable surface, 120 m of climbing." A relaxed need says so ("too few routes nearby meet it, so it was
  relaxed"); "Show the numbers" lists the rule, the level met and enforced, and the raw values.

A degraded list gets a one-line note above the tabs, for example "routes are matched by your words only".
Baseline routes keep the old percentage badge and the CBF breakdown modal.
