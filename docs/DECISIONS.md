# Decisions and deviations

Running log of decisions taken while executing [BUILD_SPEC.md](BUILD_SPEC.md). Newest sections at the bottom. Anything marked **pending owner** is still open.

## D1. Execution mode (2026-11-22)

The owner asked for the spec to be executed end to end without stopping at each phase boundary. OWNER DECISION items are therefore decided here, with the reasoning written down so they can be reversed. OWNER CHECKPOINT items that genuinely need a human (hand labels, deployment and spend) are prepared but **not** faked or skipped silently; see D5 and D6.

## D2. Providers (2026-11-22)

Verified against the owner's Google Cloud project on 2026-11-22 (region `europe-west4`, Application Default Credentials):

| Capability | Provider | Model | Status |
|---|---|---|---|
| Text generation | Vertex AI | `gemini-2.5-flash` | works |
| Embeddings | Vertex AI | `gemini-embedding-001` (3072-dim native) | works |
| Ranking | Vertex AI Ranking API | `semantic-ranker-default@latest` | **not available**: Discovery Engine API is not enabled in the project |

Because the Vertex Ranking API is unavailable, the owner supplied an OpenRouter key. OpenRouter is therefore a fourth provider (`openrouter`) next to `vertex`, `ollama` and `mock`, and is the default for ranking (`RANK_PROVIDER=openrouter`, `POST /api/v1/rerank`). It also serves as an independent second vendor for a cross-check LLM judge (D5).

Deviation from the spec's ground rule "the agent never stores keys": the owner pasted the key and asked for it to be used. It is stored only in the git-ignored `backend/.env` as `OPENROUTER_API_KEY`. It must never appear in the repo, in logs, in cache keys or in results files. `backend/env.example` documents the variable name without a value.

Provider selection is per capability:

| Variable | Default | Values |
|---|---|---|
| `LLM_PROVIDER` | `vertex` | `vertex`, `openrouter`, `ollama`, `mock` |
| `EMBED_PROVIDER` | value of `LLM_PROVIDER` (`ollama` → `mock`) | `vertex`, `openrouter`, `mock` |
| `RANK_PROVIDER` | `openrouter` | `openrouter`, `vertex`, `mock` |

With `LLM_PROVIDER=mock` every capability defaults to `mock`, so tests and CI never touch the network.

## D3. Data source: Option B, OpenStreetMap (2026-11-22)

- Option A (Outdooractive API) is rejected: no API key is in the repo or supplied, and Outdooractive content is proprietary, so route texts could not sit in a public repository anyway.
- Option B: OpenStreetMap route relations (ODbL) from a Geofabrik extract, enriched with SRTM-derived elevation. Reproducible without any credentials. Region: Bavaria (or a sub-region of it large enough for ≥ 3,000 routes).
- OSM route relations rarely carry prose. Descriptions are therefore **generated deterministically from OSM facts** (tags of the relation, geometry, elevation profile, named features along the route). Where OSM has a real `description` tag it is used and marked. This is stated in `docs/DATA.md`; generated text is labelled as generated.
- The legacy catalogue (111 Outdooractive routes from the course project) is kept as `backend/data/seed/routes.jsonl`. It is scattered over Germany, Switzerland and Italy, so it is not a coherent region and is not used for evaluation. **Pending owner:** those 111 route texts were already public in the committed `app.db`; whether Outdooractive's terms allow that is for the owner to check.

The database holds exactly one catalogue at a time (legacy seed or OSM catalogue), so ids never collide.

## D4. Findings about the baseline that the spec did not list (2026-11-22)

- The questionnaire maps "adventure types" to 13 tags (`culture`, `heritage`, `suitableforfamilies`, `loopTour`, …). Only 3 of them (`scenic`, `flora`, `fauna`) occur anywhere in the 19-tag vocabulary of the legacy catalogue (`Round trip`, `Family-friendly`, `Cultural/historical interest`, …). For the `history-culture` and `family-fun` answers the Jaccard tag score is therefore always 0: 30% of the `v0_rules` score is dead weight for those users. `v0_rules` is kept as is (it is the baseline); the evaluation makes the effect visible.
- Legacy difficulty values are 0–3 (20 / 35 / 46 / 10 routes). The scale is made explicit as 0–3.

## D5. Owner hand-label checkpoint (pending owner)

The 150 blind labels can only come from the owner. The harness produces the blind labelling sheet and a `make kappa` target. Until the owner's labels exist:

- every results table carries the note "judge not yet validated against human labels";
- as an interim signal only, a second LLM judge from a different vendor grades the same 150 pairs and LLM–LLM weighted kappa is reported. This is **not** a substitute for the human check and is labelled as such.

## D6. Deployment (pending owner)

Dockerfile, rate limit and budget cap are built. Nothing is deployed and no public URL is created without the owner's explicit approval of target and spend.

Update 2026-11-23: the Dockerfile, pre-generated narratives and a CI job that builds the image and smoke-tests it without credentials exist ([DEPLOY.md](DEPLOY.md)). Image defaults: no live providers, `NARRATIVES_LIVE=false`, rate limit on, `DAILY_BUDGET_USD=2`; seeded at build time, user data on a `/data` volume. Still **pending owner** before any deployment: target, URL, spend ceiling, which providers are live, the EU-region question for chat (D18), OSM attribution on every screen that shows route data, and the Harry Potter themed stories and souvenir graphics inherited from the course project, which are third-party intellectual property and should not go on a public demo as they are.

## D7. More baseline findings from Phase 0 (2026-11-22)

Found while writing the tests. None of these is changed in `v0_rules` (the spec allows only the two decay fixes there); the new pipeline and Phase 5 address them.

1. **Feedback is applied twice and compounds.** `POST /profiles/{id}/feedback` applies all of the profile's feedback to the user vector and saves the result back to `user_vector_json`; ranking then applies all feedback again on top of the saved vector. After *n* feedbacks the shift is roughly n(n+1)/2 + 1 steps, and the questionnaire's original vector is lost. The offline baseline (`rank_routes_v0`) applies feedback once to the original vector.
2. **A profile without a user vector gets HTTP 500**: the fallback path returns routes whose breakpoints are not loaded, and serialising them fails.
3. **"Random" recommendations are not random over the catalogue**: the query takes `limit * 3` rows before shuffling, so they always come from the lowest route ids.
4. **Feedback on a route outside the current category tab does not shift the user vector**, because route vectors are built only from that tab's candidates. The same profile gets a different adjusted vector on each tab.
5. **The penalty and the drop-at-4 rule ignore age**: only the vector shift decays; an old thumbs-down filters a route forever.
6. **The UI never sees the feedback fields**: the response schema drops `feedback_penalty`, `feedback_count`, `base_score` and `final_score`.
7. 29 of the 111 legacy routes have no tags (tag score is a flat 0.2), and many routes tie on the total score, so their order is just id order.
8. `calculate_distance_score` divides by the user's maximum distance, so a vector with maximum 0 crashes.

## D8. Catalogue region and format (2026-11-22)

- The region is the whole Free State of Bavaria, not only the Alps. Oberbayern + Schwaben alone give about 2,400 usable routes (below the 3,000 target) and lose many Allgäu routes at the Austrian and Baden-Württemberg borders. All of Bavaria is one Geofabrik extract and one administrative boundary and yields 9,347 routes after filtering and de-duplication. Details and all figures: [DATA.md](DATA.md).
- The catalogue is committed gzip-compressed (`backend/data/catalog/routes.jsonl.gz`, 4.8 MB instead of 31 MB), written with a fixed gzip mtime so the bytes depend only on the content.
- Running routes are too few (55) to evaluate; simulated users cover hiking and cycling only.
- Ascent is the weakest field (SRTM is a surface model): median absolute error about 17% against the 101 held-out routes that carry an OSM `ascent` tag, and flat lake circuits over-count.

## D9. Relevance judge: `gemini-3.5-flash`, thinking minimal, batches of 8 (2026-11-22)

Chosen by a pilot (`make judge-pilot`; details in [EVAL.md](EVAL.md) section 4, numbers in
`eval/results/_judge_pilot.json`), not by assumption. 80 stratified pairs from 20 dev personas,
reference `anthropic/claude-opus-5.5` with reasoning (another vendor, clearly stronger):

| Candidate | QWK vs reference | QWK between two batch orders | Nominal cost / pair |
|---|---|---|---|
| `gemini-2.5-flash`, thinking off, batch 8 (the spec's default) | 0.691 / 0.779 | 0.822 | 0.00026 USD |
| `gemini-2.5-flash`, thinking 1024, batch 8 | 0.796 | – | 0.00056 USD |
| `gemini-2.5-flash`, thinking off, batch 4 | 0.778 | – | 0.00031 USD |
| `gemini-3.5-flash` (`global`), thinking low, batch 8 | 0.828 / 0.812 | 0.915 | 0.00189 USD |
| `gemini-3.5-flash` (`global`), thinking minimal, batch 8 | 0.836 / 0.807 | 0.864 | 0.00116 USD |

- Agreement: the two `gemini-3.5-flash` configurations are best and indistinguishable on 80 pairs.
  `gemini-2.5-flash` without thinking is clearly worse and the least stable across batch orders.
- Cost: "minimal" is about 60% of "low". The full pool (5,749 LLM pairs) cost 7.26 USD with it; "low"
  measured about 0.0024 USD per pair live, about 14 USD. Every later system adds holes, so this
  compounds over Phase 3.
- Lifecycle: `gemini-2.5-flash` is past its documented retirement date (D2); `gemini-3.5-flash` is GA
  with retirement no earlier than 2027-05-19. It is served only on the `global` endpoint for this
  project, so judge prompts (simulated personas and OSM route facts, no personal data) leave the EU
  region. The intent writer and second judge go through OpenRouter anyway.
- Accepted cost of the choice: lower batch-order stability than "low" (0.864 vs 0.915). Revisit if
  the owner's labels show the judge is the weak link.

The relevance judge is Google, the intent writer is OpenAI (`openai/gpt-6-luna`), and the pilot
reference and interim second judge are Anthropic (`claude-opus-5.5`, `claude-sonnet-5.5`), so no
model grades text it wrote. The owner checkpoint (D5) is still open: 150 blind pairs are in
`eval/owner_labels/`. Until `make kappa` has real labels, every table says the judge is unvalidated.

## D10. Evaluation harness choices (2026-11-22)

- **Home towns** are the 407 OSM `place=city|town` nodes with a population inside Bavaria, weighted by
  population^0.5 (pure population weighting would put about a fifth of the personas in Munich).
- **Eligible set**: the travel radius widens (×1.25 steps, up to 150 km) until at least 40 routes pass
  the hard constraints and at least 10 are of the persona's own activity; 40 of 300 personas widened.
- **Running is not simulated** (55 routes, D8).
- **Hard-constraint violations are graded 0 by rule** without an LLM call (49% of pooled pairs, almost
  all from the two baselines, which do not see the home location).
- **Two pool-only contributors** (`pool_latent`, `pool_intent_lexical`) join the pool and never appear
  in result tables. `pool_latent` supplies most known-relevant pairs; this pooling bias is documented in
  EVAL.md section 4.
- **Coverage@10 has no CI**: it is a set statistic, and a with-replacement bootstrap biases it downwards.
- **Results determinism**: everything except a `volatile` block (local timings, run ledger, timestamp)
  is byte-identical on rerun; `make replay-check` proves a credential-free replay from the committed
  frozen cache `eval/cache/`.
- **Spend for Phase 2** (real, from the local spend log `python -m evalkit.llm`): 9.93 USD in total.
  Vertex 8.56 (`gemini-3.5-flash` 8.36: full pool 7.26 plus pilot; `gemini-2.5-flash` 0.20, pilot);
  OpenRouter 1.37 (`claude-sonnet-5.5` second judge 0.79, `claude-opus-5.5` pilot reference 0.53,
  `gpt-6-luna` intents 0.05). The full judging run was capped at 12 USD per process. Reruns cost 0.

## D11. Phase 3 pipeline shape and registry (2026-11-23)

One `Pipeline` class with a `PipelineConfig`; every ablation row is a registered config (`v1_filters` …
`v7_mmr`, `recsys_v1`), so each row is a complete system selected by name and fully described by the
`config` stored in its results file. `v2_bm25` and `v3_dense` rank by their own score (standalone
generators), so in the ablation chain "+ dense" *replaces* BM25 and "+ RRF" fuses both; this gives both
the chain and the standalone NDCG@10 the spec asks for. Design: [RECSYS.md](RECSYS.md).

## D12. Hard-filter policy (2026-11-23)

Missing slot → constraint not applied (and reported as not applied). Fewer than 10 eligible routes →
only the radius widens (×1.25 steps, at most ×2); tab and distance cap never relax; widened admissions
are marked; short lists are not padded. Unknown tab → empty list.

## D13. Retrieval choices (2026-11-23, all on dev)

BM25F with stemming, negation scopes and a query-side synonym list; the German glossary / decompounder
is implemented but **off** because it lowered dev recall (place-name morphemes). Dense: the structured
facts card, not title + description. RRF k = 30. Document matrix committed as float16, not int8
(int8 reorders too many near-ties). Figures: RESULTS.md "Candidate generation".

## D14. Ranker API model and whether the default uses it (2026-11-23)

The Vertex Ranking API stays unavailable (D2); ranking uses OpenRouter. Dev pilot over eight
rerankers → `mongodb/rerank-3` (reported id `rerank-3`). The learned ranker is trained with and without
the API score as a feature. **The default (`recsys_v1`) does not call the API**: the grouped-CV gain on
dev is small next to the API's per-request cost, its several-fold higher median latency and the extra
external dependency. `v7_mmr` (the same with the API) is registered and one config change away.
Disclosure: test rows of both variants existed when this was written (EVAL.md section 9).

## D15. Learned-ranker training data (2026-11-23)

Dev only. Labels: judged candidates of the `v4_rrf` top 50 plus a seeded random sample of 8 unjudged
candidates per dev persona, judged by the production judge (`"sample": "ltr-train/v1"`); unjudged
candidates are excluded, never negatives. Pinned in `eval/qrels/ltr_train_v1.jsonl`. Model selection by
5-fold CV grouped by persona. Serving evaluates the exported trees in numpy (no LightGBM at runtime).
The dominant feature (`length_cap_ratio`) encodes how the simulator draws distance caps; flagged as the
main overfitting risk (RECSYS.md section 4) — **pending owner**: whether to ship the "no suspicious
features" variant until real feedback exists.

## D16. MMR (2026-11-23)

Similarity = document-embedding cosine, min-max normalised over the candidate pairs (without it lambda
had almost no effect). Lambda 0.8 chosen on dev (smallest lambda whose dev NDCG@10 loss stays below
0.005). The diversity gain is small; MMR stays as a cheap hook for a better similarity.

## D17. Spend accounting after the outage (2026-11-23)

A network outage killed `python -m evalkit.ltr sample` mid-run, so that process never wrote its spend
row. Its judge batches were in the local cache; the real cost recorded on those cache entries was
appended to the local spend log as one "recovered" row, and the sample was rerun from cache. Total
spend: `python -m evalkit.llm`.

## D18. Chat slot extraction: Claude Haiku 5.5 via OpenRouter, reasoning off (2026-11-23)

Chosen on the scripted chat set (`backend/chat_eval/`, [CHAT.md](CHAT.md) section 8; 72 cases, 84 calls per candidate; numbers in `backend/chat_eval/results/summary.json`):

| Candidate | Slot P / R / F1 | Clarify accuracy | Schema violations | p50 / p95 latency | USD per turn |
|---|---|---|---|---|---|
| `gemini-2.5-flash`, europe-west4, thinking off | 0.984 / 0.977 / 0.980 | 1.000 | 0 | 0.85 / 1.39 s | 0.0013 |
| `gemini-3.5-flash`, global, thinking minimal | 0.985 / 1.000 / 0.992 | 0.972 | 0 | 1.50 / 3.02 s | 0.0077 |
| `gemini-3.5-flash-lite`, global, thinking minimal | 0.961 / 0.961 / 0.961 | 0.986 | 0 | 1.01 / 1.76 s | 0.0017 |
| `anthropic/claude-haiku-5.5` (OpenRouter), reasoning off | 0.970 / 1.000 / 0.985 | 1.000 | 0 | 1.12 / 2.08 s | 0.0007 |

- The LLMs differ by 1–4 errors on 72 cases; the set cannot separate them. It was written by the same agent that wrote the extractor, so all figures are optimistic.
- Haiku costs about a tenth of `gemini-3.5-flash` per turn and is faster; `gemini-2.5-flash` is past its retirement date (D2), so it is not a default even though it scores well; Flash-Lite was the weakest.
- Fallback when no OpenRouter key is configured: Vertex `gemini-3.5-flash` on `global`, thinking minimal. Overrides: `CHAT_LLM_PROVIDER`, `CHAT_LLM_MODEL`, `CHAT_VERTEX_LOCATION`, `CHAT_THINKING`, `CHAT_BUDGET_USD` (default 1 USD per process). `LLM_PROVIDER=mock` forces the offline keyword extractor.
- Accepted cost: chat messages (which may name a home town) leave the EU region with every candidate except the retired one, and OpenRouter is an additional processor. **Pending owner** if EU-only processing is required.
- No model grades its own output: the relevance judge stays Google (D9); chat extraction is Anthropic.
- Required slots are activity and place (the hard filters need them); one clarifying question each, at most two turns, then defaults (hiking; no radius filter).
- Spend for this comparison and the live checks: about 1.0 USD.

## D19. Which ranker to serve until real feedback exists: `recsys_v1_nocap` (recommendation, 2026-11-23)

Follow-up to D15 (kept as history). System-level rows, same pipeline as `recsys_v1` except the ranker,
retrained on the same pinned dev labels with the same CV (`make ltr-robust`), every top 10 judged
(RESULTS.md "Robustness of the learned ranker"):

| System | Ranker without | NDCG@10 test | Δ vs `recsys_v1`, test | Δ, dev (in-sample) | CV NDCG@10 (dev) |
|---|---|---|---|---|---|
| `recsys_v1` | – (all non-API features) | 0.760 | – | – | 0.801 |
| `recsys_v1_nocap` | `length_cap_ratio` | 0.736 | −0.024 [−0.042, −0.009] | −0.024 | 0.778 |
| `recsys_v1_nosusp` | slot ratios, home distance, intent km | 0.726 | −0.034 [−0.051, −0.018] | −0.051 | 0.768 |

**Recommendation: serve `recsys_v1_nocap` (and build the Phase 5 feedback systems on it) until real feedback
exists to retrain on; `recsys_v1` stays what it is and stays registered.** Reasons:

1. The 0.024 that `length_cap_ratio` buys is earned on the simulator that created it: persona caps are the
   top of the comfortable band times U(1.25, 1.7), and the judge reads the band, so "about two thirds of the
   cap" is a fact about the generator. The measured loss is therefore an upper bound on what real users
   would lose, not an estimate of it.
2. At serving time the feature can be missing or mean something else. Every training persona has a cap, so
   the trees never saw a missing ratio, and the cap is optional in the form and in the chat; a user who
   states a cap as a true maximum ("max 12 km") would have routes near it pushed down. Without the feature
   the ranker leans on `tag_cover`, `duration_h` and the intent's own km figure, which a real user controls.
3. `recsys_v1_nosusp` also drops home distance and the intent's km figure, which are real signals (people
   prefer nearer starts; a stated distance is a stated preference). It costs a further 0.010 on test and
   0.027 on dev for no robustness argument as strong as point 1.
4. Switching is one config value (`ltr_model="v1_noapi_nocap"`); nothing else changes.

Not checked here, and the first thing to measure with real traffic: behaviour with the cap slot empty.
~~Pending owner~~ **Decided by D25** (served default `recsys_serve` uses this ranker; the empty-cap check is in
D25).

## D20. Feedback design (2026-11-23)

Design: RECSYS.md section 10. Choices not tuned, with reasons:

- **One half-life, 60 days, for completions and rejections.** Different half-lives make the profile drift
  towards whichever sign forgets more slowly, and with a few outings a month a short half-life forgets before
  there is signal. 60 days is about a season. The simulation cannot tune it: its users never change their
  minds, so the longest memory would always win.
- **Route effects use the same decay**: a completed or rejected route stays hidden while its weight is
  ≥ 0.25 (two half-lives, about four months), then may return. Fixes D7 point 5 (an old thumbs-down hid a
  route forever) without re-showing last week's hike.
- **Applied once from the stored profile**, never written back (D7 point 1).
- **`too-far` = too far to travel** (the UI shows a car) and tightens the radius; v0 keeps reading it as route
  length. Effort and distance reasons adjust slots but neither the embedding nor the exploration posteriors;
  only `completed` and `not-interested` are taste signals.
- **Questionnaire-only users**: slots, tags and route effects apply as for everyone; Rocchio has no base
  query, so its feedback direction is used for re-scoring only and retrieval stays on the rule path.
- Functions for the API: `app.recsys.feedback.feedback_events_from_rows`, then `recsys_v1_fb` (RECSYS.md
  section 10).

## D21. Feedback parameters chosen on dev; test disagrees (2026-11-23)

Dev grid (40 seeded dev personas, 10 rounds; RESULTS.md "Dev tuning"), rule fixed before the run: highest
mean NDCG@10 over rounds 2-10. Route effects alone, slot steps, Rocchio only, re-scoring only and the light
full loop are within ±0.001 of each other; medium loses 0.011, strong 0.053. The rule picks `slots_only`
(route effects + slot steps 0.25, Rocchio and re-scoring off) and that is `recsys_v1_fb`.

On the test simulation the light full loop (`recsys_v1_fb_light`) beats it by +0.025 [+0.014, +0.036] over
rounds 2-10, while the dev choice equals route effects alone. The default is **not** switched on that
evidence (test is for reporting). Likely reason for the disagreement: the ranker is in-sample on dev, so dev
lists start near the ideal and leave feedback little to repair. Re-tune once a dev ranker that is not
in-sample exists (cross-fitted), or on real feedback; until then `recsys_v1_fb_light` is the candidate to try.

## D22. Exploration slot: built, measured, not on by default (2026-11-23)

Thompson sampling over 9 content clusters for rank 10 while a user has fewer than 10 feedback events. On the
test simulation it costs −0.011 [−0.015, −0.008] NDCG@10 over rounds 2-10 (−0.015 in round 1, shrinking as
users leave the cold-start window) and raises cumulative coverage after 10 rounds from 16.7% to 18.7% of the
catalogue; completions per persona 7.84 vs 7.94. The simulated user never discovers new tastes (its
preferences are fixed and fully reflected in the judge's grades), so the benefit exploration exists for
cannot appear here. Whether a 1-2 point NDCG cost is worth more coverage is a product call: ~~pending owner~~
**decided by D25** (exploration on in the served default); `recsys_v1_fb_explore` is registered and one config
change away.

## D23. Simulation protocol (2026-11-23)

- Simulation judgements go to `eval/qrels/sim_qrels.jsonl`, not the pooled qrels, so the single-request
  tables keep their top-10 pool. A pair can carry a grade in both files (graded in different batches);
  each table reads its own, and `make replay-check` checks each file against its own replay.
- The behaviour model, its probabilities and its limits are in EVAL.md section 10; it reads the judge's
  grades, so the simulated user agrees with the judge by construction.
- The per-round metric is computed after the last round against all judgements of the run, so a replay
  reproduces it exactly.
- Spend for Phase 5 and the rows below (local spend log): about 5.8 USD (simulation 3.7, dev tuning 0.8,
  robustness rows 0.35, end-to-end rows 1.0 including 300 chat calls).

## D24. Chat evaluation in the harness; end-to-end comparison; ledger fix (2026-11-23)

- The scripted chat evaluation moved from `backend/chat_eval/` (as D18 cites it) to `backend/evalkit/chat/`,
  `eval/chat/` and `eval/results/chat/` + `_chat_eval.json`, with `git mv`; `make chat-eval` replays it.
- End-to-end on test: questionnaire only 0.364, + raw intent 0.760, + chat 0.748 (chat − raw −0.012
  [−0.037, +0.009]). The chat path does not beat typing the same text into a box, and moves the radius
  centre when the message names a town (2.6% violations@10). The comparison cannot speak for real users:
  the intents were written from the latent profile the judge reads (RESULTS.md, EVAL.md section 11).
- The LLM ledger now attributes spend to the requested model (the ranker pilot's rows had been filed under
  the configured default reranker). Rows logged before the fix keep the old attribution.

## D25. The served default: `recsys_serve` (decided by the coordinator, 2026-11-23)

The API serves **`recsys_serve`** = the ranker trained without `length_cap_ratio` (D19) + the feedback loop with
the dev-chosen settings (`recsys_v1_fb`'s, D21) + the exploration slot on (D22). It is a composition of
registered parts; nothing new was trained. `recsys_v1` and every other system stay as they were. **This is a
judgement call made without real users**: every number below comes from simulated users, an LLM judge not yet
checked against the owner, and (for the feedback rows) a behaviour model written for this project.

What it costs and buys, from RESULTS.md "Served default" and "Empty profile slots":

- Single request (no feedback yet), test: NDCG@10 0.724 vs 0.759 for `recsys_v1`, −0.035 [−0.053, −0.020];
  of that, −0.024 is the ranker (D19) and −0.011 [−0.015, −0.006] the exploration pick at rank 10. Dev shows
  the same split.
- 10-round simulation, test: 0.681 over rounds 2-10 vs 0.315 for `recsys_v1` without feedback and 0.701 for
  `recsys_v1_fb` (full-feature ranker, no exploration). Exploration on the served ranker costs −0.011
  [−0.015, −0.008] (as on the full-feature ranker) and raises cumulative coverage after 10 rounds from 16.7%
  to 18.7%.
- Users who leave the cap empty: removing the cap costs the served system −0.114 NDCG@10 (routes beyond the
  persona's own cap reach the list: 17.8% violations@10), and `recsys_v1` −0.180 (20.2%). Without a cap the
  served system is ahead of `recsys_v1` by +0.031 [+0.001, +0.064]: the measured version of D19's point 2,
  and the reason the "cap" feature's simulator gain is not worth its fragility.
- Users who leave the home empty: there is no radius filter, and NDCG@10 falls to 0.134 (0.126 with the cap
  also empty), with 84-85% of the top 10 outside the persona's travel radius. Neither the ranker nor any
  later stage substitutes for the home; the UI should ask for it (the chat already treats the place as
  required, CHAT.md section 3).

Why these three parts anyway: the cap feature's gain is a property of how the persona generator draws caps,
and the empty-cap check shows the full-feature ranker degrading more when real users skip the field;
the dev-chosen feedback settings are the ones chosen by the pre-stated rule (`recsys_v1_fb_light` did better
on test, D21, but test is for reporting); exploration is part of the target architecture, its simulated cost
is about one NDCG point, and the benefit it exists for (finding tastes the form did not capture) cannot
appear with simulated users whose preferences are fixed and fully known to the judge.

Alternatives, all registered and one config change away: `recsys_v1` (best single-request number, keeps the
cap feature), `recsys_v1_fb` / `recsys_v1_fb_explore` (full-feature ranker with the loop),
`recsys_serve_noexplore` (exploration off), `recsys_v1_fb_light` (the light full loop), `recsys_v1_nosusp`
(the most conservative ranker). Spend for this follow-up (local spend log): about 0.8 USD.

