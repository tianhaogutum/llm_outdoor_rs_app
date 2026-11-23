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


## D26. Phase 6 write-up: corrections and the open items in one place (2026-11-23)

Written while turning the README into the report, adding [FAILURES.md](FAILURES.md) and [REVIEW.md](REVIEW.md), and
checking the docs against each other and the code. Earlier entries are left as written; corrections are listed here.

- **Served system in docs and comments.** Several places still said `recsys_v1` is what the API serves (code comments
  in `systems/pipelines.py` and the recommendations endpoint, RECSYS.md, SERVING.md's paths table, DEPLOY.md). Since
  D25 it is `recsys_serve`; fixed. `recsys_v1` is the Phase 3 default and the reference row of most comparisons. Its
  registry description string still starts with "DEFAULT" because it is stored in the committed results files and
  changing it would break the byte-identical replay.
- **D15's "pending owner"** (ship the "no suspicious features" ranker?) was settled by D19 and D25: the served ranker
  drops only `length_cap_ratio`.
- **RECSYS.md section 12** said the served pipeline's dense retrieval is moved by Rocchio; with the dev-chosen
  feedback settings (D21) Rocchio and re-scoring are off, so it is not.
- **EVAL.md** quoted Phase 2 pool and frozen-cache counts as if current; they are now marked as Phase 2 snapshots and
  point to the generated figures.
- **Demo narratives** were selected with `recsys_v1` before D25 (DEPLOY.md section 4); re-selecting for
  `recsys_serve` changes part of the set. Added to the deployment checklist rather than regenerated (a live call,
  and the deployment itself is pending).
- **Latency of the served default** was not measured before; `make latency` now includes `recsys_serve` (one live run,
  under a tenth of a cent).
- **Generated numbers in the README and REVIEW.md.** `make report` now also rewrites marked blocks in README.md,
  docs/FAILURES.md and docs/REVIEW.md; `tests/evalkit/test_docs_fresh.py` fails if any block or RESULTS.md is stale,
  or if README / REVIEW.md prose outside the blocks contains a metric value. The other docs (EVAL.md, D-entries here,
  SERVING.md's hand measurements) still contain typed figures from the time they were written; they are labelled as
  snapshots or point to RESULTS.md.
- **What the failure analysis changes:** nothing in the served configuration. It shows the learned ranker overriding
  good retrieval for stated minority needs (MTB skill, strollers, small children) and the judge penalising wishes the
  request never stated; the fixes are new slots and features plus an evaluation change (FAILURES.md "Synthesis"),
  not a different registered system.

**Still open, all in one place** (REVIEW.md "What is still open for you" has the steps):

| Item | Owner of the decision | Where |
|---|---|---|
| 150 blind relevance labels, then `make kappa`, `make eval-all`, `make report` | owner | D5, EVAL.md section 4 |
| Deployment: target, URL, spend, live providers, attribution, third-party theme, analytics, retention, narrative re-selection | owner | D6, DEPLOY.md section 6 |
| Outdooractive texts in the public history of `app.db` | owner | D3 |
| EU-only processing for chat messages | owner | D18 |
| Confirm or change the served default chosen by the coordinator | owner | D25 |
| Generation model `gemini-2.5-flash` past its retirement date | owner | LLM_CLIENT.md |
| Re-tune feedback parameters with a cross-fitted dev ranker or real feedback | technical | D21 |

## D27. Iteration 2: stated needs as constraints; acceptance rule fixed before any dev result (2026-11-23)

**Where the hypothesis comes from (disclosure).** From the *test* failure analysis ([FAILURES.md](FAILURES.md)):
half of the ten worst test queries of `recsys_serve` state a verifiable need (stroller, train, a technical MTB
ride, a loop) that the learned ranker outweighs, two more state a length or a day count the pipeline does not
read. That observation is the only thing test contributes to this iteration. Which needs, the phrases, the
thresholds, the relaxation order, the floor and the way of using the needs are developed on dev and on general
language knowledge; test is run once, for the variant the rule below picks, and reported whatever it shows.

**Does the failure pattern replicate on dev?** Same analysis on dev (`python -m evalkit.failures --split dev
-n 20`, `eval/results/_failures_recsys_serve_dev.json`), classes assigned by hand with the definitions of
FAILURES.md "Synthesis":

| Dev worst | A: stated need overridden | B: length / days not read | C: unstated wish graded | D: no ranking failure |
|---|---|---|---|---|
| 10 | 0 | 3 (P162, P278, P265) | 1 (P175) | 6 |
| 20 | 1 (P026) | 7 (+ P094, P252, P214, P007) | 1 | 11 |

Class B replicates (four of the twenty state a day count, one says "short"); class A barely does. The reason is
known and matters for everything below: the learned ranker was trained on the dev personas' grades, so on dev it
has memorised exactly the users whose needs it overrides on test. A catalogue-only check that does not depend
on the ranker's fit shows the mechanism is there on dev too: among dev personas who state the need, the share of
`recsys_serve`'s top 10 that meets it strictly is low for the minority needs (MTB, train, multi-day, stroller;
`eval/results/_needs_dev.json`, `served_top10_meets_needs`).

**The acceptance rule, written down before any dev NDCG of a variant was computed** (also in
`evalkit/needs_eval.py: RULE`, stored in `_needs_dev.json`):

1. On dev (200 personas, every top 10 judged), the paired-bootstrap 95% CI of mean NDCG@10(variant) −
   NDCG@10(`recsys_serve`) has its lower bound above 0;
2. mean Violations@10 on dev is not above `recsys_serve`'s;
3. the stage's own compute adds less than 20 ms at p95.

Among accepted variants the one with the larger mean dev difference is served (under a new name; `recsys_serve`
stays as it is); if none is accepted the stage stays off by default and this is recorded as a null result.
Variants compared: the filter in front of retrieval (`recsys_serve_needs`), the filter plus promotion of strict
matches when it had to relax (`recsys_serve_needs_promote`), the post-ranker demotion (`recsys_serve_needs_demote`).
Extractor gate, also fixed in advance: a need with a latent counterpart is enabled only if its dev precision
against the latent profile is at least 0.8 (all are 1.0 on dev; `no_steps` and `public_transport` never fire on
dev intents, so they rest on unit tests and general language only).

**What dev can and cannot say here.** Two biases in opposite directions: the baseline's ranker is in-sample on
dev (optimistic for `recsys_serve`, and it hides the class-A failures the stage targets), while the stage's
thresholds were set by reading dev intents and judgements (optimistic for the stage). Neither is removable
without a second development split or a cross-fitted ranker; the test run is the check.

**Dev result and the decision (written before any test run of a variant).** Every top 10 judged on dev
(`_needs_dev.json`, `dev`). Paired difference of mean NDCG@10 against `recsys_serve`, dev, 200 personas:
filter −0.013 [−0.024, −0.003], filter + promote −0.025 [−0.040, −0.009], demote −0.031 [−0.044, −0.020];
no rise in violations (all 0); the stage's own compute is under 1 ms at p95. **No variant meets rule 1, so by the
pre-stated rule the change is not accepted: the stage stays off by default and `recsys_serve` remains the served
system.** Per need, on dev the stage helps where a day count or a one-way route is stated and costs most where
MTB or the train is stated.

**A post-hoc check, explicitly not used for the decision.** The in-sample bias named above was then measured
instead of only named: the served ranker retrained five times on four fifths of the dev personas (the folds of
its own grouped CV, same labels, parameters and rounds), each dev persona ranked by the copy that never saw its
labels (`python -m evalkit.needs_eval crossfit`; holes judged, dev only). With that ranker the baseline's dev
NDCG@10 falls to about its test level, and all three variants are *ahead* of it, with CIs above zero (filter
+0.018 [+0.006, +0.029], filter + promote +0.022 [+0.007, +0.036], demote +0.011 [+0.001, +0.022]). So the rule's
comparison was dominated by the ranker having memorised the dev users: the sign of the dev result flips when
that is removed. This is a reason to pre-register the next iteration's rule on a cross-fitted dev comparison; it
is not a licence to change this iteration's decision after seeing it, and the decision above stands.

**What test is used for.** All three variants are registered and evaluated on both splits for the tables
(reporting only; the decision is above and does not change with them). The failure analysis on test is rerun
for `recsys_serve_needs_promote`, chosen now, before test, as the variant the cross-fitted check favours.

**Test, run once after the decision above** (RESULTS.md "Iteration 2"; 100 personas, every top 10 judged; the
numbers below are a snapshot, the generated tables are authoritative). Paired NDCG@10 difference against
`recsys_serve`: filter +0.024 [+0.007, +0.046], filter + promote +0.025 [+0.004, +0.049], demote +0.026 [+0.009,
+0.047]; violations stay at zero; the stage adds well under a millisecond, so end-to-end latency is the served
system's. Test agrees with the post-hoc cross-fitted dev check and disagrees with the pre-registered dev
comparison. Of the ten worst test queries of `recsys_serve`, the promote variant repairs the class-A stroller,
MTB and train queries and the four-day tour, leaves the two queries without a verifiable need unchanged, and makes
seven other test queries worse by more than 0.1, mostly where the catalogue's check is stricter than the judge's
reading of the same need (FAILURES.md "Iteration 2").

**Decision, unchanged by test:** the served default stays `recsys_serve`; `RECSYS_SYSTEM` default, `env.example`
and the API are not switched; the stage, the three registered variants and their rows stay. This is the second
time a dev rule built on system-level dev rows of the learned ranker has disagreed with test (D21 was the first).
For the owner: serving a variant now would be a judgement call against the pre-registered rule (one setting,
`RECSYS_SYSTEM=recsys_serve_needs_promote` or `_demote`); the clean route is a next iteration whose rule is stated
on a cross-fitted dev comparison before anything is run, with the train check and the promotion of dropped needs
reconsidered on dev. Spend for iteration 2 (local spend log): about 0.8 USD (dev holes 0.4, cross-fitted check 0.2,
test holes 0.25).

## D28. Iteration 3: cross-fitted dev as the dev standard, and a fresh holdout decides (pre-registered 2026-11-23)

**Why.** D27's acceptance rule used in-sample dev NDCG. The learned ranker was trained on the dev personas, so for
any change that overrides it (filter, demotion, feedback) the dev comparison favours the unchanged system; the
cross-fitted post-hoc check and the single test run both pointed the other way. Test has now been looked at for
these variants, so it cannot decide between them cleanly. Iteration 3 fixes the dev protocol and decides on users
nobody has seen.

**1. Dev protocol (done before this entry; commit `359fe5d`).** Cross-fitted dev evaluation is a harness
feature for every system with a learned ranker (`make eval SYSTEM=<name> CROSSFIT=1`, EVAL.md section 13): five
copies of the system's ranker, each trained exactly like it (same pinned labels, features, LightGBM parameters and
rounds) without one fold of dev personas (its own grouped-CV folds); each dev persona ranked by the copy that never
saw it; holes judged (dev). It reproduces D27's post-hoc check exactly (same per-persona NDCG@10 for all 200 dev
personas and the four systems). It now exists for all 16 stored systems with a learned ranker (cost about 0.2 USD
of dev judging) and is reported next to in-sample dev (RESULTS.md "Dev with a cross-fitted learned ranker"). From
now on a dev number used for a decision about a learned-ranker system is the cross-fitted one.

**2. Pre-registration. Written and committed before any holdout persona, intent or judgement exists.** The code
that applies it is `backend/evalkit/holdout.py` (`RULE`, `decide`; commit `cd1faac`, tested on synthetic
data in `tests/evalkit/test_holdout.py`); this text is the commitment, the code implements it. Nothing below may be
changed after the holdout exists; a bug fix to the code would be disclosed here with its effect.

*Candidates* (as registered at `b0d3490`, unchanged; no new tuning, no new variant): the baseline `recsys_serve`
(pipeline config sha256 prefix `96e7a72620f8`) and the three iteration-2 variants `recsys_serve_needs`
(`52ad30ffcc06`), `recsys_serve_needs_promote` (`d2bab40ac2be`), `recsys_serve_needs_demote` (`44e8ab8d04e6`); all
use the committed ranker `v1_noapi_nocap` and the committed document embeddings.

*Primary metric.* Mean NDCG@10 over the holdout personas, every top 10 judged; paired per persona against
`recsys_serve`.

*Step 1, gate on cross-fitted dev.* A variant is tested on the holdout only if the paired-bootstrap 95% CI (1,000
resamples, seed 20261122, the harness standard) of its cross-fitted dev NDCG@10 minus `recsys_serve`'s has its lower
bound above 0. Disclosure: these numbers exist already (dev may be looked at): `recsys_serve_needs` +0.018 [+0.006,
+0.029], `_promote` +0.021 [+0.007, +0.036], `_demote` +0.011 [+0.001, +0.021]. All three pass, so m = 3 variants
go to the holdout. The gate is a filter, not the test.

*Step 2, holdout test with a correction for three variants.* For the gated variants jointly: one-sided simultaneous
95% lower bounds of the mean paired holdout difference NDCG@10(variant) − NDCG@10(`recsys_serve`), single-step max-T
over a joint paired bootstrap of the holdout personas (10,000 resamples, seed 20261122; `metrics.
simultaneous_lower_bounds`): each resample draws one set of persona indices shared by all variants; T = the largest
studentised deviation over the variants; c = its 95% quantile; bound = mean − c × bootstrap SE. The family-wise
error of "this variant beats `recsys_serve`" is 5% over the three together. Why this and not Bonferroni: the three
variants share most of their lists (two are the same filter), so their differences are strongly correlated and
Bonferroni would be needlessly conservative; max-T uses the measured correlation and is never looser than
Bonferroni when the variants are positively correlated. Why one-sided: the action is one-sided (a variant is served
only if better; "worse" and "no difference" both mean keeping `recsys_serve`). The ordinary two-sided 95% paired CIs
are reported next to it and decide nothing.

*Step 3, eligibility and choice.* A variant is eligible if its simultaneous lower bound is above 0 **and** its mean
holdout Violations@10 is not above `recsys_serve`'s. Among eligible variants the one with the largest simultaneous
lower bound wins (ties: filter, promote, demote, in that order). Precondition: every top 10 of the four systems on
the holdout is 100% judged; otherwise no decision is taken.

*If a variant wins:* it is registered as **`recsys_serve_v2`** with exactly the winning variant's config;
`recsys_serve` stays registered and unchanged; the API default (`backend/app/settings.py`, `backend/env.example`,
the serving tests, SERVING.md, DEPLOY.md) switches to `recsys_serve_v2`. The narrative demo route set is re-selected
only if the new default materially changes it (stated either way).

*Null result:* no eligible variant. `recsys_serve` stays the served default; the three variants stay registered and
off; this is recorded as a null result. It means a gain could not be shown at this size, not that there is none:
on test the paired per-persona differences had an SD of about 0.10-0.11, so at 150 personas the SE is about 0.008-0.009
and, with c near 2, a true gain of 0.02 would be detected only about half to two thirds of the time. No further variant, threshold or rule
is tried on this holdout afterwards.

*Holdout.* 150 personas, generator `personas/v1` with exactly the main parameters (`evalkit.personas.PARAMS`), new
seed **20261128** (the latent and questionnaire stream; seed + 2 the intent style stream), every persona in split
`holdout`, ids `H001`... = draw index. Duplicate check: a draw whose latent profile or whose system-visible profile
slots (questionnaire vector, tab, home, travel radius, distance cap) equal those of any main persona or an earlier
holdout draw is skipped (30 spare draws; skips stay visible as id gaps); identical intent texts are counted and
reported, not removed. Intents by the same writer as before (`openai/gpt-6-luna` via OpenRouter, prompt
`intent/v2`, temperature 0.9, cached). Its own qrels file `eval/qrels/holdout_qrels.jsonl`; pool = top 10 of the four
candidates plus the two pool-only contributors (`pool_latent`, `pool_intent_lexical`); judge `g35f-min-b8`, rubric
v1, hard-constraint violations graded 0 by rule; no other system is pooled on the holdout before the decision is
written. Cost is estimated before judging (`--estimate ... --holdout`); if judging cannot be completed inside the
iteration's 8 USD allowance, no decision is taken and that is reported.

*Afterwards.* The holdout is used only for this decision and then as a reporting split. The worst-queries failure
analysis of the system served after this decision moves to the holdout, because test has been used for development
(D27).

**3. Outcome (written after the holdout was judged; nothing above was changed).** Commit order: dev protocol
`359fe5d`, rule as code `cd1faac`, this pre-registration `bd8f1d9`, holdout personas `171d5a9`, holdout judged
and decided `4b7c129`.

*Holdout.* 150 personas, seed 20261128, no draw skipped as a duplicate; one intent text ("A quiet mountain bike
ride to a summit", H117) equals a dev persona's (P102) and was kept as registered. Mix: hiking 108, touring 29,
MTB 13; parties solo 32, couple 29, friends 21, older 19, school-age families 16, young families 13, with a dog 11,
stroller 8, parent with teenager 1; 26 train travellers. 3,684 pairs judged (holdout qrels sha256 prefix
`69db8f93f8de`), every top 10 of the four candidates judged; about 4.8 USD.

*Rule applied* (`_holdout.json`; numbers are a snapshot, RESULTS.md "Iteration 3" is generated from the files):
all three variants pass the cross-fitted dev gate, so m = 3 and the max-T critical value is c = 2.00. Holdout
NDCG@10: `recsys_serve` 0.771 [0.744, 0.796]; paired differences against it, 95% CI and simultaneous one-sided
lower bound: filter `recsys_serve_needs` +0.021 [+0.010, +0.034], bound **+0.009**; filter + promote +0.019
[+0.003, +0.034], bound +0.003; demote +0.006 [−0.007, +0.020], bound −0.008. Violations@10 are 0 for all four.
Eligible: filter and promote; the larger bound is the filter's.

**Decision: `recsys_serve_needs` wins. It is registered as `recsys_serve_v2` (exactly its config) and is the API
default** (`backend/app/settings.py`, `backend/env.example`, serving tests, SERVING.md, DEPLOY.md);
`recsys_serve` stays registered and unchanged. `recsys_serve_v2` returns the same lists as `recsys_serve_needs` on
all 300 main, 200 cross-fitted dev and 150 holdout personas. The narrative demo route set does not change (the
selection grid carries no stated needs; re-selection with `recsys_serve_v2` gives the same 400 routes as with
`recsys_serve`; DEPLOY.md section 4), so the narratives were not regenerated.

*What it does not show.* The gain is modest and its corrected lower bound is close to zero; Recall@50 of the
filter is lower than `recsys_serve`'s on the holdout (it removes routes from the candidates); everything rests on
simulated users and a judge not yet validated against the owner. The feedback simulation was not rerun with the
stage. The latency benchmark of `recsys_serve_v2` (one live run) is within the 500 ms p95 budget; its maximum
includes one embedding call that timed out and was retried.

*Holdout failure analysis* (`make failures`, now on the holdout for the served system; FAILURES.md): of the ten
worst holdout queries, 6 are class C (the judge grades wishes the request never stated), 3 a new class E (the
stage's own choices: MTB relaxed before train for a cyclist, a stroller check stricter than the judge, a stated
loop and train enforced where the judge would trade them), 1 class A (an effort limit the stage does not read). It
also found an evaluation bug: the judge's card prints the distance cap rounded to whole kilometres
(`evalkit/cards.py`), while the rule-grader uses the exact cap. Fixing it changes the judge prompt and therefore
needs re-judging; it is listed below, not done here.

*Spend for iteration 3* (local spend log): about 5.0 USD (cross-fitted dev holes 0.2, holdout intents under 0.02,
holdout judging 4.8, latency under 0.01).

**Still open after D28** (in addition to D26's owner items, of which "confirm or change the served default" now
refers to `recsys_serve_v2`): the stage's class-E fixes and an effort need, developed on cross-fitted dev; the
judge card's cap rounding (re-judge); re-tuning the feedback parameters on cross-fitted dev (D21); rerunning the
feedback simulation with the stage. *(Card rounding and the simulation: done in D29.)*

## D29. The judge card's rounding: fixed, affected pairs re-judged; the served default in the simulation (2026-11-23)

**The bug.** The judge's persona card (`evalkit/cards.py`, now called `card/v1`) printed the travel radius, the
comfortable distance band and the distance cap with `:.0f`, and the route card printed the distance from home the
same way, while the rule-grader and the hard filter use the exact values. A holdout persona with a 6.4 km cap read
"absolutely no more than 6 km", and the judge graded 6.0-6.4 km routes 0 for "exceeding" it (FAILURES.md, H149).
Checked line by line: ascent (whole metres against a tolerance in tens), days (integers), route length (0.1 km
against a cap stored with one decimal) and every other number on the cards were printed exactly enough. 411 of the
450 personas' caps have a decimal; every radius is a whole number, so the radius rounding never mattered.

**The fix.** `card/v2` prints radius, band and cap as stored (one decimal at most) and the distance from home to
0.1 km. A route the rules admit can then never be printed above a hard limit (unit test over random limits); for
the soft band, a printed position can differ from the exact one only for a route within 0.05 km of an edge (66 of
the re-judged pairs are such routes; immaterial for a rubric that asks for "inside or very close to" the band). New judgements carry `"card": "card/v2"`; lines without
the key are v1. `card/v1` stays in the code, unchanged, because the persona intents, the judge pilot, the second
judge on the owner set and every v1 judgement replay from prompts built with it. The owner's labelling page was
re-rendered with v2 cards (same 150 pair ids; nothing had been labelled); the interim second judge stays as recorded
(v1).

**Which pairs were affected** (`evalkit/rejudge.py`, rule `affected`): an LLM-graded v1 pair whose printed length is
above the printed cap although it is within the exact cap (`cap`), or whose position relative to the comfortable
band (below / inside / above) differs between printed and exact numbers (`band`), or whose printed distance from
home is above the printed radius although it is within it (`radius`: none). Rule-graded pairs never reached the
judge.

| Qrels file | LLM pairs | Affected: cap | Affected: band | Re-judged | Changed | Up / down |
|---|---|---|---|---|---|---|
| pooled (`qrels.jsonl`) | 14,828 | 39 | 424 | 463 | 138 | 81 / 57 |
| simulation (`sim_qrels.jsonl`) | 2,709 | 1 | 89 | 90 | 24 | 10 / 14 |
| holdout (`holdout_qrels.jsonl`) | 3,684 | 10 | 129 | 139 | 42 | 26 / 16 |
| **total** | 21,221 | 50 | 642 | **692** | **204** | 117 / 87 |

Each file was re-judged separately with `card/v2` (production judge, batched per persona over its affected pairs,
seed 0). The new line replaces the old one and keeps it under `previous` (old grade, reason and batch), so both
grades are in the file and `make replay-check` replays both; `eval/results/_rejudge.json` has the old-to-new
matrices per file and reason.

**Which changes the card caused.** Re-judging in new, smaller batches changes grades by itself (D9: batch-order
stability). A control of 200 *unaffected* pairs from the same personas, re-judged the same way
(`rejudge_control.jsonl`, grades not used), changed 42 (21%), 20 up and 22 down. Against that:

- *Cap*: 32 of 50 changed (64%), **all upwards**, mean grade 0.34 → 1.20. The bug, as described.
- *Band*: 172 of 642 changed (27%), 85 up and 87 down: hardly above the control's rate. Changes agree with the
  direction the exact band predicts somewhat more often than not (107 vs 65), so the band rounding had a small
  effect, but most of these 172 are the judge's batch noise, now recorded with the corrected card.

**Consequences** (everything rescored: `make eval-all`, `make simulate`, `make sim-tune`, `make failures`, `make
report`; snapshot figures, RESULTS.md is generated):

- **D28 (holdout decision): unchanged.** Re-applied mechanically to the corrected holdout qrels (sha256 prefix now
  `f3f4685399e2`): all three variants still pass the cross-fitted dev gate (filter +0.018 [+0.006, +0.029], promote
  +0.022 [+0.008, +0.036], demote +0.012 [+0.002, +0.023]); c = 2.00; simultaneous lower bounds filter **+0.010**
  (was +0.009), promote +0.005 (was +0.003), demote −0.006 (was −0.008); violations 0 for all. Eligible: filter and
  promote; the filter has the larger bound. `recsys_serve_needs` (served as `recsys_serve_v2`) still wins;
  `_holdout.json` was rewritten by the rule's code, nothing else changed.
- **D25: unchanged.** Every main-split NDCG@10 moved by at most 0.005 and no paired comparison changed sign:
  `recsys_serve` − `recsys_v1` on test is −0.036 [−0.053, −0.020] (was −0.035); in the simulation `recsys_serve`
  0.678 over rounds 2-10 vs 0.316 for `recsys_v1` and 0.699 for `recsys_v1_fb`, exploration −0.011 [−0.014, −0.008].
- **D27: unchanged.** In-sample dev, no variant meets rule 1 (filter −0.013 [−0.024, −0.003]); cross-fitted dev
  within 0.001 of before (`_needs_dev.json` recomputed, no judging).
- **D21: its rule now picks a different grid point.** `make sim-tune` on the corrected labels (2 new pairs judged):
  `light` 0.7749 vs `slots_only` 0.7744 over rounds 2-10, so the pre-stated rule (highest mean) now names
  `light`, by 0.0006, where the five light settings are within ±0.0015 of each other and each one's CI against
  route effects alone is about ±0.015. **Not acted on**: the registered systems and the served default keep
  `slots_only`. Reasons: the flip is far inside the comparison's noise and comes from grades the control shows
  are mostly re-judging noise; D21 itself says the in-sample dev comparison is the wrong basis and that the
  re-tune belongs on cross-fitted dev or real feedback (still open); and changing the feedback settings changes
  the served system, which is the coordinator's call (D25). Stated here so the call can be made.
- **The learned ranker's training labels: not retrained.** Of the 18,380 pinned labels (`ltr_train_v1.jsonl`), 398
  were re-judged and 121 now differ from the corrected grade (0.66%; 73 up, 48 down; most of them band changes at
  noise level); 32 of the 398 are cap pairs, 21 of which changed (all upwards). That is too small to move a LambdaMART model trained on
  18,380 rows in any way the evaluation could detect, and retraining would replace the committed ranker behind
  every system, including the one the holdout decision was taken on, so the D28 comparison would no longer be of
  the systems it pre-registered. The file stays as the record of what the ranker was trained on. Same for the MMR
  sweep labels (82 of 13,354 differ) and the ranker-API pilot labels (12 of 2,095).
- **Failure analysis.** H149 rose from 0.474 to 0.733 and left the holdout's ten worst; H002 (class C) took its
  place, so the class counts are unchanged. The test list has the same ten personas. The interim LLM-vs-LLM
  agreement on the owner set moved from QWK 0.758 to 0.753 (7 of its 150 pairs were re-judged).
- Not recomputed, snapshots of their time: `_cg_dev.json` (D13), `_ltr*.json` (pinned labels),
  `_failures_recsys_serve_dev.json` (D27's hand classes refer to it).

**Feedback simulation with the served default (D28 open item).** `make simulate` (test, 100 personas, every shown
route judged) now includes `recsys_serve_v2` and its no-exploration twin (a config override, not a registered
system). Over rounds 2-10 the served default is at the level of `recsys_serve` (+0.006 [−0.017, +0.029]): it starts
ahead (round 1 +0.022 [+0.006, +0.043]) and the lead shrinks every round, reversing by round 10 (−0.018 [−0.053,
+0.018]). A likely reason, not checked: the filter narrows the candidates, so fewer suitable routes are left once
the user has done or rejected some. Exploration costs −0.009 [−0.011, −0.006] on it. Reporting only; nothing is
decided on it. **The holdout simulation was not run**: its judging was estimated at 4.77 USD (2,254 pairs, `python
-m evalkit.simulate estimate --holdout`), more than was left of this task's 6 USD allowance; the command exists.

**Spend** (local ledger): 2.46 USD (re-judging 1.27, control 0.44, simulation and dev tuning 0.75).

**Still open** (D26's owner items unchanged): the stage's class-E fixes and an effort need, developed on
cross-fitted dev; re-tuning the feedback parameters on cross-fitted dev (D21, now with the note above); the
holdout feedback simulation if the cost is wanted; whether the stated-need filter should relax as the user
consumes routes (the simulation pattern above), to be developed on dev.

## D30. Feedback settings after the card fix: unchanged (2026-11-23)

On the corrected labels (D29) the D21 tuning rule names `light` over `slots_only` by a margin far inside the noise of the 40-persona dev tuning set. Switching would change the served system on a difference the tuning cannot resolve, so the served feedback settings stay as chosen in D21/D25. Revisit with real feedback data, not with more simulation.
