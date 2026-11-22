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
