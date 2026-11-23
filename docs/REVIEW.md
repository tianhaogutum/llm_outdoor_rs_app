# Owner review guide

BUILD_SPEC.md section 5 lists eight questions the owner should be able to answer without notes. Each answer
below is short and grounded in this project's results; the numbers are in generated blocks (`make report`
rewrites them from `eval/results/*.json`) or in the named table, never typed. Each answer says where the
evidence is and how to rerun the experiment by hand. All commands run from the repo root; unless marked
"live", they replay from the committed frozen cache and need no credentials.

Every number here rests on simulated users and an LLM judge that has not been checked against your labels.
Answer 2 is therefore the one to settle first.

## 1. Why pooling for judgements, and what bias does it introduce?

**Answer.** There are no real users and no ground truth, and grading every route for every persona is far too
expensive, so relevance is graded only where it matters for the metrics: the union of every system's top 10
per persona (TREC-style pooling), plus two pool-only contributors (a matcher that reads the hidden latent
profile, and plain BM25 of the request) so that the relevant set does not depend on what two weak baselines
happened to retrieve. A new system's unjudged top-10 routes ("holes") are judged on demand, so every reported
top 10 is fully judged. Pairs that break a hard constraint are graded 0 by rule, without an LLM call.

The bias: relevance is **relative to the pool**. Routes nobody put in a top 10 stay unjudged, so Recall@50 counts
only known relevant routes and NDCG's ideal ranking is a lower bound on the true ideal. The relevant set leans
towards what the latent matcher finds, which rewards systems that resemble it. The pool grows as systems are
added, so absolute numbers shift (`make eval-all` rescoring every stored run keeps tables mutually consistent).
Dev personas also got a deeper random sample for training the ranker, so dev and test absolute numbers are not
comparable; compare systems within a split.

<!-- BEGIN GENERATED review-pooling (make report; do not edit) -->
- Pool: top 10 of every registered system plus the pool-only contributors `pool_latent`, `pool_intent_lexical`.
- 23,764 judged pairs over 300 personas; 9,408 (39.6%) graded 0 by rule, 14,356 by the LLM judge.
- Known relevant (grade ≥ 2) routes per persona: median 18, min 1, max 49; personas without any: 0.
- Reference, the latent matcher that supplies much of the relevant set: NDCG@10 0.765 [0.733, 0.794] on test (`_reference.json`).
<!-- END GENERATED review-pooling -->

**Evidence:** [EVAL.md](EVAL.md) section 4; RESULTS.md "Data and judgements" and "Reference: the latent
matcher"; `backend/evalkit/pool.py`; `eval/qrels/qrels.jsonl` (every line records grade, reason, judge, batch).
**Rerun:** `cd backend && .venv/bin/python -m evalkit.evaluate --estimate recsys_serve` counts the holes a
system would add without judging; `make eval SYSTEM=recsys_serve` (no holes, replays); `make eval-all`.

## 2. What does agreement between my labels and the LLM judge tell me, and what would I do if it were low?

**Answer.** Weighted kappa between your 150 blind labels and the judge's grades on the same pairs says whether
the judge's grades mean what *you* mean by a good route for that person. Every NDCG, every ablation step and the
learned ranker (trained on the judge's grades) inherit the judge's view; agreement is what licenses reading
them as quality. It does not validate the simulated users or the requests they write, only the grading. Right
now there is no human figure at all: the interim check is a second LLM from another vendor, and two LLMs
agreeing is not evidence that either agrees with a person.

If agreement is low (as a rule of thumb, QWK below about 0.6), do not trust any number in RESULTS.md. Read the
disagreements, which `make kappa` lists largest first with the judge's reason: if they cluster (for example the
judge rewarding wishes the request never stated, FAILURES.md class C, or a systematic 2-versus-3 boundary), fix
the rubric (`eval/qrels/rubric_v1.md`) or the route card the judge sees, rerun the judge pilot against your labels
instead of the LLM reference, then re-judge the qrels (live, costs money), retrain the ranker on the new grades,
and rerun the evaluation. If agreement is moderate but the disagreements are noisy rather than systematic,
consider collapsing grades 2 and 3 and reporting binary metrics.

<!-- BEGIN GENERATED review-agreement (make report; do not edit) -->
- owner labels: **missing** (the 150 blind pairs are in `eval/owner_labels/`); interim, **not a human check**: a second LLM judge (`anthropic/claude-sonnet-5.5`) agrees with the production judge at QWK 0.758 (exact 0.61, within one grade 0.97) on the same 150 pairs.
- Judge pilot: production judge `g35f-min-b8` vs an LLM reference on 80 pairs: QWK 0.836, exact 0.79, mean grade +0.11 relative to the reference; between two batch orders QWK 0.864.
<!-- END GENERATED review-agreement -->

**Evidence:** RESULTS.md "Judge pilot" and "Judge validation"; [EVAL.md](EVAL.md) section 4; D5, D9.
**Rerun:** open `eval/owner_labels/label.html` in a browser, label all 150 pairs, press *Export labels*, save
the file as `eval/owner_labels/owner_labels.csv`, then `make kappa && make eval-all && make report` (the tables
then carry your QWK instead of "not yet validated"). The judge pilot: `make judge-pilot` (replays).

## 3. Why RRF rather than a weighted score sum for fusing BM25 and dense results?

**Answer.** RRF uses only ranks. BM25 scores are unbounded and depend on the query (how many terms, how rare
they are); cosine similarities of these embeddings sit in a narrow band. A weighted sum would need both put on
one scale per query and a weight that would change with the length of the request; RRF needs one constant, and
that constant barely matters on dev. Fusing beats either generator alone on test, in NDCG and in recall, while
neither generator alone is clearly better than the other: they find different routes. What was **not** done: a
calibrated weighted sum was never run as a competitor, so "RRF is better than a weighted sum" is an argument
from robustness, not a measurement.

<!-- BEGIN GENERATED review-rrf (make report; do not edit) -->
- Test NDCG@10 / Recall@50: BM25 `v2_bm25` 0.551 [0.514, 0.589] / 0.757 [0.713, 0.799]; dense `v3_dense` 0.529 [0.493, 0.564] / 0.726 [0.680, 0.774]; RRF `v4_rrf` 0.603 [0.568, 0.641] / 0.842 [0.809, 0.871].
- RRF − BM25 (test, paired): NDCG@10 +0.053 [+0.023, +0.083], Recall@50 +0.085 [+0.060, +0.108]. RRF − dense: NDCG@10 +0.075 [+0.049, +0.100].
- RRF constant on dev (Recall@50): k 10: 0.750 [0.712, 0.787], k 30: 0.749 [0.710, 0.783], k 60: 0.739 [0.699, 0.776], k 100: 0.730 [0.691, 0.769].
- Cosine band of one dev persona's eligible set: min 0.585, max 0.708 (BM25 scores are unbounded and query-dependent).
<!-- END GENERATED review-rrf -->

**Evidence:** RESULTS.md "Candidate generation" (generators, BM25 variants, RRF constant, embedded text);
[RECSYS.md](RECSYS.md) section 3; D13. **Rerun:** `make cg-tune` (dev experiments, replays) and
`make eval SYSTEM=v4_rrf`.

## 4. Which stage contributed the most NDCG, and which contributed nothing?

**Answer.** The hard filter contributed most, but read that carefully: the course project's scorer never sees
where the user lives, so most of its top 10 is out of reach and graded 0 by rule; the filter mostly fixes that.
Among the stages that rank feasible routes, BM25 (replacing the rule scorer) and the learned ranker give the
two largest steps, of similar size; fusing BM25 with dense retrieval adds a smaller one. Nothing measurable: dense retrieval on its own versus BM25, MMR, and routing the
request through the chat instead of using the typed text. The ranker API adds a little as a feature of the
learned ranker and costs a per-request call (D14), so the default does not use it. The served default's two
additions cost NDCG on the simulator on purpose: the ranker without the cap feature and the exploration slot.
The ranker's dev gains are in-sample and much larger than on test.

<!-- BEGIN GENERATED review-stages (make report; do not edit) -->
| Step | ΔNDCG@10 (test, paired) | ΔNDCG@10 (dev, paired) |
|---|---|---|
| `v0_rules` − `random` | -0.001 [-0.013, +0.010] | -0.001 [-0.007, +0.004] |
| `v1_filters` − `v0_rules` | +0.352 [+0.320, +0.383] | +0.337 [+0.312, +0.363] |
| `v2_bm25` − `v1_filters` | +0.181 [+0.132, +0.232] | +0.149 [+0.116, +0.180] |
| `v3_dense` − `v2_bm25` | -0.022 [-0.065, +0.023] | +0.022 [-0.004, +0.049] |
| `v4_rrf` − `v3_dense` | +0.075 [+0.049, +0.100] | +0.040 [+0.025, +0.056] |
| `v5_rank_api` − `v4_rrf` | +0.039 [+0.009, +0.067] | +0.065 [+0.046, +0.085] |
| `v6_ltr` − `v5_rank_api` | +0.132 [+0.094, +0.173] | +0.286 [+0.265, +0.309] |
| `v7_mmr` − `v6_ltr` | -0.005 [-0.011, +0.002] | -0.005 [-0.007, -0.002] |
| `recsys_v1` − `v7_mmr` | -0.011 [-0.025, +0.002] | -0.016 [-0.021, -0.010] |
| `recsys_serve` − `recsys_v1` | -0.035 [-0.053, -0.020] | -0.035 [-0.043, -0.028] |

Also: `v6_ltr_noapi` − `v4_rrf` (learned ranker without the API) +0.159 [+0.122, +0.194]; `v6_ltr` − `v6_ltr_noapi` (the API score as a feature) +0.012 [+0.001, +0.024]; questionnaire only → + intent (`recsys_v1_qonly` → `recsys_v1`) +0.395 [+0.359, +0.431]; + chat instead of the raw intent (`recsys_v1_chat` − `recsys_v1`) -0.012 [-0.037, +0.009]. Dev rows of the learned ranker are in-sample.
<!-- END GENERATED review-stages -->

**Evidence:** RESULTS.md "Phase 3 ablation", "Other comparisons", "Phase 4 end-to-end", "Served default".
**Rerun:** any row with `make eval SYSTEM=<name>` (for example `v4_rrf`, `v6_ltr_noapi`, `v7_mmr`), then
`make report`. MMR's trade-off: `make mmr-sweep` (dev, writes `docs/mmr_tradeoff.png`).

## 5. Where does the learned ranker overfit to the simulated users?

**Answer.** In four places, and the uncomfortable part is that only the first is fully measured.

1. **It memorises personas.** Its in-sample NDCG on dev is far above grouped cross-validation, and its
   system-level dev rows are in-sample; only grouped CV and test are honest.
2. **Its largest Phase 3 feature, `length_cap_ratio`, encodes the persona generator.** Personas' distance caps
   are their comfortable band times a random factor, and the judge reads the band, so "about two thirds of the
   cap" is a fact about the simulator. Removing it costs NDCG on the simulator, and the ranker without it is
   better when the cap is left empty, which real users may well do. That is why the served default drops it
   (D19, D25).
3. **It learns the judge, not people.** It is trained on the judge's grades, which read the same catalogue facts
   and the hidden profile; it ends up about as good as the matcher that reads that profile. That is circularity
   ([EVAL.md](EVAL.md) section 5), not evidence of quality for real users.
4. **It learns majority priors that fail for minorities.** The failure analysis shows the ranker overriding good
   retrieval for an advanced mountain biker (a higher technical grade is penalised), a family with small children
   (short routes are penalised) and a stroller (one concept among several). These are regularities of the
   persona mix, and the training split has few such personas. With real data some of them may be true and some
   not; the simulation cannot tell which.

Related: the feedback parameters chosen on dev disagree with test (D21), probably because the in-sample ranker
leaves feedback less to repair on dev.

<!-- BEGIN GENERATED review-overfit (make report; do not edit) -->
- `v6_ltr` on judged candidates: in-sample 0.968, grouped CV 0.808 [0.790, 0.826], test 0.828 [0.799, 0.853].
- System level: `v6_ltr` dev (in-sample) 0.912 [0.897, 0.926] vs test 0.775 [0.746, 0.803]; `recsys_v1` dev 0.892 [0.876, 0.907] vs test 0.759 [0.733, 0.784].
- `length_cap_ratio`: 25.1% of the gain of the Phase 3 ranker. Without it (`recsys_v1_nocap` − `recsys_v1`): test -0.024 [-0.042, -0.009], dev -0.024 [-0.030, -0.018]. Without the whole suspicious group (`recsys_v1_nosusp`): test -0.034 [-0.051, -0.018].
- Cap slot left empty: `recsys_v1` loses -0.180 [-0.212, -0.149], the served system -0.114 [-0.142, -0.086]; served minus Phase 3 without the cap +0.031 [+0.001, +0.064].
- Feedback parameters: chosen on dev (`slots_only`); on the test simulation the light full loop beats the dev choice by +0.025 [+0.014, +0.036] (rounds 2-10).
<!-- END GENERATED review-overfit -->

**Evidence:** RESULTS.md "Learned ranker", "Robustness of the learned ranker", "Empty profile slots",
"Reference"; [RECSYS.md](RECSYS.md) section 4; [FAILURES.md](FAILURES.md) class A; D15, D19, D21.
**Rerun:** `make ltr` and `make ltr-robust` retrain on the pinned dev labels (LightGBM; on macOS the Makefile
sets the OpenMP library path inline); `make eval SYSTEM=recsys_v1_nocapslot` and
`make eval SYSTEM=recsys_serve_nocap` replay the empty-cap checks.

## 6. What is the cost and p95 latency per request, and which stage dominates?

**Answer.** The served default makes one upstream call per request with a new intent, the query embedding, and
that call dominates both cost and latency; local compute (filter, BM25, ranker trees, MMR, exploration) is a
small fraction, and a repeated intent is served from the cache. With the ranker API the rerank call dominates
and costs far more per request, for no measurable gain. The warm benchmark is one laptop and one network; the
first request after start-up is slower because the Vertex client is created lazily, and the embedding call has a
long tail, which the serving layer bounds with a deadline and a BM25-only fallback.

<!-- BEGIN GENERATED review-cost (make report; do not edit) -->
Warm benchmark (`make latency`: 60 sequential requests after 5 warm-up requests, LLM cache off, live calls, one arm64 laptop); cost per request = mean nominal upstream cost over the test personas.

| System | Ranker API | p50 ms | p95 ms | max ms | Largest stage (p95, upstream) | Local compute p95 ms | Cost/request | p95 budget |
|---|---|---|---|---|---|---|---|---|
| `recsys_serve` | no | 100 | 203 | 4933 | dense 197 ms | 11.1 | 0.000005 USD | 500 ms (within) |
| `recsys_v1` | no | 104 | 239 | 4055 | dense 220 ms | 20.0 | 0.000005 USD | 500 ms (within) |
| `v7_mmr` | yes | 510 | 759 | 2101 | rank_api 563 ms | 24.3 | 0.000782 USD | 1500 ms (within) |

Median time per stage of `recsys_serve` (local + upstream): dense 89.3 ms, ltr 5.1 ms, filter 0.7 ms, mmr 0.3 ms, explore 0.2 ms, bm25 0.2 ms, fusion 0.1 ms. Ranker API on (`v7_mmr`) − `recsys_v1`: NDCG@10 +0.011 [-0.002, +0.025] on test.
<!-- END GENERATED review-cost -->

**Evidence:** RESULTS.md "Latency and cost against the budget" and "Latency"; [SERVING.md](SERVING.md)
sections 4-5 (measured by hand on a running server). **Rerun (live, needs Vertex credentials, a fraction of a
cent):** `cd backend && .venv/bin/python -m evalkit.latency --systems recsys_serve`, then `make report`.

## 7. Which failure cases remain, and what data would fix them?

**Answer.** On the ten worst test queries of the served default, most relevant routes were retrieved and then
ranked down by the learned ranker. The largest class (A) is the ranker overriding retrieval for a stated
minority need (mountain bikers who want a challenge, strollers, small children, an upper bound on length read as a
target); class B is length the served pipeline cannot read (a vague request, a four-day tour); classes C and D are
the evaluation's fault rather than the system's (the judge penalises wishes the user never stated, such as
arriving by train). Counts per class are in the block below. The data that would fix them: slots for
stroller, train arrival, technical skill and shape mapped to catalogue fields that already exist (steps,
`Train access`, technical grade, loop), a comfortable-distance slot and day parsing, more minority personas in
training or real feedback from such users, and intents that state every hard requirement (or a judge that grades
only what the request says).

<!-- BEGIN GENERATED review-failures (make report; do not edit) -->
Known-relevant routes (grade ≥ 2) of the 10 worst test queries of `recsys_serve`, by where they dropped out of the top 10: exploration 1, filter 0, mmr 3, ranker 76, retrieval 34; shown 40 (of 154).

- Class A: 5 of 10 (P018, P260, P145, P003, P177)
- Class B: 2 of 10 (P110, P218)
- Class C: 2 of 10 (P190, P225)
- Class D: 1 of 10 (P048)
<!-- END GENERATED review-failures -->

**Evidence:** [FAILURES.md](FAILURES.md) (generated traces and hand-written diagnoses); `eval/results/_failures.json`.
**Rerun:** `make failures && make report` (replays each request from the frozen cache and checks it against the
stored run).

## 8. What would I measure first if this had real users?

**Answer.** Three things, in this order. (1) **How users fill in the profile:** the share of requests without a
home, a radius or a cap, and what they type. The empty-slot rows show the served system losing most of its
quality without a home, and the whole case for the served ranker (D19) rests on users leaving the cap empty or
meaning it differently; that is a cheap log query, not an experiment. (2) **An interleaving test of
`recsys_serve` against `recsys_v1`**, the decision with the least offline support, on clicks and route-detail
opens; interleaving needs far less traffic than an A/B test. (3) **Whether the judge predicts behaviour:**
completion rate by judged grade on the routes users were shown. Log, from day one, the full per-stage trace of
every list, impressions, and the exploration pick's propensity, so that the exploration slot can later be
evaluated counterfactually and an A/B test of exploration (randomised by user, measured over weeks) can follow.

**Evidence:** README "What would change with real traffic"; RESULTS.md "Empty profile slots"; D19, D22, D25.

## What is still open for you

1. **The 150 blind labels (D5).** `eval/owner_labels/label.html` is self-contained (rubric, both cards, grades
   saved in the browser); export `owner_labels.csv` into `eval/owner_labels/`, then
   `make kappa && make eval-all && make report`. If agreement is low, follow answer 2: read the disagreements,
   fix the rubric or the route card, re-judge (live, costs money), retrain the ranker (`make ltr-sample`,
   `make ltr`, `make ltr-robust`), re-evaluate. Then update the prose that says the judge is unvalidated
   (README "Limitations", [EVAL.md](EVAL.md) sections 4 and 8).
2. **Deployment (D6).** Nothing is deployed. The checklist in [DEPLOY.md](DEPLOY.md) section 6: target and URL,
   spend ceiling and provider-side budget alerts, which providers are live (none by default), the EU-region
   question for the chat (D18), OSM attribution on every screen with route data, the Harry Potter themed texts
   and graphics inherited from the course project, Vercel Analytics, data retention, and re-selecting the
   pre-generated narratives for `recsys_serve`.
3. **Items marked pending in [DECISIONS.md](DECISIONS.md)** (D26 lists them in one place):
   - D3: whether Outdooractive's terms allowed the course project's 111 route texts in the public `app.db`
     history (the file is no longer tracked, but it is in `main`'s history).
   - D18: whether chat messages must stay in the EU (every chat candidate except a retired model leaves it).
   - D25: the served default was chosen by the coordinator, not by you; the alternatives are registered and one
     setting away (`RECSYS_SYSTEM`).
   - [LLM_CLIENT.md](LLM_CLIENT.md): the narrative generation model `gemini-2.5-flash` is past its retirement date
     in the region used; switching means `global` (data leaves the EU) or waiting.
   - D21 (technical, not a decision for you): re-tune the feedback parameters once a cross-fitted dev ranker or
     real feedback exists.
