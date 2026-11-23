# Offline evaluation protocol

How recommenders are evaluated before there is any real traffic, and what the numbers can and
cannot tell you. Results are in [RESULTS.md](RESULTS.md) (generated, never hand-edited); decisions
are in [DECISIONS.md](DECISIONS.md) D9-D10; the catalogue is described in [DATA.md](DATA.md).

**Read every number with this in mind: the users are simulated, relevance is judged by an LLM,
and the judge has not yet been checked against a human (the owner's 150 labels are pending).**

## 1. Pipeline at a glance

```
personas (latent profile -> questionnaire -> user vector; profile slots; LLM-written intent)
   │  persona.to_query(): only the visible part
   ▼
system (registry: random, v0_rules, v1_filters ... v7_mmr, recsys_v1)  ->  eval/runs/<system>.jsonl (top 50 + trace)
   │
pool = top-10 of every system ∪ top-10 of two pool-only contributors
   │
holes -> hard-constraint violation? grade 0 by rule : LLM judge (batched, cached)  ->  eval/qrels/qrels.jsonl
   │
metrics per persona -> bootstrap over personas (test, dev, all)  ->  eval/results/<system>.json
   │
make report  ->  docs/RESULTS.md
```

| Command | What it does |
|---|---|
| `make personas` | regenerate `eval/personas/personas.jsonl` (byte-identical from cache) |
| `make eval SYSTEM=<name>` | run the system on all 300 personas, judge pool holes, write results |
| `make eval-all` | rescore every stored run against the current qrels (tables stay mutually consistent) |
| `make report` | render `docs/RESULTS.md` from `eval/results/*.json` |
| `make judge-pilot` | judge pilot (cached) |
| `make owner-set` / `make kappa` | blind labelling set + interim second judge / agreement report |
| `make freeze` / `make replay-check` | export the frozen LLM cache / prove a credential-free replay |

Code: `backend/app/recsys/` (shared with the API: catalogue, `UserQuery`, result and trace types,
hard constraints, systems registry) and `backend/evalkit/` (harness). Data artefacts: `eval/`.

## 2. Simulated users (personas)

300 personas from one seeded generator (`backend/evalkit/personas.py`; every parameter is in
`eval/personas/generator_params.json`; summary statistics in `eval/personas/stats.json` and
`eval/results/_personas.json`). Each has three layers.

**Latent profile (never shown to a system).** Sampled in this order, each step conditioned on the
previous ones:

1. *Activity*: hiking 68% / cycling 32%; within cycling a latent taste for touring (72%) or
   mountain biking (28%). The UI tab is `hiking` or `cycling`. **Running is not simulated**: the
   catalogue has 55 running routes, too few for a meaningful eligible set.
2. *Home*: a real Bavarian town or city (407 OSM `place=city|town` nodes with a population tag,
   inside the Bavarian boundary, from the same Geofabrik snapshot as the catalogue;
   `eval/personas/towns.json`, built once by `python -m evalkit.towns`). Weight =
   population^0.5: population weighting alone would put about a fifth of all personas in Munich;
   the square root keeps all seven Regierungsbezirke represented (88 Oberbayern, 49 Schwaben,
   43 Mittelfranken, 39 Oberfranken, 35 Unterfranken, 26 Oberpfalz, 20 Niederbayern; 197 distinct
   towns). Villages are not homes: their residents are represented by the nearest town.
3. *Party*, conditioned on activity: solo, couple, friends, family with young children (3-6),
   family with school children (7-13), parent with teenager (MTB), parents with a stroller,
   with a dog, older walkers/riders (65+). Some couples and families also bring a dog.
4. *True fitness* on the 0-3 route scale, normal per party (families, stroller and older lower,
   friends higher, MTB +0.4), clipped.
5. *Comfortable distance band* around a centre that grows with fitness, capped per party
   (e.g. 5.5 km for young families hiking), lognormal noise; *absolute distance cap* = band top
   × U(1.25, 1.7). Multi-day trips (5% of fit hikers, 10% of fit touring cyclists; never families)
   scale band, cap and ascent by the number of days.
6. *Ascent tolerance* grows with fitness, capped per party (250 m young families, 150 m stroller,
   450 m school families, 600 m older). *Technical tolerance* (SAC T1..T4+, MTB S0..S3+): 0 for
   families, stroller, older and touring cyclists; grows with fitness otherwise.
7. *Transport*: train users (15%, 35% in cities > 100k) get a hard wish for a start reachable by
   train. *Travel radius*: 20-80 km (car) or 25-60 km (train).
8. *Desires*: 2-4 main wishes (weight 0.6-1.0) and 0-2 nice-to-haves (0.2-0.5) over things the
   catalogue records (lake, swimming, refreshment, mountain hut, beer garden, summit, viewpoint,
   waterfall/gorge, caves/rocks, quiet nature, culture, river, playground, wildlife), drawn from
   propensities per activity, boosted per party (families: playground, lake, swimming; older:
   refreshment, culture; dog: quiet nature, water). Huts are rare outside Oberbayern/Schwaben.
9. *Shape* (loop / one-way / any; families mostly loops, train users more often one-way),
   *surface* (stroller paved; MTB natural; touring often paved), *dislikes* (alpine terrain, busy
   roads, crowds, steep sections) by party.

**Eligible set.** If fewer than 40 routes pass the persona's hard constraints, or fewer than 10 of
the persona's own activity (MTB routes for an MTB persona), the travel radius is widened in steps
of ×1.25 up to 150 km (people in sparse areas drive further). This happened for 40 personas, mostly
cyclists. Eligible-set sizes: min 40, p10 61, median 282, p90 927 (hiking median 400, touring 74,
MTB 185 of which a median of 15 are MTB routes).

**Questionnaire (what the product's form collects).** Answered from the latent profile the way a
person would, coarsely and with noise: true fitness < 1 → beginner, < 2 → intermediate, else
advanced, then misjudged by one level with probability 0.17 (overestimate) / 0.08 (underestimate)
(53 overestimate, 28 underestimate). Adventure types follow the desires (culture → history-culture;
a nature wish → natural-scenery; families → family-fun) with misses and an 8% "tick everything".
The narrative style is random (families lean playful). The **user vector is exactly what
`translate_questionnaire_to_vector` returns** for these answers (a test pins this for all 300).

**Profile slots (what a product form could plausibly collect).** Home coordinates, maximum travel
radius, absolute distance cap. The hard filter may use them; `v0_rules` does not.

**Free-text intent.** Written by `openai/gpt-6-luna` (OpenRouter) from the latent profile, in one
of four lengths (vague 3-9 words 17%, short 27%, medium 38%, detailed 18%) and five voices, with
instructions to mention only what is on the person's mind, paraphrase, round numbers, avoid
catalogue tag names and not invent wishes. The writer is a different vendor from the judge
(section 5). Cached at temperature 0.9, so the file is deterministic from cache. Median 27 words
(p10 7, max 51).

**Split.** 200 dev / 100 test, by persona, fixed by the seed. Phase 3 trains only on dev.

Examples (test split unless noted; latent → intent):

| id | latent summary | questionnaire | intent |
|---|---|---|---|
| P020 | hiking, family with children 4 and 6, Viechtach, 40 km, 4-8 km (cap 10), ≤ 250 m; wants quiet nature, playground, swimming; loop | intermediate, natural-scenery | "Quiet forest walk with a playground" |
| P018 | hiking with a stroller, Forchheim, 30 km, 4-9 km (cap 13), ≤ 150 m, paved; caves/rocks, playground | intermediate, natural-scenery + family-fun | "Could you suggest an easy, stroller-friendly walk of around 8 km with an interesting cave or rock formation and a playground for the children? Please keep the climbing gentle." |
| P014 | older walkers, Günzburg, 30 km, 6-14 km (cap 21), ≤ 600 m; culture, caves/rocks, lake; dislikes steep | intermediate, all three types | "An easy walk to some castle ruins or interesting rock formations, without steep sections." |
| P003 | MTB, five friends 18-29, Burgkunstadt, 60 km, 32-68 km (cap 96), ≤ 1550 m, up to S3; beer garden, lake, quiet nature | advanced, natural-scenery | "A quiet, challenging mountain bike ride with forest trails beside a lake and a beer garden along the way. For a group of friends, about 30–70 km." |
| P032 (dev) | touring cyclist, solo, Augsburg, 39 km, 48-103 km (cap 152); beer garden, culture, river | intermediate, natural-scenery | "Could you suggest a cycling route near Augsburg that follows a river and passes a castle or monastery, with a beer garden along the way? Around 60 km would suit me, on roads, cycleways or good tracks rather than alpine terrain." |

## 3. What a system may see

`persona.to_query()` builds a `UserQuery` (`app/recsys/query.py`) with: the user vector, the
category tab, the profile slots (home, travel radius, distance cap), the intent, optional feedback
and a request id. Nothing else. A test asserts the field set and that changing a persona's latent
profile does not change its query. `random` sees only the tab; `v0_rules` sees the vector and the
tab (exactly like the product today: `rank_routes_v0` over the tab's routes in id order).

## 4. Pooling and judgements

**Pool.** For each persona, the union of the top-10 of every registered system plus two
**pool-only contributors** (`backend/evalkit/pool.py`). Like the manual runs in TREC they are not
competitors, never appear in result tables, and exist only so that the judged relevant set does
not depend on what two weak baselines happened to retrieve:

- `pool_latent`: a hand-written utility over the latent profile and catalogue facts
  (multiplicative fit for distance band, ascent, technical grade, party, dislikes, shape, surface,
  times desire coverage), restricted to routes that pass the hard constraints.
- `pool_intent_lexical`: BM25 of the intent over title + description + tags, same restriction.

Why it matters: NDCG's ideal DCG and Recall@50's denominator come from the judged pairs. If only
the baselines were pooled, a persona's "ideal" would be the best of the baselines' top-10s (almost
all violating the radius), so a later system finding genuinely good routes would be judged against
an artificially low ideal and its recall would be undefined. With the contributors, the 300
personas have 2,707 known relevant (grade ≥ 2) pairs, median 10 per persona (min 1, max 21), and
every persona has at least one.

**Bias that remains.** (1) The relevant set is dominated by one contributor: `pool_latent` found
2,147 of the 2,707 relevant pairs, 1,701 of them found by no other run (`pool_intent_lexical` 951 /
509 unique; `random` 26, `v0_rules` 34). The ideal DCG therefore leans towards what a latent-profile
matcher finds. A system that finds relevant routes of a different kind is still judged correctly
(holes are judged on demand, see below), but routes nobody has retrieved stay unjudged, so recall
is relative to the pool and NDCG's ideal is a lower bound on the true ideal. (2) Pool depth is 10.
Routes a system ranks 11-50 count for Recall@50 only if some run put them in a top-10. (3) The pool
grows as systems are added, so absolute numbers shift. That is why `make eval-all` rescoring all
stored runs against the current qrels is the only way to get mutually consistent tables.

**Holes are judged on demand.** `make eval SYSTEM=x` pools x's top-10 with the contributors and
grades every pair not yet in the qrels (within `LLM_BUDGET_USD_PER_RUN`). Each results file reports
`judged_share_at_10`; it is 100% for every reported system, and NDCG/MRR are only computed for
fully judged top-10 lists.

**Rule-graded pairs.** A pair that violates one of the three deterministic hard constraints
(wrong category tab, start beyond the travel radius by great-circle distance, longer than the
distance cap; `app/recsys/constraints.py`) gets **grade 0 by rule, without an LLM call**. That
holds for 5,598 of the 11,347 pooled pairs (49.3%; radius 4,836, radius + cap 699, cap 63). The same
predicates drive the violation-rate metric and the Phase 3a filter.

**Rubric.** `eval/qrels/rubric_v1.md` (version tag `rubric/v1` on every qrels line):
3 = fits the intent and every constraint of this person; 2 = fits the main intent with at most a
minor mismatch; 1 = feasible and the right activity but misses the point or has a clear mismatch;
0 = wrong activity, unsuitable or unsafe for this party, or irrelevant. It also tells the judge
how to read the data's known weaknesses (proximity-derived features: "Refreshment stops
available" is on 66% of routes; DEM ascent overestimates flat lake and forest circuits; a loop's
start is not necessarily a trailhead; missing features are not proof of absence).

**LLM judge.** Sees the persona's full latent profile, the intent and a factual route card (title,
type, shape, start locality and *straight-line distance from the persona's home*, length,
ascent/descent, highest point, DIN moving time, difficulty and hardest grade, surface shares,
named features near the line with counts, protected areas, access at the start, tags, and the OSM
mapper's note if any; the generated description is not included because it restates the same
facts). Persona and catalogue text are wrapped with `wrap_untrusted` and the system prompt carries
`UNTRUSTED_DATA_RULE` (a test checks that an injected closing tag in an intent is neutralised).
Routes of one persona are graded 8 per call (balanced batches), in a seeded order; temperature 0;
structured output (`{"grades": [{"route", "reason", "grade"}]}`) validated by the client; a batch
that fails the schema is split in halves, a skipped label is re-graded alone. Every qrels line
stores grade, reason, judge config, the model id reported by the provider, rubric, and the batch it
was graded in (so the exact call can be replayed).

**Choosing the judge (pilot, `make judge-pilot`).** 80 pairs from 20 dev personas stratified over
activity style and party; per persona one of `pool_latent`'s top 3, one of its ranks 4-15, one
feasible route from `v0_rules`' top 50 (else a mid-ranked feasible one) and one random feasible
route, so the pilot spans the grade range. Reference: `anthropic/claude-opus-5.5` with reasoning,
grading the 4 pilot pairs of a persona alone. Candidates grade them inside a batch of 8 (4 filler
routes of the same persona). Batch-order sensitivity: the same pilot pairs again with different
fillers and order. Results (nominal cost per graded pair):

| Candidate | QWK vs reference | exact | cost/pair | QWK between batch orders |
|---|---|---|---|---|
| gemini-2.5-flash, thinking off, batch 8 | 0.691 (0.779 2nd order) | 0.60 | 0.00026 | 0.822 |
| gemini-2.5-flash, thinking 1024, batch 8 | 0.796 | 0.71 | 0.00056 | – |
| gemini-2.5-flash, thinking off, batch 4 | 0.778 | 0.70 | 0.00031 | – |
| gemini-3.5-flash (global), thinking low, batch 8 | 0.828 (0.812) | 0.78 | 0.00189 | 0.915 |
| **gemini-3.5-flash (global), thinking minimal, batch 8** | **0.836 (0.807)** | 0.79 | 0.00116 | 0.864 |

Chosen: **`g35f-min-b8`** (`gemini-3.5-flash`, `global` endpoint, thinking `minimal`, batches of
8). It agrees with the reference as well as the "low" configuration, at about 60% of its cost (the
full pool cost 7.26 USD; "low" was measured at about 0.0024 USD per pair live, about 14 USD).
Every Phase 3 system will add holes, so the cost compounds. It is a GA model (retirement no
earlier than 2027-05), whereas `gemini-2.5-flash` is past its documented retirement date and,
without thinking, was the least stable. The price: lower batch-order stability than "low" (0.864
vs 0.915 QWK between orders; 80% vs 88% exact). All candidates grade slightly higher than the
reference (+0.04 to +0.11 on the 0-3 scale). The pilot is small (80 pairs; a QWK difference of a
few hundredths is noise) and the reference is itself an LLM, not ground truth. See D9.

**Owner checkpoint (prepared, not faked).** `eval/owner_labels/label.html` (self-contained: the
rubric, both cards, 0-3 per pair, saved in localStorage, CSV export) and `sheet.csv` hold 150
LLM-judged pairs stratified over judge grade (38/38/37/37) and split (100 dev / 50 test), at most two
per persona per grade, without judge grades. Rule-graded pairs are excluded: they are 0 by
definition. `make kappa` reports "owner labels missing" until
`eval/owner_labels/owner_labels.csv` exists, then quadratic-weighted kappa, the confusion matrix
and the disagreements largest first (`eval/results/_kappa.json`). **Interim signal only:**
`anthropic/claude-sonnet-5.5` (another vendor, reasoning low) graded the same 150 pairs: QWK 0.758
against the production judge, exact agreement 0.61, within one grade 0.97. Two LLMs agreeing is
not evidence that either agrees with a person. If the owner's QWK comes out low (say below 0.6),
the rubric or the cards are wrong for this domain: fix them and re-judge before trusting any
number here.

## 5. Guard against circularity

The concern: if relevance were defined by the same function a system optimises, the system would
look good by construction. Why judged relevance here is **not** `v0_rules`' scoring function:

- `v0_rules` is a fixed weighted sum of three sub-scores on the *questionnaire vector*: difficulty
  range (from the self-reported fitness), length range (three fixed bands) and Jaccard overlap with
  13 questionnaire tags (only `scenic`, `flora`, `fauna` exist in the catalogue vocabulary, D4).
- Relevance is graded by an LLM from the *latent* profile and the *free-text intent*, using
  attributes the rule scorer never sees: distance from home and travel radius, party (children's
  ages, stroller, dog, older walkers), ascent tolerance, technical grade, specific wishes (lake,
  hut, waterfall, train access...), loop vs one-way, surface, dislikes. The questionnaire is
  noisy by design (a quarter of personas misjudge their fitness), so even the shared notions
  (difficulty, length) differ between the vector and the truth.
- The outcome confirms they are different functions: `v0_rules` is no better than `random`
  (RESULTS.md).

Where circularity can still creep in:

1. **Same information on both sides of the latent profile.** Intents are written *from* the latent
   profile and the judge also reads the latent profile. A system that uses the intent well is
   rewarded for recovering what the generator put in, which is the point, but the intent writer's
   paraphrases cannot contain wishes the latent profile lacks. Real users have wishes no
   structured profile captures. Mitigation: the writer (OpenAI) and the judge (Google) are
   different vendors, so the judge does not grade text phrased by itself. The pilot reference and
   the second judge are from a third vendor (Anthropic).
2. **Generated descriptions.** Route descriptions are generated from the same OSM facts the judge
   reads on the route card (DATA.md section 14). A text retriever over descriptions is therefore
   matching the judge's evidence, not independent prose. The judge card leaves the generated
   description out, but the facts are the same.
3. **Pool-only contributors.** `pool_latent` uses the latent profile directly, as the judge does;
   it supplies most known-relevant pairs. A Phase 3 learned ranker trained on dev qrels learns
   partly what this matcher and the judge agree on. Keep `pool_latent` out of every system, and
   report test-split results only.
4. **Same vendor family over time.** If a later stage uses Gemini (for example to parse intents
   or embed them), it shares training data and biases with the judge. Say so wherever it applies.
5. **Rubric awareness.** The rubric names catalogue weaknesses (proximity tags, DEM ascent). A
   system tuned on dev results could learn the judge's reading of those rather than users'.

## 6. Metrics (exact definitions; `backend/evalkit/metrics.py`, hand-computed unit tests)

Per persona, on the system's top 50 (`RUN_DEPTH`), with the persona's judged pairs as `qrels`:

- **NDCG@10**: DCG = Σᵢ (2^gᵢ − 1) / log₂(i + 1) over ranks i = 1..10; ideal DCG from the persona's
  10 best judged grades (all judged pairs, not only the retrieved ones). Undefined (excluded and
  counted) if the persona has no judged pair with grade > 0. Raises if a top-10 route is unjudged.
- **MRR@10**: 1 / rank of the first route with grade ≥ 2 within the top 10, else 0.
- **Recall@50**: share of the persona's known grade ≥ 2 pairs that appear in the top 50.
  **Pool-relative**: routes nobody pooled are unknown; a route at rank 11-50 that no run put in a
  top-10 is unjudged and counts as not relevant.
- **Violations@10**: share of the top 10 that violate any hard constraint.
- **Coverage@10**: distinct routes in all personas' top-10 lists ÷ catalogue size (9,347). A
  statistic of the whole set, not a per-persona mean, so it has **no bootstrap CI** (a
  with-replacement bootstrap biases it downwards). It grows with the number of personas, so compare
  systems only on the same split.
- **ILD@10** (intra-list diversity): mean pairwise (1 − Jaccard) between the catalogue tag sets of
  the 10 routes. Tags come from the pipeline's fixed vocabulary, so the measure does not depend on
  any learned model and stays comparable across phases. Caveat: shape and access tags (loop,
  parking, bus) count as much as content tags.
- **Cost per request**: mean nominal upstream cost (LLM / embedding / rank calls recorded in the
  trace, at their original price even when served from cache). Baselines: 0.
- **Latency p50 / p95**: nominal = locally measured compute + recorded upstream latency. The local
  part varies run to run, so it lives in the results' `volatile` block, together with this run's
  ledger. The deterministic part reports upstream latency only.

**Confidence intervals.** 95% percentile bootstrap over personas (1,000 resamples, fixed seed
20261122), per split. **Paired bootstrap** for differences between two systems: per-persona
differences resampled with the same indices, giving a CI for the mean difference and P(Δ ≤ 0).
This is what the Phase 3 ablation table uses to decide which stage adds nothing. Headline numbers
are on the **test** split (100 personas); dev (200) is shown separately and is what tuning may
look at.

## 7. Determinism and the frozen cache

- Persona generation, pooling, batching and bootstrap are seeded. All JSON is written with sorted
  keys and rounded floats.
- Every LLM call goes through `app.llm` (cache key = provider + model + full prompt + parameters).
  A second `make eval` makes no LLM call and produces a results file **byte-identical except for
  its `volatile` block** (sorted last: local timings, run ledger, timestamp). Checked on the real
  results and by `tests/evalkit/test_pipeline.py` with the mock provider.
- `make freeze` re-issues, from the local cache, every call behind the committed artefacts (300
  intents, the 885 judge batches recorded in the qrels, the pilot, the second judge) and exports
  exactly those 1,484 entries to `eval/cache/` (300 KB). The harness adds `eval/cache/` as a
  frozen layer unless `LLM_CACHE_FROZEN_DIRS` is set.
- `make replay-check` runs everything in a subprocess with an **empty cache directory,
  `LLM_CACHE_ONLY=true` and blanked credentials**: it regenerates the personas (identical intents),
  replays every recorded judge batch (identical grades), re-runs the pilot (identical summary) and
  recomputes all results (identical outside `volatile`). A cache miss fails instead of calling a
  provider. On a fresh clone `make eval SYSTEM=v0_rules` therefore needs no credentials. A new system
  with unjudged holes does need a judge (and spends money).
- Caveat: the cache key embeds Pydantic's JSON schema of the judge output; a Pydantic upgrade that
  changes schema generation invalidates the frozen judge entries (replay would then fail loudly).

## 8. Limitations, plainly

- **Simulated users.** The latent model is ours: its correlations (families avoid climbs, older
  walkers like inns) are plausible, not measured. Nobody's real preferences are in this data. A
  system can only be shown to serve *these* personas well.
- **LLM-judged relevance, unvalidated.** Until the owner's 150 labels exist, agreement with a human
  is unknown. LLM-LLM agreement (QWK 0.76) is an interim plausibility check only. Judges grade
  slightly generously relative to the reference.
- **Judge stability.** The production judge's grades depend somewhat on batch composition (QWK 0.86
  between two batchings in the pilot). Holes judged later sit in different batches than earlier
  pairs, which adds noise of that size to comparisons between systems pooled at different times.
- **Catalogue facts are imperfect** (DATA.md section 15): proximity-derived tags, DEM ascent error
  (flat lake loops), loop starts that are not trailheads, generated descriptions, OSM mapping bias
  (Franconia dense, Allgäu thin, few MTB routes). The judge reads the same imperfect facts a system
  does; a route that is great in reality but poorly mapped is judged poorly.
- **Pool bias** (section 4): relevance is pool-relative and leans towards the latent matcher.
- **The baselines are nearly blind to the constraints.** Neither `random` nor `v0_rules` sees the
  home location, so about 93% of their top-10 is rule-graded 0. Their NDCG mostly measures that,
  not ranking quality among feasible routes. Phase 3a's hard filter shows the split (RESULTS.md ablation).
- **Small test split.** 100 personas: CIs on NDCG are wide relative to the baselines' values; use
  paired comparisons.
- **No running, no seasons, no weather, no opening hours, no crowding data.** Single-request
  evaluation: no sessions, no feedback loop yet (Phase 5).

## 9. Phase 3 additions to the protocol

The pipeline itself is described in [RECSYS.md](RECSYS.md); this section lists what Phase 3 changed in
the harness and in the data the numbers rest on.

- **Systems with upstream calls.** `make eval` injects one LLM client for the systems' own calls
  (query embeddings on Vertex, the ranker API on OpenRouter; mock under `LLM_PROVIDER=mock`), with the
  committed frozen layer. Async systems (`arecommend`) run 8 personas at a time; their local time is the
  sum of the stages' own compute and excludes waiting for upstream calls, whose *recorded* latency is
  added as nominal upstream latency. Recorded latencies include the cold start of the first calls of a
  run; the warm end-to-end benchmark is `make latency` (`eval/results/_latency.json`, machine-dependent,
  not replayed).
- **Hole estimates.** `python -m evalkit.evaluate --estimate <systems...>` runs the systems and counts
  pool holes and the judge's expected cost without judging or writing anything. Each Phase 3 system was
  estimated before its `make eval`.
- **Ablation and paired differences.** `make report` renders the ablation chain
  (`evalkit/report.py: ABLATION`) on the test split with paired bootstrap differences between
  consecutive rows and a few further pairs (`EXTRA_PAIRS`).
- **Deeper judgements for training (dev only).** The learned ranker's training sample added judged pairs
  outside any system's top 10: a seeded uniform sample of unjudged `v4_rrf` candidates of every **dev**
  persona, marked `"sample": "ltr-train/v1"` in `qrels.jsonl`. Consequence: dev personas have more
  judged pairs than test personas, so dev ideal DCG and the Recall@50 denominator are computed from a
  deeper pool; dev and test absolute numbers are less comparable than before Phase 3 (compare systems
  within a split). Test qrels are still top-10 pooling only. The training labels are pinned in
  `eval/qrels/ltr_train_v1.jsonl` so later judgements cannot leak back into training.
- **Dev results of the learned ranker are in-sample.** Its training labels are the dev judgements, so
  `v6_*`, `v7_mmr` and `recsys_v1` on dev are optimistic. Decisions about the ranker used grouped
  cross-validation on dev (`_ltr.json`), never its system-level dev row and never test.
- **Dev experiments without new judgements** (`_cg_dev.json`, `_rank_pilot.json`) use a *condensed*
  NDCG@10 (unjudged routes dropped before the cut) plus pool-relative Recall@50; the MMR sweep
  (`_mmr_sweep.json`) judges every top 10 on dev.
- **Reference row.** `_reference.json` scores the pool-only latent matcher with the same metrics; the
  report shows it in a separate table labelled "not a system". It reads the simulated user's latent
  profile and is in no comparison.
- **Frozen replay** now also re-runs every stored system that makes upstream calls (and the ranker pilot
  and MMR sweep) from the frozen layer and compares rankings, so query embeddings and ranker scores are
  in `eval/cache/`. The catalogue embedding matrix is not in the cache: it is the committed artefact
  `backend/app/recsys/artifacts/doc_embeddings.npz` (float16) that both the evaluation and the API use.
- **What I looked at on test before deciding.** The default system's choice of *no* ranker API was made
  after the test rows of `v6_ltr` and `v6_ltr_noapi` had been computed (RECSYS.md section 4). The
  decision rule was dev CV gain against cost and latency; test agreed in sign with dev and did not change
  it. No system was modified after looking at its test numbers.
- Plugging in a further system works as before: a module in `backend/app/recsys/systems/` with
  `@register_system(name, description=..., uses_llm=...)`; `config()` must return everything that
  determines the output, including model ids; never read `persona.latent` or the pool-only
  contributors; train on `split == "dev"` only.
