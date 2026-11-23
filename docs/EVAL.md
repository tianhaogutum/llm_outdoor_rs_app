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
system (registry: random, v0_rules, v1_filters ... v7_mmr, recsys_v1, recsys_serve, ...)  ->  eval/runs/<system>.jsonl (top 50 + trace)
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
| `make eval SYSTEM=<name> CROSSFIT=1` | dev personas, learned ranker cross-fitted (section 13): the dev standard for systems with a learned ranker |
| `make eval SYSTEM=<name> HOLDOUT=1` | the 150 holdout personas (section 14), own qrels; reporting only since the D28 decision |
| `make holdout-personas` / `make holdout-decision` | draw the holdout (done once) / re-apply the D28 rule to the stored results |
| `make eval-all` | rescore every stored run against the current qrels (tables stay mutually consistent) |
| `make report` | render `docs/RESULTS.md` from `eval/results/*.json` |
| `make judge-pilot` | judge pilot (cached) |
| `make owner-set` / `make kappa` | blind labelling set + interim second judge / agreement report |
| `make freeze` / `make replay-check` | export the frozen LLM cache / prove a credential-free replay |
| `make simulate` / `make sim-tune` | Phase 5 feedback simulation on test / parameter grid on dev (section 10) |
| `make chat-eval` | scripted chat slot evaluation, replayed from `eval/chat/cache/` (section 11) |
| `make ltr-robust` | the default ranker retrained without the simulator-shaped features (section 11) |

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

**Split.** 200 dev / 100 test, by persona, fixed by the seed. Phase 3 trains only on dev. A third set, the
**holdout** (150 personas, same generator, another seed), was drawn in iteration 3 after its use was registered
(section 14).

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
an artificially low ideal and its recall would be undefined. With the contributors, every persona has at
least one known relevant (grade ≥ 2) pair (current counts: RESULTS.md "Data and judgements").

**Bias that remains.** (1) The relevant set is dominated by one contributor. At the end of Phase 2 (only
`random` and `v0_rules` pooled besides the contributors) `pool_latent` had found 2,147 of the 2,707 relevant
pairs, 1,701 of them found by no other run (`pool_intent_lexical` 951 / 509 unique; `random` 26, `v0_rules` 34).
Later systems added relevant pairs of their own, but the latent matcher's share is still large. The ideal DCG therefore leans towards what a latent-profile
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
held for 5,598 of the 11,347 pairs pooled in Phase 2 (49.3%; radius 4,836, radius + cap 699, cap 63);
current figures are in RESULTS.md "Data and judgements". The same predicates drive the violation-rate metric and
the Phase 3a filter.

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

**Card version (D29).** The first card (`card/v1`, lines without a `card` key) printed the persona's travel
radius, comfortable band and distance cap rounded to whole kilometres (a 6.4 km cap read "no more than 6 km"),
while the rules use the exact values. `card/v2` (`"card": "card/v2"`, every judgement since D29) prints them as
stored and the distance from home to 0.1 km, so no route the rules admit is printed above a hard limit. Pairs whose
grade the rounding could have changed (`python -m evalkit.rejudge estimate`: printed length above the printed cap,
a different position relative to the comfortable band, or printed distance above the printed radius) were
re-judged with `card/v2` in every qrels file; the new line keeps the old judgement under `previous` and
`make replay-check` replays both. A control sample of unaffected pairs re-judged the same way measures how much
re-judging alone changes (`eval/results/_rejudge.json`).

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
- `make freeze` re-issues, from the local cache, every call behind the committed artefacts (the 300
  intents, every judge batch recorded in the qrels and the simulation qrels, the pilot, the second judge, the
  systems' query embeddings and ranker calls, the ranker pilot, the MMR sweep) and exports exactly those
  entries to `eval/cache/` (its `manifest-*.json` has the count; at the end of Phase 2 it was 1,484 entries,
  300 KB). The harness adds `eval/cache/` as a frozen layer unless `LLM_CACHE_FROZEN_DIRS` is set.
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
  evaluation except the Phase 5 simulation (section 10), whose reactions come from a behaviour model
  written for this project.

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
  cross-validation on dev (`_ltr.json`), never its system-level dev row and never test. Since iteration 3 the
  system-level dev row has a cross-fitted counterpart (section 13), which is the dev number to use.
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

## 10. Phase 5: the feedback simulation

`make simulate` (`backend/evalkit/simulate.py`); results `eval/results/_simulation.json`, lists and events
`eval/runs/simulation.jsonl`, figure `docs/convergence.png`, tables in RESULTS.md "Feedback loop". The feedback
design itself is in [RECSYS.md](RECSYS.md) section 10; decisions D20-D23.

**Read every number here with two layers of simulation in mind:** the users are simulated (section 2) *and*
their reactions come from a behaviour model written for this project (below). A feedback loop tuned or
judged on it is tuned and judged on these assumptions; it says nothing about how real people react to a list.

**Rounds.** 100 test personas x 10 rounds, one week apart. In round t every system gets
`persona.to_query()` plus the persona's events from rounds 1..t-1 as `FeedbackEvent`s with age
7 x (t - s) days, and returns its top 10. Systems compared: `v0_rules` with its own mechanism (it sees only
the four negative reasons: the product has no completion signal), `v1_filters` (filter + rule scorer,
same mechanism, so the comparison is not just "one system knows where the user lives"), `recsys_v1` with
feedback off (control: the same list every round), `recsys_v1_fb_route` (route effects only: recent
completions and rejections are not shown again), `recsys_v1_fb` (the loop as chosen on dev),
`recsys_v1_fb_light` (the full loop at the grid's light setting) and `recsys_v1_fb_explore` (`recsys_v1_fb`
plus the exploration slot). All run in lock-step, so every round's holes are judged together.

**Judging.** Every shown route is judged before the user reacts, so every round is 100% judged:
hard-constraint violations by rule, the rest by the production judge, batched per persona over all
systems of the round. These judgements go to `eval/qrels/sim_qrels.jsonl` (same record format, replayed by
`make replay-check`), not into the pooled `qrels.jsonl`: the single-request tables keep their top-10 pool
and their numbers. The per-round metric reads both files. Volume was estimated first
(`python -m evalkit.simulate estimate`, which proxies unjudged grades from the latent utility and writes
nothing). Because a round's batches are formed from the holes of all systems of that round, the grade of
a pair can depend on which other systems were simulated (the batch-order effect of section 8).

**The behaviour model** (`BehaviourModel`, never visible to a system). For one shown list, in rank order:

- the user looks at rank i with probability 0.95^(i-1), and skips routes they already completed or
  rejected (no repeat outing within the simulation, no second thumbs-down);
- a looked-at route is completed with probability 0, 0.03, 0.25, 0.5 for judged grade 0, 1, 2, 3; at most
  one completion per round (one outing a week);
- otherwise it is rejected with probability 0.5 / 0.2 for grade 0 / 1 (never for 2 or 3); at most three
  rejections per round;
- the reason comes from the **latent profile vs the route's facts**, first that applies: start beyond the
  latent travel radius -> `too-far`; wrong category, or MTB vs non-MTB mismatch for a cyclist ->
  `not-interested`; length above 1.2 x the top of the comfortable band, ascent above 1.2 x the ascent
  tolerance, or a technical grade above the tolerance -> `too-hard`; length below 0.7 x the band's lower end
  with ascent below half the tolerance -> `too-easy`; else `not-interested`;
- every random draw is a hash of (seed, persona, round, route, purpose): the same route at the same round
  gets the same draw for every system (common random numbers), and a rerun is identical.

Its limits, plainly: the probabilities are invented, not measured; completion depends only on the judge's
grade, so the simulated user agrees with the judge by construction (a system that pleases the judge is
rewarded twice); the user's preferences never change, so a decay half-life cannot be tuned on it (a
longer memory always wins); rejection reasons are clean and always given; position bias is a single
geometric decay; no session effects, no fatigue, no seasons. A loop tuned on this model is tuned on these
assumptions.

**Per-round metric.** NDCG@10 of the round's list where routes the user completed or rejected in an
**earlier** round have gain 0 *and* are removed from the ideal ranking. So a system is not punished for no
longer showing what the user already did (its ideal shrinks with it), and gains nothing by showing it again
(a re-shown completed route is a wasted slot). Everything else uses its judged grade. The metric is
computed after the last round against every judgement made in the run, so a round's ideal does not
depend on which routes happened to be judged by then (and a replay reproduces it). Also reported per
round: coverage@10 over the 100 lists and cumulative coverage (distinct routes shown so far), completion
counts and events by reason. CIs: bootstrap over personas; paired differences per persona over the mean
of rounds 2-10 (round 1 has no feedback) and for round 10 alone.

**Tuning on dev.** `make sim-tune` runs the same simulation on 40 seeded dev personas for a grid of
feedback parameters (`TUNING`: route effects only; slot steps only; Rocchio only; re-scoring only; the full
loop light / medium / strong) and picks the highest mean NDCG@10 over rounds 2-10. Test personas are never
used for a choice. Caveat: the learned ranker is in-sample on dev, so dev lists start close to the ideal and
leave feedback less to repair than on test; the grid therefore leans towards light feedback.

## 11. Phase 4 and robustness rows in the harness

- **Chat slot evaluation** moved into the harness layout (history kept with `git mv`): cases
  `eval/chat/utterances.jsonl`, frozen cache `eval/chat/cache/<candidate>/`, runner and scorer
  `backend/evalkit/chat/`, results `eval/results/chat/<candidate>.json` and the summary
  `eval/results/_chat_eval.json`. `make chat-eval` replays it without any provider call; `make report`
  renders slot precision/recall per slot type, clarifying-decision accuracy and schema-violation rate;
  `make replay-check` re-runs it and compares. Protocol and caveats: CHAT.md section 8.
- **End-to-end comparison** (test split, three registered systems, 100% judged): `recsys_v1_qonly`
  (questionnaire only: the intent is dropped, so the pipeline takes its no-intent path), `recsys_v1`
  (questionnaire + the intent as typed) and `recsys_v1_chat` (questionnaire + chat: the session starts from
  the form's slots, the intent is sent as one chat message through the real turn logic, and the pipeline
  gets the merged slots, the projected vector and the extractor's restated intent). The chat model is
  pinned to the D18 choice in the harness (`evaluate.chat_client`), whatever credentials are configured,
  so a replay asks the frozen cache for the same calls. What it can and cannot show: RESULTS.md section
  header. The personas' intents were written by an LLM from the same latent profile the judge reads, so
  free text here carries exactly the information the judge grades against.
- **Robustness rows** `recsys_v1_nocap` and `recsys_v1_nosusp` (Part D): the default with rankers retrained
  (`make ltr-robust`, same pinned dev labels, same CV) without `length_cap_ratio` or without the suspicious
  group; evaluated like every system on all personas. D19 has the recommendation.
- **Served default and empty slots** (D25): `recsys_serve` is evaluated like every system (all personas,
  100% judged) and in the simulation, next to its no-exploration twin `recsys_serve_noexplore`.
  `recsys_serve_nocap`, `recsys_serve_nohome`, `recsys_serve_noslots` and `recsys_v1_nocapslot` remove the
  cap, the home (and with it the radius), or both from the request before the system sees it; relevance is
  still judged against the persona's real limits, so routes beyond them are graded 0 by rule. They are checks,
  not product variants.


## 12. Iteration 2: an improvement driven by the failure analysis

**Disclosure.** The hypothesis of iteration 2 (stated, verifiable needs should be constraints; RECSYS.md section 13)
came from reading the ten worst *test* queries of the served system ([FAILURES.md](FAILURES.md)). That is the only
thing test contributed: which needs, the trigger phrases, the thresholds, the relaxation order, the floor and the
choice between a filter and a demotion were developed on dev intents, dev judgements and general language
knowledge (English and German). Test was run once per registered variant, after the decision, for the tables.

**Protocol**, in the order it was carried out (each step committed before the next):

1. The same worst-queries analysis on dev (`python -m evalkit.failures --split dev -n 20`,
   `eval/results/_failures_recsys_serve_dev.json`), classified by hand: class B (lengths and day counts) replicates,
   class A (a stated need overridden by the ranker) barely does, because the ranker was trained on these very
   personas (D27 has the table).
2. The extractor measured on all dev intents against the latent profiles (`python -m evalkit.needs_eval
   extractor`): precision and recall per need where a latent field corresponds; how often the served top 10 meets
   a stated need. **The latent profile is used here only to measure the extractor; no system reads it.**
3. The acceptance rule written down (D27, `evalkit/needs_eval.py: RULE`) before any dev NDCG of a variant existed:
   a paired-bootstrap CI of the dev NDCG@10 difference against `recsys_serve` above zero, no rise in violations,
   under 20 ms of stage compute at p95.
4. The three variants on dev (`python -m evalkit.needs_eval dev`): every top 10 judged (dev holes only), paired
   bootstrap against the stored `recsys_serve` run. Decision by the rule.
5. A post-hoc check, labelled as such and not used for the decision: the same comparison with a cross-fitted
   ranker (`python -m evalkit.needs_eval crossfit`; five copies of the served ranker trained like it on four fifths
   of the dev personas, each persona ranked by the copy that did not see it; LightGBM needed, holes judged on dev).
6. `make eval SYSTEM=<variant>` for each variant (both splits, 100% judged), `make eval-all`, the failure analysis
   of the variant chosen for it before the run (`python -m evalkit.failures --system recsys_serve_needs_promote`),
   `make report`.

**What the in-sample ranker does to the comparison.** The served ranker was trained on the dev personas' judged
candidates, so on dev it reproduces the judge's preferences for exactly these users, including the minority users
whose needs it overrides on test. Any change that overrides the ranker therefore looks worse on dev than it is: the
baseline is optimistic and the failures the change targets are hidden. Iteration 2 measured the size of this:
against the in-sample ranker every variant loses on dev, against the cross-fitted ranker every variant wins
(RESULTS.md "Iteration 2"). In the other direction, the thresholds were set by reading dev intents and
judgements, which favours the stage on dev. A rule on system-level dev rows of a system containing the learned
ranker should use a cross-fitted ranker; this one did not, and its decision stands as written (D27).

**Limits specific to this iteration.** Dev intents are English and clean (all extracted needs with a latent
counterpart have precision 1.0 there), so the German triggers and the rarer phrasings rest on unit tests, not on
measured data. `no_steps` and `public_transport` never occur in dev intents. The judge reads the latent profile,
and the intents were written from it, so a stated need that matches the latent profile is graded as a need; real
users may state needs they would trade away. Results files: `eval/results/_needs_dev.json` (dev work),
`eval/results/recsys_serve_needs*.json` (rows), `_failures_recsys_serve_needs_promote_test.json`.


## 13. Cross-fitted dev evaluation: the dev standard for systems with a learned ranker (iteration 3)

**Why.** Every learned ranker in this project is trained on the dev personas' judgements (RECSYS.md section 4). A
system-level dev row of any system that contains one is therefore in-sample, and the bias is large: about 0.15
NDCG@10 for the served system (RESULTS.md "Dev with a cross-fitted learned ranker"). It is worse for comparisons
than for single rows. A change that keeps the ranker's ordering inherits the memorised dev users; a change that
overrides the ranker (a filter, a demotion, feedback) gives that advantage up, so dev systematically favours "no
change". This decided D21 and D27 the wrong way round relative to test.

**What `make eval SYSTEM=<name> CROSSFIT=1` does** (`backend/evalkit/crossfit.py`, `python -m evalkit.evaluate
--system <name> --crossfit`):

1. Finds the learned-ranker model in the system's config (`ltr.name`, also inside the chat and empty-slot
   wrappers); a system without one is refused, because its dev row is not in-sample.
2. Trains K = 5 copies of that model exactly like the committed one: the same pinned labels
   (`eval/qrels/ltr_train_v1.jsonl`), the same feature set (its training variant), the same LightGBM parameters
   and the same number of boosting rounds, each on the dev personas outside one fold. The folds are the ranker's
   own grouped-CV folds (`evalkit.ltr._folds`: by persona, seeded). The copies are written once to
   `eval/crossfit/ltr_<model>_cf<k>.json.gz` (numpy-evaluable trees, a reference block checked on load), so
   rescoring and replay never need LightGBM. Training them does; the Makefile sets the macOS OpenMP path.
3. Runs the system on the 200 dev personas, each with the copy whose training set excluded it. Nothing else in
   the system changes.
4. Pools and judges the new top-10 holes like any dev hole (into the main qrels; dev pairs only), and writes
   `eval/runs/crossfit/<name>.jsonl` and `eval/results/crossfit/<name>.json` (dev metrics only; the results file
   carries `bench.label` "dev, cross-fitted ranker" and the fold files' digests in `config.crossfit`).

`make eval-all` rescores the stored cross-fitted runs with the others; `make replay-check` re-runs them from the
committed fold copies and the frozen cache and compares rankings and results; `make report` shows the cross-fitted
dev NDCG next to the in-sample one (column in "All systems, dev split", and the section "Dev with a cross-fitted
learned ranker" with paired differences on cross-fitted dev next to the same differences in-sample and on test).

**Check.** It reproduces the iteration-2 post-hoc check exactly: the same fold models, and for `recsys_serve` and
the three stated-need variants the same per-persona NDCG@10 on all 200 dev personas, hence the same paired
differences (D27). The post-hoc command `python -m evalkit.needs_eval crossfit` now uses these fold copies.

**The standard from now on.** For any system with a learned ranker, a dev number used for a decision is the
cross-fitted one; the in-sample row is shown, labelled "in-sample", and is never the basis of a choice.
Comparisons between such systems on dev are paired over the dev personas on cross-fitted rows (a system without a
learned ranker enters with its ordinary dev row, which is not in-sample).

**What it does not fix.** (1) One hyper-parameter leaks: the number of rounds was chosen by grouped CV over all
dev personas, so each copy inherits a choice that saw its held-out fold (probably small: the mean CV curves in
`_ltr.json` and `_ltr_robust.json` move by a few thousandths between 200 rounds and the chosen number). (2) Everything else tuned on dev (thresholds, the stated-need rules, MMR lambda, RRF k) is still
tuned on dev; cross-fitting removes only the ranker's memory of the dev users. (3) The copies are trained on four
fifths of the data, so cross-fitted dev is slightly pessimistic for the ranker itself. (4) It is not a substitute
for a split nobody has looked at, which is why iteration 3 decides on a fresh holdout (D28).

## 14. The holdout (iteration 3)

**Why.** By iteration 3 both splits had been used for development: dev for every choice and all training, test to
find the failure pattern behind iteration 2 and to report its variants. Neither could decide cleanly between the
served default and those variants. A fresh set of simulated users, drawn after the decision rule was committed,
can (DECISIONS.md D28; the commit order in `git log` is the evidence).

**Generation** (`python -m evalkit.holdout generate`, `make holdout-personas`; refuses to run if the file exists):
the generator of section 2 with exactly the same parameters (`evalkit.personas.PARAMS`) and seed 20261128, all
personas in split `holdout`, ids `H001`... = draw index. A draw whose latent profile or whose visible profile slots
(questionnaire vector, tab, home, radius, cap) equal those of a main persona or an earlier draw is skipped (none
were); identical intent texts are reported, not removed (one, a short MTB request, equals a dev persona's).
Intents by the same writer, prompt and temperature (section 2). Files: `eval/personas/holdout.jsonl`,
`holdout_params.json` (seed, parameters, duplicate report), `holdout_stats.json`.

**Judging.** Its own qrels file `eval/qrels/holdout_qrels.jsonl`, so the main tables and the ranker's training
data are untouched. Pool: the top 10 of the four D28 candidates plus the two pool-only contributors; every top 10
judged by the production judge; hard-constraint violations graded 0 by rule. Runs and results:
`eval/runs/holdout/`, `eval/results/holdout/`; the decision: `eval/results/_holdout.json`
(`evalkit/holdout.py: decide`, rewritten by `make eval-all`, checked by `make replay-check`, which also
regenerates the holdout personas and replays its judge batches from the frozen cache).

**The rule** (D28, committed before the personas existed): a cross-fitted dev gate (section 13), then one-sided
simultaneous 95% lower bounds of the paired holdout NDCG@10 differences for the gated variants together
(single-step max-T over a joint paired bootstrap, `metrics.simultaneous_lower_bounds`), a violations guard, the
largest eligible bound wins. Outcome and numbers: RESULTS.md "Iteration 3", D28.

**Use from now on.** Reporting only: the holdout table in RESULTS.md and the failure analysis of the served
system (`make failures`, FAILURES.md). Nothing is to be tuned or chosen on it; a system added to it later pools
new holes into its qrels, which shifts the ideal DCG of the other rows (rescore with `make eval-all`; the D28
decision is recorded in DECISIONS.md with the qrels digest it was taken on).

**Limits.** The same simulator and judge as everything else, so the holdout guards against overfitting to the dev
and test *users*, not against the simulator's or the judge's assumptions. 150 personas give a standard error of
under one point of NDCG@10 for a paired difference; small true effects can be missed. The judge's card rounded the
distance cap to whole kilometres while the rule-grader uses the exact cap (found in the holdout failure analysis);
fixed in D29 (card v2, section 4) and the affected pairs of every qrels file, the holdout's included, re-judged.
The D28 decision was re-applied to the corrected holdout qrels and is unchanged.
