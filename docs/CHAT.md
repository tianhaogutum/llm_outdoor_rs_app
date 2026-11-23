# Conversational preference elicitation (Phase 4, first part)

A chat that turns what a person writes ("easy loop near Tegernsee with a hut, max 12 km") into the
profile's structured slots plus a free-text intent, asks at most two clarifying questions, and writes
the result to the stored profile so that any recommender behind `GET /api/routes/recommendations` uses
it. Related: [BUILD_SPEC.md](BUILD_SPEC.md) Phase 4, [LLM_CLIENT.md](LLM_CLIENT.md) (the only LLM path),
[DATA.md](DATA.md) section 7 (the tag vocabulary).

Design rule: **the model proposes, code decides.** The model only fills a structured schema; validation,
merging, the turn policy, place resolution and the assistant's reply text are deterministic code.

## 1. Layout

```
backend/app/chat/
  vocab.py       activities, bands and their km / difficulty ranges, the tag vocabulary (= pipeline.tags.TAG_NAMES)
  schema.py      ChatExtraction (wire, lenient), SlotUpdate (strict), ChatSlots, ChatState, API models
  extract.py     system prompt, few-shot examples, sanitizer, extract_slots() (no database)
  merge.py       apply_update(): the merge rules (pure)
  engine.py      ChatSession: extract -> resolve place -> merge -> turn policy -> reply
  projection.py  slots -> user vector (incremental) and -> recsys.query.UserQuery
  places.py      place resolvers (DB table or in-memory from the catalogue file) + exonyms
  rules.py       keyword extractor: the mock provider's answer offline, and a baseline row
  llm.py         the chat's own LLMClient (model, location, thinking, budget)
backend/app/services/chat_service.py   load/store a ChatSession on a DemoProfile; user_query_for_profile()
backend/app/api/v1/chat.py             the endpoints
backend/alembic/versions/a7c3e9d2b5f4  demo_profiles.activity, demo_profiles.chat_state_json
backend/chat_eval/                     scripted set, runner, scorer, results, frozen cache
frontend/components/preference-chat.tsx  chat box + chips (onboarding and results page)
```

## 2. API

The app mounts its API at `/api` (not `/api/v1` as the spec text says).

| Method | Path | Body | What it does |
|---|---|---|---|
| POST | `/api/profiles/{id}/chat` | `{"message": "..."}` (1–1000 chars) | one user message in, one assistant turn out |
| GET | `/api/profiles/{id}/chat` | – | current conversation, slots and intent |
| PATCH | `/api/profiles/{id}/chat/slots` | a `SlotUpdate` (strict fields) | the UI chips: same merge rules, no LLM |
| DELETE | `/api/profiles/{id}/chat` | – | new conversation (history, clarify count, defaults reset; slots kept) |

Response (`ChatResponse`): `status` (`clarify` / `ready` / `idle`), `reply` (written by code, in the
user's language), `question` (`{slot, text, options}` or null), `changes` (slot by slot: `{slot, old,
new, source}` with source `chat`, `edit` or `default`), `dropped` (proposed values the validator
rejected, with a reason), `slots` (the current slots plus `category`, `distance_range_km`,
`difficulty_range`, read-only `questionnaire_tags`, and the `sources`), `intent`, `clarify_turns_used` /
`clarify_turns_max`, `defaults_applied`, `turns` (stored history), `model`, `cached`.

Errors: 404 unknown profile, 422 invalid body, 502 the model's answer violated the schema after the
client's re-asks, 503 budget exhausted or provider unreachable (504 on a timeout). A failed turn changes
nothing on the server.

## 3. Slots, required slots, defaults, turn policy

| Slot | Values | Where it lands on the profile |
|---|---|---|
| `activity` | hiking, cycling, mtb, running | `demo_profiles.activity`; UI tab / category filter (mtb -> cycling tab) |
| `difficulty` | easy, moderate, hard (= the questionnaire's beginner/intermediate/advanced ranges [0,1], [1,2], [2,3]) | vector `difficulty_range` |
| `distance_band` | short, medium, long (hiking = questionnaire ranges 0–8, 5–20, 10–50 km; running ×1, MTB ×2.5, cycling ×4) | vector `min/max_distance_km` |
| `duration_hours` | 0.25–48 | mapped to a band with planning speeds 4 / 9 / 16 / 10 km/h (hike / run / bike / MTB) |
| `distance_cap_km` | 1–400 | `distance_cap_km` (hard filter) and clips the vector's max |
| `place` | a catalogue locality, coordinates from the `places` table | `home_name/home_lat/home_lon` |
| `travel_radius_km` | 1–300 | `travel_radius_km` (hard filter, with a place) |
| `tags_preferred` / `tags_avoided` | the 50 catalogue tags only | vector `preferred_tags` (appended) / new key `avoided_tags` |
| `intent` | ≤ 300 chars, one sentence | `intent_text` (later: BM25 query and embedding) |

**Required slots: `activity` and `place`.** These are the two things the hard filters cannot do without:
the category filter needs the activity (a route list mixing walks and 80 km rides is wrong for
everyone), and the radius filter needs a centre. Everything else has a sensible default that already
exists: difficulty and distance come from the questionnaire, tags are optional, no cap means no cap.

**Turn policy.** After merging a message: if a required slot is missing and fewer than two clarifying
questions were asked in this conversation, ask exactly one question, about the first missing slot
(activity, then place), with quick replies for the activity. Otherwise the turn is `ready`. If required
slots are still missing then, the documented defaults are applied once and recorded in
`defaults_applied`: **activity -> hiking** (82% of the catalogue), **place -> none** (no radius filter;
the whole catalogue is eligible). A defaulted slot is not asked for again in the conversation; the user
can set it any time. A place the lookup cannot resolve ("Atlantis", "Wien") is reported in `dropped` and
the reply, and counts as missing.

**Default radius.** When a message sets a new place and no radius is known or stated, the radius is set
to **50 km** with source `default` (shown as "Within 50 km (default)", removable).

## 4. Merge rules (`app/chat/merge.py`, tested in `tests/chat/test_merge.py`)

Null fields never change anything. Applied in this order:

1. `clear`: listed slots become empty; clearing `place` also clears the radius; `tags` empties both lists.
   A cleared difficulty or distance band falls back to the questionnaire's value (the pre-chat vector).
2. `activity` replaces.
3. Difficulty: an absolute band replaces; else `difficulty_change` (easier/harder) moves one band,
   clamped; with no current band, easier = easy, harder = hard.
4. Length band, first that applies: absolute band; `distance_target_km` ("about 10 km") mapped for the
   activity; `duration_hours` × planning speed, mapped; `distance_change` (shorter/longer) one band,
   clamped (none known: shorter = short, longer = long). A stated duration is stored; a length
   statement without a duration clears an older one.
5. `distance_cap_km` replaces. 6. `place` replaces (+ the default radius, above). 7. `travel_radius_km` replaces.
8. Tags have three states (wanted / neutral / avoided): `tags_remove` -> neutral ("no lakes after all"),
   `tags_avoid` -> avoided ("no summit"), `tags_add` -> wanted. A tag both added and avoided in one
   message is dropped as ambiguous.
9. `intent`: a non-empty restatement replaces the stored one. The model sees the stored intent and is
   asked to carry it forward with the corrections applied.

The vector is updated incrementally: only keys behind slots that changed are rewritten, so feedback
adjustments on other keys survive. `v0_rules` reads `difficulty_range`, distances and `preferred_tags`
(lowercased on both sides), so chat tags now reach the baseline's tag score; it ignores `avoided_tags`
and the slot columns.

**For the offline harness:** `app.chat.extract_slots(utterance)` (no database; place stays a name),
`app.chat.run_conversation(messages, resolver=IndexPlaceResolver.from_catalog(...), vector=...)`,
`ChatSession.user_query(request_id=...)` / `app.chat.user_query_from_slots(slots, vector, ...)`, and
`app.services.chat_service.user_query_for_profile(profile)` for a stored profile.

## 5. Untrusted text and validation

- The user message and the current slots (they contain user-derived text) go into the prompt through
  `wrap_untrusted`; `UNTRUSTED_DATA_RULE` and an explicit rule against following, revealing or
  paraphrasing instructions are in the system prompt. A fake closing delimiter in the message is
  neutralised by the wrapper.
- Two schemas. `ChatExtraction` (what the model fills) advertises the enums in its JSON schema and
  marks every field required-nullable, but validates values leniently; a structural failure is a schema
  violation (counted in the ledger, re-asked by the client). `sanitize()` then checks every field on its
  own and builds the strict `SlotUpdate`: enums, numeric ranges (above), vocabulary membership
  (case-insensitive canonicalisation), list lengths (≤ 8), place-name plausibility (letters, ≤ 6 words,
  no digits), intent length. Failing fields are dropped with a reason; the rest of the answer survives.
  Unknown keys (`home_lat`, `total_xp`, `user_vector_json`) are ignored by the wire schema and forbidden
  by the strict one, so nothing but the schema's fields can reach the profile.
- `SlotUpdate` carries the same validators, so the chip endpoint cannot bypass them (422).
- The only model-written free text that reaches the user or the profile is the intent: markup characters
  stripped, ≤ 300 chars, rejected if it contains the data delimiter or shares 8 consecutive words with
  the system prompt (a rejected intent is not echoed back). Place names reach the profile only via the
  catalogue lookup; coordinates never come from the model. The assistant's reply is a template.
- Vertex finding: a schema with two 50-value tag enums is rejected with `400 INVALID_ARGUMENT`; only
  `tags_add` carries the enum, the avoid/remove lists are plain strings checked by code. With optional
  fields Gemini omitted most of them; marking all fields required fixed it.

## 6. Frontend

`components/preference-chat.tsx`: message log, input, the clarifying question with "question n of 2"
and quick-reply buttons, then "What I will use": chips for every slot (enum chips cycle through their
values on click, × removes; defaults are dashed and labelled; wanted tags `+ Lake`, avoided `no Summit
route`), the questionnaire's tags as read-only chips, the intent ("Searching for: …", removable), and
anything ignored with its reason. "new conversation" calls DELETE.

- Onboarding: after the three questions and the optional details step, the summary screen (the profile
  now exists) shows the chat as an optional "tell me more in your own words" step. The questionnaire
  path is unchanged; the chat's slots are stored in the local profile on "start exploring".
- Results page: a "Refine your recommendations" box above the activity tabs for logged-in profiles.
  Every turn or chip edit that changed something refreshes the list; if the activity changed, the
  matching tab is selected (which refetches). Without a backend profile the box is hidden.

## 7. Model choice (scripted set below; nominal cost/latency as recorded)

| Candidate | Overall P / R / F1 | enum P/R | number P/R | place P/R | tags P/R | clarify acc. | schema viol. (failed calls) | p50 / p95 ms | USD / turn |
|---|---|---|---|---|---|---|---|---|---|
| `gemini-2.5-flash`, europe-west4, thinking off | 0.984 / 0.977 / 0.980 | 0.981 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.976 / 0.932 | 1.000 | 0 (0) | 849 / 1385 | 0.00132 |
| `gemini-3.5-flash`, global, thinking minimal | 0.985 / 1.000 / 0.992 | 0.981 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.978 / 1.000 | 0.972 | 0 (0) | 1499 / 3025 | 0.00766 |
| `gemini-3.5-flash-lite`, global, thinking minimal | 0.961 / 0.961 / 0.961 | 0.944 / 1.000 | 0.889 / 0.941 | 1.000 / 1.000 | 1.000 / 0.909 | 0.986 | 0 (0) | 1013 / 1764 | 0.00170 |
| **`anthropic/claude-haiku-5.5` via OpenRouter, reasoning off** | 0.970 / 1.000 / 0.985 | 0.962 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.957 / 1.000 | 1.000 | 0 (0) | 1123 / 2081 | 0.00072 |
| keyword rules (offline mock, not an LLM) | 0.931 / 0.844 / 0.885 | 0.957 / 0.863 | 0.857 / 0.706 | 1.000 / 0.938 | 0.902 / 0.841 | 0.958 | 0 (0) | – | 0 |

72 cases, 84 extraction calls per candidate; intent-mention hit rate 1.0 for every candidate (6 cases).
Default: **Claude Haiku 5.5 via OpenRouter** (`CHAT_DEFAULTS` in `app/chat/llm.py`; Vertex
`gemini-3.5-flash` on `global` when no OpenRouter key is configured). Reasons: the four LLMs differ by 1–4
errors on 72 cases, which this set cannot separate; Haiku is about a tenth of `gemini-3.5-flash`'s cost
per turn with lower latency; `gemini-2.5-flash` is past its retirement date; Flash-Lite was the weakest.
Accepted cost: user messages (which may name a home town) leave the EU with every candidate except the
retired `gemini-2.5-flash`, and OpenRouter adds a processor. Proposed as a DECISIONS entry.

## 8. Scripted evaluation

`python -m chat_eval.run --all [--cache-only] [--freeze]` (from `backend/`). Cases:
`chat_eval/data/utterances.jsonl`, 72 cases / 84 user turns: single slot (10), multi slot (10), vague
(8, incl. a two-turn clarification and the default fallback after two questions), corrections over two
turns (8), negations (6), number forms (8), family/stroller/dog/wheelchair (5), place typos and an
English exonym (6), out-of-vocabulary wishes (5), injection attempts (6); 46 English and 26 German cases (none mix languages within one message: a gap).
Each case has a starting profile (`start`), the expected *changes* after the last turn (`expect`),
optional acceptable alternatives (`optional`), the expected clarifying question (`clarify`) and, for
out-of-vocabulary wishes, words the intent must mention.

**Matching rules (fixed before the first run, `chat_eval/scoring.py`):** only changes made by the chat
count (defaults excluded). Enums exact; place = same catalogue locality after both names go through the
same resolver; `duration_hours` ±25% or ±0.5 h, `distance_cap_km` ±10% or ±1 km, `travel_radius_km` ±25%
or ±5 km (whichever larger); tags as set precision/recall over the changed tags. A wrong value counts
as FN and FP (only FN if it is a listed alternative). Micro-averaged per slot type and overall. Clarify
*decision accuracy* = asked vs not asked; *slot accuracy* also needs the right slot. Schema-violation
rate = invalid answers / provider attempts; validator drops are reported separately.

Determinism: the touched cache is exported to `chat_eval/cache/<candidate>/`; a replay with
`--cache-only` and an empty local cache reproduces every results file except its `volatile` block
(checked by diffing two replays). Results: `chat_eval/results/<candidate>.json`, `summary.json`.

**Worst misses (all LLMs unless named).**
- Injected JSON with in-range values passes: `{"activity": "running", …} easy loop please` set the activity
  to running for every model (the 5000 km radius, `home_lat` and the made-up tag were dropped).
- "Ohne Hund unterwegs" ("without the dog") became *avoid Dog-friendly* for every model.
- "reachable by train" -> `Train access` instead of the expected `Public transport access` (2.5 Flash, Lite).
- `gemini-3.5-flash`: "egal wie weit" ("however far") cleared the place, not only the radius, so it asked
  for the place again; "family day out with kids" was read as hiking, so no activity question (also Lite).
- Flash-Lite: "hmm, make that 12" after "max 20 km" kept 20 and set a length band instead.
- Haiku: "15 km max, with a view" also added `Mostly unpaved`; "make that 12" also set a band.

**Read the numbers with this in mind.** The utterances, the expected slots, the prompt and the keyword
extractor were all written by the same agent, in one sitting, with no second vendor or human writing
or labelling cases. The cases therefore sit inside the agent's own idea of what users say, phrased
the way its prompt anticipates; the high F1 (0.96–0.99) measures agreement between the agent's prompt
and the agent's labels, not robustness to real users. The keyword baseline's 0.89 shows the effect most
clearly: its rules were written next to the cases. Expect real messages (longer, vaguer, dialect,
multi-intent, off-topic) to score clearly lower; 72 cases also cannot separate models that differ by a
few errors. Not tuned on: the prompt was not changed after the first run.

## 9. Limits and open points

- Regions are not places: "Allgäu", "Fränkische Schweiz" stay in the intent only; the place list is the
  catalogue's localities (3,591, Bavaria).
- Chip edits change slots but not the intent text, which can then disagree (e.g. "easy" left in the intent
  after the difficulty chip was cleared); the next chat turn restates it.
- The reply template shows enum values in English inside German replies ("Aktivität hiking").
- Injected but valid values (above) can only affect the user's own profile; there is no cross-user path.
- `v0_rules` ignores place, radius, cap and avoided tags; they take effect with the new pipeline's filters.
- The end-to-end NDCG@10 comparison (questionnaire only vs + chat) is a later step.
