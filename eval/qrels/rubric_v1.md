# Relevance rubric v1

You grade how well ONE route fits ONE person, on a 0-3 scale. You see the person's full profile
(activity, home, party, fitness, comfortable distance, climbing and technical level, wishes,
dislikes), the text they typed into the app, and a factual card for each route (including its
straight-line distance from their home). Grade each route on its own; the other routes in the same
request are not a reference point.

## Grades

- **3 - fits fully.** Right activity; length inside or very close to the comfortable range; climbing
  and technical level within what they are comfortable with; suitable for everyone in the party;
  offers the person's main wishes (all of them, or nearly all when they have several).
- **2 - fits the main intent, at most a minor mismatch.** For example: one main wish missing while
  the others are met; somewhat longer or shorter than comfortable but still reasonable; climbing a
  little above their comfort; the shape (loop vs one-way) or surface is not the preferred one.
- **1 - feasible and the right activity, but misses the point or has a clear mismatch.** For
  example: none of the main wishes; clearly too long, too short or too much climbing for this
  person; poor fit for the party (rough surface for a stroller, long road sections with small
  children, a train user whose start has no station).
- **0 - unsuitable or irrelevant.** Wrong activity (a touring cyclist on a technical MTB trail, a
  mountain biker on a city cycleway with nothing they asked for), unsuitable or unsafe for this
  party (technical or alpine terrain beyond their level, a big climb with toddlers), or nothing to
  do with the request.

Pairs that break one of the person's hard limits (wrong category tab, start beyond their travel
radius, longer than their absolute maximum) are graded 0 by rule and are never shown to you.

## How to read the data

- Wishes marked "main wish" matter most; "nice to have" items only separate 2 from 3 at the margin.
  Do not reward a route for features the person did not ask for.
- What the person typed is what they want today; when it adds or stresses something, take it into
  account. The profile's limits still apply.
- Features are found by proximity to the route line: an inn 70 m from the line counts as "near the
  line", and a restaurant in a village the route passes is likely but not guaranteed to be open.
  "Refreshment stops available" is on two thirds of all routes for that reason.
- Ascent comes from a 30 m surface model: it overestimates flat routes (lake shores, forests), so a
  lake loop showing 200-300 m of ascent is probably nearly flat. Summit heights are cut a little.
- The start of a loop is where the mapped relation begins, not necessarily a trailhead or car park.
- Missing features are not proof of absence (mapping is incomplete), but do not assume features
  that are not listed.
- Estimated moving time follows DIN 33466 (4 km/h walking, no breaks); families and older walkers
  are slower.

## Output

For every route, give a short reason (one sentence, at most 25 words, naming the deciding facts)
and then the grade.
