/**
 * Plain-language lines for the per-stage explanation of a recommended route
 * (backend: recsys.explanation/v1 and v1.1, which adds the optional `needs` stage; docs/SERVING.md
 * "Explanations").
 *
 * Every sentence is built from a field of the explanation; `numbers` holds the raw values for the
 * "show the numbers" view, so the modal is readable without being a wall of figures.
 */
import type { ApiRouteExplanation, ApiServedBy } from "./api-types";

export type StageTone = "good" | "neutral" | "warn";

export interface ExplanationLine {
  key: string;
  title: string;
  text: string;
  tone: StageTone;
  numbers: [string, string][];
}

export function ordinal(n: number): string {
  const s = n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] || "th";
  return `${n}${s}`;
}

function km(x: number): string {
  return x < 10 ? x.toFixed(1) : Math.round(x).toString();
}

const TAB_LABEL: Record<string, string> = {
  "Hiking trail": "hiking trail",
  "Theme trail": "theme trail",
  Cycling: "cycling route",
  Mountainbiking: "mountain-bike route",
  "Long distance cycling": "long-distance cycling route",
  Jogging: "running route",
  "Trail running": "trail-running route",
};

/** What a learned-ranker feature means to a hiker, phrased for a positive (+) or negative (-) push. */
const FEATURE_TEXT: Record<string, [string, string]> = {
  length_cap_ratio: ["its length suits your distance limit", "its length is far from what suits your limit"],
  length_band_pos: ["its length sits well in your usual range", "its length is outside your usual range"],
  length_km: ["its length", "its length"],
  intent_km_gap: ["its length is close to the distance you mentioned", "its length differs from the distance you mentioned"],
  loop_match: ["it is a loop, as you asked", "it is not the shape you asked for"],
  is_loop: ["it is a loop", "it is not a loop"],
  intent_loop: ["you asked for a loop", "you asked for a loop"],
  fused_rank: ["it is high in the combined text and meaning search", "it is lower in the combined search"],
  rrf: ["it scores high in the combined search", "it scores lower in the combined search"],
  bm25: ["your words appear in its description", "few of your words appear in its description"],
  bm25_rank: ["it ranks well on your words", "it ranks lower on your words"],
  dense: ["it is close in meaning to what you wrote", "it is less close in meaning to what you wrote"],
  dense_rank: ["it ranks well on meaning", "it ranks lower on meaning"],
  tag_cover: ["it has the features you asked for", "it misses some features you asked for"],
  tag_hits: ["it is tagged with what you asked for", "fewer of your wishes are tagged on it"],
  neg_hits: ["it avoids what you said no to", "it has something you said no to"],
  pois_per_km: ["plenty to see along the way", "little to see along the way"],
  popularity: ["it is a well-known route", "it is a less-known route"],
  paved_share: ["its surface mix suits the request", "its surface mix suits the request less"],
  busy_road_share: ["little busy road", "some busy road sections"],
  ascent_m: ["its climbing suits you", "its climbing is a lot for you"],
  ascent_per_km: ["its steepness suits you", "it is steep for you"],
  ascent_fit: ["its ascent fits your level", "its ascent is above your level"],
  difficulty: ["its difficulty suits you", "its difficulty suits you less"],
  difficulty_in_range: ["its difficulty is in your range", "its difficulty is outside your range"],
  technical: ["no technical sections", "technical sections"],
  intent_easy: ["you asked for something easy", "you asked for something easy"],
  duration_h: ["its duration suits you", "its duration suits you less"],
  home_km: ["it starts close to home", "it starts further from home"],
  home_ratio: ["it starts well inside your radius", "it starts near the edge of your radius"],
  rule_difficulty: ["difficulty matches your profile", "difficulty differs from your profile"],
  rule_distance: ["length matches your profile", "length differs from your profile"],
  rule_tags: ["shares your interests", "shares few of your interests"],
  waymarked: ["it is waymarked", "it is not waymarked"],
  osm_description: ["it has a mapper's description", "it has no mapper's description"],
  is_mtb: ["it is a mountain-bike route", "it is a mountain-bike route"],
  intent_mtb: ["you asked for mountain biking", "you asked for mountain biking"],
  rank_api: ["the semantic ranker likes it", "the semantic ranker likes it less"],
};

export function featureText(name: string, contribution: number): string {
  const t = FEATURE_TEXT[name];
  if (!t) return name.replace(/_/g, " ");
  return contribution >= 0 ? t[0] : t[1];
}

const REASON_LABEL: Record<string, string> = {
  "too-hard": "too difficult",
  "too-easy": "too easy",
  "too-far": "too far",
  "not-interested": "not interested",
  completed: "completed",
};

/** A matched search term as the user wrote it (stems such as "hik" map back to "hiking"). */
export function displayTerm(term: string, intent?: string | null): string {
  const words = (intent || "").toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  if (words.includes(term)) return term;
  return words.find((w) => w.startsWith(term)) || term;
}

export function explanationLines(
  e: ApiRouteExplanation,
  homeName?: string | null,
  intent?: string | null,
): ExplanationLine[] {
  const st = e.stages;
  const lines: ExplanationLine[] = [];
  const home = homeName ? homeName : "home";

  if (e.exploration?.is_exploration) {
    const d = e.exploration.detail || {};
    lines.push({
      key: "explore",
      title: "Something new",
      text: `An exploration pick: shown to learn your taste${d.label ? ` (${String(d.label)})` : ""}, not because it is your closest match.`,
      tone: "neutral",
      numbers: Object.entries(d).map(([k, v]) => [k, JSON.stringify(v)]),
    });
  }

  if (st.filter) {
    const f = st.filter;
    const parts: string[] = [];
    const nums: [string, string][] = [["verdict", f.verdict]];
    parts.push(
      f.category.allowed
        ? `It is a ${TAB_LABEL[f.category.value] || f.category.value}, as the tab asks.`
        : `It is a ${TAB_LABEL[f.category.value] || f.category.value} (all activities shown).`,
    );
    if (f.radius.applied && f.radius.distance_km !== null && f.radius.limit_km !== null) {
      if (f.radius.ok) {
        parts.push(`${km(f.radius.distance_km)} km from ${home}, inside your ${km(f.radius.limit_km)} km radius.`);
      } else if (f.radius.relaxed) {
        parts.push(
          `${km(f.radius.distance_km)} km from ${home}: beyond your ${km(f.radius.limit_km)} km radius, because too few routes are closer (searched up to ${km(f.radius.limit_used_km || 0)} km).`,
        );
      }
      nums.push(["distance from home", `${f.radius.distance_km} km`], ["your radius", `${f.radius.limit_km} km`]);
      if (f.radius.limit_used_km && f.radius.limit_used_km !== f.radius.limit_km) {
        nums.push(["radius used", `${f.radius.limit_used_km} km`]);
      }
    } else {
      parts.push("No home set, so distance from home was not checked.");
    }
    if (f.distance_cap.applied && f.distance_cap.limit_km !== null) {
      parts.push(`${km(f.distance_cap.length_km)} km long, within your ${km(f.distance_cap.limit_km)} km limit.`);
      nums.push(["length", `${f.distance_cap.length_km} km`], ["your limit", `${f.distance_cap.limit_km} km`]);
    } else {
      nums.push(["length", `${f.distance_cap.length_km} km`], ["your limit", "none set"]);
    }
    lines.push({
      key: "filter",
      title: "Fits your limits",
      text: parts.join(" "),
      tone: f.verdict === "pass" ? "good" : "warn",
      numbers: nums,
    });
  }

  if (st.needs && st.needs.checks.length) {
    const n = st.needs;
    const parts = n.checks.map((c) => {
      const asked = c.source === "slot" ? `Your profile asks for ${c.label}` : `You asked for ${c.label}`;
      if (c.ok) return `${asked}: ${c.fact}.`;
      if (!c.applied) return `${asked}, but too few routes nearby meet it, so it was relaxed: ${c.fact}.`;
      return `${asked}; this route meets it only partly: ${c.fact}.`;
    });
    const moved =
      n.rank_before != null && n.rank_after != null && n.rank_before !== n.rank_after
        ? ` Moved from ${ordinal(n.rank_before)} to ${ordinal(n.rank_after)} by how well it meets these.`
        : "";
    lines.push({
      key: "needs",
      title: "What you asked for",
      text: parts.join(" ") + moved,
      tone: n.unmet === 0 ? "good" : "warn",
      numbers: n.checks.flatMap((c) => [
        [`${c.need}: rule`, c.threshold] as [string, string],
        [`${c.need}: level met / enforced`, `${c.met_level ?? "none"} / ${c.applied ? c.applied_level : "relaxed away"}`] as [
          string,
          string,
        ],
        ...Object.entries(c.values).map(([k, v]) => [`${c.need}: ${k}`, JSON.stringify(v)] as [string, string]),
      ]),
    });
  }

  if (st.bm25) {
    const terms = [...new Set(st.bm25.matched_terms.map((t) => displayTerm(t.term, intent)))];
    lines.push({
      key: "bm25",
      title: "Matches your words",
      text: terms.length ? `Matches: ${terms.slice(0, 5).join(", ")}.` : "None of your words appear in its text.",
      tone: terms.length ? "good" : "neutral",
      numbers: [
        ["text score (BM25)", st.bm25.score.toFixed(2)],
        ["text rank", `${st.bm25.rank}${e.n_eligible ? ` of ${e.n_eligible}` : ""}`],
        ...st.bm25.matched_terms.map((t) => [`"${t.term}"`, `+${t.contribution.toFixed(2)}`] as [string, string]),
      ],
    });
  }

  if (st.dense) {
    lines.push({
      key: "dense",
      title: "Close in meaning",
      text: `The ${ordinal(st.dense.rank)} closest in meaning to what you wrote${e.n_eligible ? `, of ${e.n_eligible} routes that fit your limits` : ""}.`,
      tone: st.dense.rank <= 20 ? "good" : "neutral",
      numbers: [
        ["similarity (cosine)", st.dense.similarity.toFixed(3)],
        ["meaning rank", String(st.dense.rank)],
      ],
    });
  }

  if (st.fusion) {
    lines.push({
      key: "fusion",
      title: "Both searches combined",
      text: `${ordinal(st.fusion.rank)} when the word search and the meaning search are combined.`,
      tone: "neutral",
      numbers: [
        ["combined rank", String(st.fusion.rank)],
        ["RRF score", st.fusion.rrf_score.toFixed(4)],
        ["k", String(st.fusion.k)],
        ...Object.entries(st.fusion.inputs).map(([k, v]) => [`${k} rank`, String(v)] as [string, string]),
      ],
    });
  }

  if (st.rank_api) {
    lines.push({
      key: "rank_api",
      title: "Semantic ranker",
      text: `The semantic ranker placed it ${ordinal(st.rank_api.rank)}.`,
      tone: "neutral",
      numbers: [["score", st.rank_api.score.toFixed(3)], ["model", st.rank_api.model]],
    });
  }

  if (st.ltr) {
    const top = st.ltr.features.slice(0, 3);
    const pros = top.filter((f) => f.contribution > 0).map((f) => featureText(f.name, f.contribution));
    const cons = top.filter((f) => f.contribution < 0).map((f) => featureText(f.name, f.contribution));
    let why = "";
    if (pros.length) why += ` Mostly because ${pros.join(", and ")}.`;
    if (cons.length) why += ` Held back a little because ${cons.join(", and ")}.`;
    lines.push({
      key: "ltr",
      title: "How it ranked",
      text: `Ranked ${ordinal(st.ltr.rank)} of ${e.n_candidates ?? "the"} candidates before diversity.${why}`,
      tone: st.ltr.rank <= 10 ? "good" : "neutral",
      numbers: [
        ["ranker score", st.ltr.score.toFixed(3)],
        ["start value", st.ltr.bias.toFixed(3)],
        ...st.ltr.features.slice(0, 8).map(
          (f) =>
            [
              `${f.name}${f.value === null ? " (missing)" : ` = ${Number(f.value.toFixed(3))}`}`,
              `${f.contribution >= 0 ? "+" : ""}${f.contribution.toFixed(3)}`,
            ] as [string, string],
        ),
      ],
    });
  }

  if (st.v0_rules) {
    const r = st.v0_rules;
    const fits: string[] = [];
    if (Math.abs(r.difficulty - 0.4) < 1e-6) fits.push("difficulty");
    if (Math.abs(r.distance - 0.3) < 1e-6) fits.push("length");
    lines.push({
      key: "v0_rules",
      title: "Matches your profile",
      text:
        (fits.length ? `Its ${fits.join(" and ")} fit${fits.length === 1 ? "s" : ""} your questionnaire answers.` : "Partly matches your questionnaire answers.") +
        " (Tell us what you are looking for to search by your own words.)",
      tone: fits.length ? "good" : "neutral",
      numbers: [
        ["difficulty (of 0.4)", r.difficulty.toFixed(2)],
        ["length (of 0.3)", r.distance.toFixed(2)],
        ["interests (of 0.3)", r.tags.toFixed(2)],
        ["total", r.total.toFixed(3)],
        ...(r.feedback_penalty != null ? ([["feedback penalty", `×${r.feedback_penalty}`]] as [string, string][]) : []),
      ],
    });
  }

  if (st.mmr) {
    const m = st.mmr;
    const moved = m.rank_before - m.rank_after;
    lines.push({
      key: "mmr",
      title: "Keeping the list varied",
      text:
        moved > 0
          ? `Moved up from ${ordinal(m.rank_before)} to ${ordinal(m.rank_after)}: it differs from the routes above it.`
          : moved < 0
            ? `Moved down from ${ordinal(m.rank_before)} to ${ordinal(m.rank_after)}: it is similar to a route above it.`
            : `Stayed ${ordinal(m.rank_after)}.`,
      tone: "neutral",
      numbers: [
        ["lambda", String(m.lambda)],
        ["relevance", m.relevance.toFixed(3)],
        ["max similarity to routes above", m.max_similarity.toFixed(3)],
        ["penalty", m.penalty.toFixed(3)],
        ["rank before → after", `${m.rank_before} → ${m.rank_after}`],
      ],
    });
  }

  const fb = e.feedback;
  if (fb && (fb.events_considered > 0 || st.feedback)) {
    const parts: string[] = [];
    if (fb.on_this_route.length) {
      parts.push(
        `You gave feedback on this route before (${fb.on_this_route
          .map((x) => `${REASON_LABEL[x.reason] || x.reason}, ${Math.round(x.age_days)} days ago`)
          .join("; ")}).`,
      );
    }
    if (fb.hidden_routes > 0) {
      parts.push(
        `${fb.hidden_routes} route${fb.hidden_routes === 1 ? " you rejected or completed recently is" : "s you rejected or completed recently are"} left out of this list.`,
      );
    }
    if (fb.rule_penalty != null) parts.push(`Its score was reduced to ${Math.round(fb.rule_penalty * 100)}% by your feedback.`);
    if (!parts.length) parts.push(`Your ${fb.events_considered} feedback event${fb.events_considered === 1 ? "" : "s"} did not touch this route.`);
    lines.push({
      key: "feedback",
      title: "Your feedback",
      text: parts.join(" "),
      tone: "neutral",
      numbers: [
        ["feedback events", String(fb.events_considered)],
        ["hidden routes", String(fb.hidden_routes)],
        ...(fb.window_days ? ([["hidden for", `${fb.window_days} days`]] as [string, string][]) : []),
        ...Object.entries(st.feedback || fb.stage || {}).map(([k, v]) => [k, JSON.stringify(v)] as [string, string]),
      ],
    });
  }
  return lines;
}

const SERVED_REASON: Record<string, string> = {
  no_intent: "ranked by your questionnaire answers (tell us in your own words what you want for a closer match)",
  provider_unavailable: "the language service is unavailable, so routes are matched by your words only",
  timeout: "the language service was slow, so routes are matched by your words only",
  daily_budget: "today's demo budget is used up, so routes are matched by your words only",
  serving_budget: "the demo budget is used up, so routes are matched by your words only",
  rate_limited: "many requests in a short time, so routes are matched by your words only",
  catalogue_mismatch: "the database holds another catalogue, so the simple rule scorer ranked these",
  pipeline_not_loaded: "the recommender is not loaded, so the simple rule scorer ranked these",
};

/** A short note on how the list was produced, or null when the configured system served normally. */
export function servedNote(s?: ApiServedBy | null): string | null {
  if (!s || s.path === "random") return null;
  if (!s.reason || s.reason === "configured") return null;
  return SERVED_REASON[s.reason] || s.reason;
}
