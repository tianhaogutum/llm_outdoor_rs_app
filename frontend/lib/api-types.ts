/**
 * TypeScript types for TrailSaga – Hogwarts Expedition Series Backend API responses
 * These match the Pydantic schemas defined in backend/app/api/schemas.py
 */

export interface ApiBreakpoint {
  id: number;
  route_id: number;
  order_index: number;
  poi_name: string | null;
  poi_type: string | null;
  latitude: number | null;
  longitude: number | null;
  main_quest_snippet: string | null;
  at_m?: number | null; // position along the route in metres (OSM breakpoints)
  mini_quests: ApiMiniQuest[];
}

export interface ApiMiniQuest {
  id: number;
  breakpoint_id: number;
  task_description: string;
  xp_reward: number;
}

export interface RecommendationScoreBreakdown {
  difficulty: {
    score: number;
    weight: number;
    weighted_score: number;
    user_range: [number, number];
    route_value: number;
  };
  distance: {
    score: number;
    weight: number;
    weighted_score: number;
    user_range: [number, number];
    route_value: number;
  };
  tags: {
    score: number;
    weight: number;
    weighted_score: number;
    user_tags: string[];
    route_tags: string[];
  };
  total: number;
  // Feedback-related fields (optional, added by backend when feedback exists)
  feedback_adjusted?: boolean;
  base_score?: number;
  final_score?: number;
  feedback_penalty?: number;
  feedback_count?: number;
}

/** A named OSM feature on the route (lake, hut, castle, ...) */
export interface ApiHighlight {
  type: string;
  name: string;
  at_m: number;
}

export interface ApiAccessPoint {
  name?: string | null;
  dist_m: number;
  fee?: string;
  capacity?: number;
}

/** Compact display facts of an OSM route (routes.facts_json); every key is optional */
export interface ApiRouteFacts {
  highlights?: ApiHighlight[];
  access?: {
    train_station?: ApiAccessPoint;
    tram_stop?: ApiAccessPoint;
    bus_stop?: ApiAccessPoint;
    parking?: ApiAccessPoint;
  };
  end_access?: {
    train_station?: ApiAccessPoint;
    tram_stop?: ApiAccessPoint;
    bus_stop?: ApiAccessPoint;
  };
  surface?: {
    paved_share?: number;
    unpaved_share?: number;
    path_share?: number;
    track_share?: number;
    road_share?: number;
  };
  min_ele_m?: number;
  max_ele_m?: number;
  max_grade?: string;
  est_days?: number;
  end_locality?: string;
  protected_areas?: string[];
  description_source?: "generated" | "osm+generated" | string;
  ref?: string;
  network?: string;
  operator?: string;
  website?: string;
}

export interface ApiRoute {
  id: number;
  title: string;
  category_name: string | null;
  length_meters: number | null;
  duration_min: number | null;
  difficulty: number | null; // 0-3 scale
  short_description: string | null;
  location: string | null;
  elevation: number | null;
  tags_json: string | null; // JSON array of tags
  xp_required: number; // XP needed to unlock this route
  base_xp_reward: number; // XP earned from completing this route (Base XP)
  story_prologue_title: string | null;
  story_prologue_body: string | null;
  story_epilogue_body: string | null;
  gpx_data_raw: string | null; // GPX track data for map visualization
  breakpoints: ApiBreakpoint[];
  is_locked: boolean; // Computed field based on user XP
  recommendation_score?: number | null; // CBF score (0.0-1.0)
  recommendation_score_breakdown?: RecommendationScoreBreakdown | null; // Score components
  // Catalogue facts (OSM routes; null for the legacy catalogue)
  source?: string | null; // "osm"
  source_url?: string | null; // OSM relation page
  activity?: string | null; // hiking | cycling | mtb | running
  start_lat?: number | null;
  start_lon?: number | null;
  is_loop?: boolean | null;
  ascent_m?: number | null; // == elevation (total ascent)
  descent_m?: number | null;
  locality?: string | null;
  admin_district?: string | null;
  admin_region?: string | null;
  facts?: ApiRouteFacts | null;
  attribution?: string | null; // must be shown with OSM data (ODbL)
}

export interface ApiRouteGeometry {
  route_id: number;
  coords: [number, number][]; // [lon, lat]
  bbox: [number, number, number, number];
  ele_profile_m?: number[] | null;
  ele_step_m?: number | null;
  length_m?: number | null;
  source?: string | null;
  source_url?: string | null;
  attribution?: string | null;
}

export interface ApiPlace {
  id: number;
  name: string;
  admin_district: string | null;
  admin_region: string | null;
  lat: number;
  lon: number;
  accuracy_m: number | null;
  route_count: number;
  match: "exact" | "prefix" | "word" | "fuzzy";
  score: number;
}

export interface ApiPlaceSearchResponse {
  places: ApiPlace[];
  total: number;
}

export interface ApiHomeLocation {
  name: string;
  lat?: number | null;
  lon?: number | null;
}

export interface ApiRecommendationResponse {
  routes: ApiRoute[];
  total: number;
  is_personalized: boolean;
}

export interface ApiProfile {
  id: number;
  total_xp: number;
  level: number;
  user_vector_json: string | null;
  genai_welcome_summary: string | null;
  unlocked_routes_json: string | null;
  home?: { name: string; lat: number; lon: number } | null;
  travel_radius_km?: number | null;
  distance_cap_km?: number | null;
  intent?: string | null;
  activity?: ChatActivity | null;
}

// ---------------------------------------------------------------------------
// Chat preference elicitation (backend/app/chat/schema.py, docs/CHAT.md)
// ---------------------------------------------------------------------------

export type ChatActivity = "hiking" | "cycling" | "mtb" | "running";
export type ChatDifficulty = "easy" | "moderate" | "hard";
export type ChatDistanceBand = "short" | "medium" | "long";
export type ChatSlotName =
  | "activity"
  | "difficulty"
  | "distance_band"
  | "duration_hours"
  | "distance_cap_km"
  | "place"
  | "travel_radius_km"
  | "tags_preferred"
  | "tags_avoided"
  | "intent";
export type ChatClearable =
  | "activity"
  | "difficulty"
  | "distance_band"
  | "duration_hours"
  | "distance_cap_km"
  | "place"
  | "travel_radius_km"
  | "tags"
  | "intent";

export interface ApiChatSlots {
  activity: ChatActivity | null;
  difficulty: ChatDifficulty | null;
  distance_band: ChatDistanceBand | null;
  duration_hours: number | null;
  distance_cap_km: number | null;
  place: { name: string; lat: number; lon: number; admin_district?: string | null } | null;
  travel_radius_km: number | null;
  tags_preferred: string[];
  tags_avoided: string[];
  intent: string | null;
  category: "hiking" | "cycling" | "running" | null; // UI tab that follows from the activity
  distance_range_km: [number, number] | null;
  difficulty_range: [number, number] | null;
  questionnaire_tags: string[];
  sources: Record<string, "chat" | "edit" | "default" | "profile">;
}

export interface ApiChatChange {
  slot: ChatSlotName;
  old: unknown;
  new: unknown;
  source: "chat" | "edit" | "default";
}

export interface ApiChatDropped {
  field: string;
  reason: string;
  value?: string | null;
}

export interface ApiChatTurn {
  role: "user" | "assistant";
  text: string;
  at?: string | null;
  question_slot?: "activity" | "place" | null;
}

export interface ApiChatResponse {
  profile_id: number;
  status: "clarify" | "ready" | "idle";
  reply: string;
  question: { slot: "activity" | "place"; text: string; options: string[] } | null;
  changes: ApiChatChange[];
  dropped: ApiChatDropped[];
  slots: ApiChatSlots;
  intent: string | null;
  clarify_turns_used: number;
  clarify_turns_max: number;
  defaults_applied: string[];
  turns: ApiChatTurn[];
  model?: string | null;
  cached?: boolean | null;
}

/** PATCH /api/profiles/{id}/chat/slots: the chips (same strict fields as a chat turn, no LLM) */
export interface ApiChatSlotEdit {
  activity?: ChatActivity;
  difficulty?: ChatDifficulty;
  distance_band?: ChatDistanceBand;
  distance_cap_km?: number;
  travel_radius_km?: number;
  place_name?: string;
  tags_add?: string[];
  tags_avoid?: string[];
  tags_remove?: string[];
  clear?: ChatClearable[];
}

export interface ApiProfileCreate {
  fitness: string; // "beginner" | "intermediate" | "advanced"
  type: string[]; // ["history-culture", "natural-scenery", "family-fun"]
  narrative: string; // "adventure" | "mystery" | "playful"
  // Optional slots (hard filters / retrieval); omit for "no constraint"
  home?: ApiHomeLocation | null;
  travel_radius_km?: number | null;
  distance_cap_km?: number | null;
  intent?: string | null;
}

export interface ApiProfileCreateResponse {
  id: number;
  welcome_summary: string;
  user_vector: Record<string, any>;
  profile?: ApiProfile | null;
}

export interface ApiSouvenir {
  id: number;
  demo_profile_id: number;
  route_id: number;
  completed_at: string; // ISO datetime string
  total_xp_gained: number;
  genai_summary: string | null;
  xp_breakdown_json: string | null; // JSON string containing XP breakdown details
  pixel_image_svg: string | null; // LLM-generated pixel art SVG
  route?: ApiRoute;
  unlocked_achievements?: ApiAchievement[]; // Achievements unlocked when completing this route
}

export interface RouteCompleteRequest {
  route_id: number;
  completed_quest_ids: number[];
}

export interface RouteCompleteResponse {
  souvenir: ApiSouvenir;
  xp_breakdown: {
    base: number;
    quests: number;
    difficulty_multiplier: number;
    total: number;
  };
  total_xp_gained: number; // XP earned from this route completion
  new_total_xp: number; // New total XP after this completion
  new_level: number;
  newly_unlocked_achievements?: Array<{
    id: number;
    achievement_key: string;
    name: string;
    description: string;
    icon: string;
  }>;
}

export interface SouvenirListResponse {
  souvenirs: ApiSouvenir[];
  total: number;
}

export interface ApiProfileStatistics {
  total_distance_km: number;
  total_elevation_m: number;
  routes_completed: number;
  achievements_unlocked: number;
  activity_breakdown: Record<string, number>;
}

export interface FeedbackCreate {
  route_id: number;
  reason: string; // too-hard, too-easy, too-far, not-interested
}

export interface FeedbackResponse {
  id: number;
  demo_profile_id: number;
  route_id: number;
  reason: string;
}

/**
 * Achievement types
 */
export interface ApiAchievement {
  id: number;
  achievement_key: string;
  name: string;
  description: string;
  icon: string;
  condition_type: string;
  condition_value: string;
  unlocked?: boolean;
  unlocked_at?: string | null;
}

export interface ApiProfileAchievement {
  id: number;
  achievement_key: string;
  name: string;
  description: string;
  icon: string;
  unlocked: boolean;
  unlocked_at: string | null;
}

/**
 * XP Breakdown structure (stored as JSON string in xp_breakdown_json)
 */
export interface XpBreakdown {
  base_xp: {
    difficulty: number;
    distance: number;
    duration: number;
    elevation: number;
    total: number;
  };
  mini_quests: {
    completed_count: number;
    total_xp: number;
    details: Array<{
      quest_id: number;
      description: string;
      xp: number;
    }>;
  };
  total_xp_gained: number;
}

/**
 * Helper function to map API difficulty (0-3) to frontend difficulty labels
 * 0 -> easy, 1 -> moderate, 2 -> difficult, 3 -> expert
 */
export function mapApiDifficulty(
  difficulty: number | null
): "easy" | "medium" | "hard" | "expert" {
  if (difficulty === null || difficulty === 0) return "easy";
  if (difficulty === 1) return "medium"; // "moderate" displayed as "medium" in UI
  if (difficulty === 2) return "hard"; // "difficult" displayed as "hard" in UI
  return "expert";
}

/**
 * Helper function to map API category_name to frontend route type
 * Maps backend category_name values to frontend categories based on CATEGORY_MAPPING
 */
export function mapApiCategory(
  categoryName: string | null
): "hiking" | "running" | "cycling" {
  if (!categoryName) return "hiking";

  const lower = categoryName.toLowerCase();

  // Running: "Jogging", "Trail running"
  if (lower.includes("run") || lower.includes("jogging")) return "running";

  // Cycling: "Cycling", "Mountainbiking", "Long distance cycling"
  if (
    lower.includes("cycling") ||
    lower.includes("mountain") ||
    lower.includes("bike")
  )
    return "cycling";

  // Hiking: "Theme trail", "Hiking trail", and everything else
  return "hiking";
}
