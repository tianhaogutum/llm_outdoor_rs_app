"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  questionnaireQuestions,
  UserProfile,
  defaultUserProfile,
} from "@/lib/mock-data";
import { apiClient } from "@/lib/api-client";
import { ApiChatResponse, ApiPlace, ApiProfileCreate, ApiProfileCreateResponse } from "@/lib/api-types";
import { PreferenceChat } from "@/components/preference-chat";
import { ArrowRight, ArrowLeft, Sparkles, MapPin } from "lucide-react";

// Optional last step: slots used by hard filters (radius, distance cap) and retrieval (intent).
const RADIUS_OPTIONS: Array<{ value: number | null; label: string }> = [
  { value: null, label: "ANY" },
  { value: 10, label: "10 KM" },
  { value: 25, label: "25 KM" },
  { value: 50, label: "50 KM" },
  { value: 100, label: "100 KM" },
];
const DISTANCE_CAP_OPTIONS: Array<{ value: number | null; label: string }> = [
  { value: null, label: "ANY" },
  { value: 5, label: "5 KM" },
  { value: 10, label: "10 KM" },
  { value: 20, label: "20 KM" },
  { value: 40, label: "40 KM" },
  { value: 80, label: "80 KM" },
];
const INTENT_MAX_LENGTH = 500;

interface OnboardingQuestionnaireProps {
  onComplete: (profile: UserProfile) => void;
}

export function OnboardingQuestionnaire({
  onComplete,
}: OnboardingQuestionnaireProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<Record<string, any>>({});
  const [showSummary, setShowSummary] = useState(false);
  const [explorerType, setExplorerType] = useState("");
  const [isCreatingProfile, setIsCreatingProfile] = useState(false);
  const [backendProfileId, setBackendProfileId] = useState<number | null>(null);
  const [welcomeSummary, setWelcomeSummary] = useState<string | null>(null);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);

  // Step 4 (optional): home, travel radius, distance cap, free-text intent
  const [homeQuery, setHomeQuery] = useState("");
  const [home, setHome] = useState<ApiPlace | null>(null);
  const [placeSuggestions, setPlaceSuggestions] = useState<ApiPlace[]>([]);
  const [isSearchingPlaces, setIsSearchingPlaces] = useState(false);
  const [placeSearchError, setPlaceSearchError] = useState<string | null>(null);
  const [travelRadiusKm, setTravelRadiusKm] = useState<number | null>(null);
  const [distanceCapKm, setDistanceCapKm] = useState<number | null>(null);
  const [intent, setIntent] = useState("");
  const searchSeq = useRef(0);
  // Optional chat step after the questions (docs/CHAT.md); its slots overwrite the form's on completion
  const [chatState, setChatState] = useState<ApiChatResponse | null>(null);

  const totalSteps = questionnaireQuestions.length + 1;
  const isDetailsStep = currentQuestion === questionnaireQuestions.length;
  const progress = ((currentQuestion + 1) / totalSteps) * 100;

  // Debounced place autocomplete against the catalogue's own localities
  useEffect(() => {
    const q = homeQuery.trim();
    if (home && q === home.name) return; // the field shows the chosen place
    if (q.length < 2) {
      setPlaceSuggestions([]);
      setPlaceSearchError(null);
      return;
    }
    const seq = ++searchSeq.current;
    const timer = setTimeout(async () => {
      setIsSearchingPlaces(true);
      try {
        const response = await apiClient.searchPlaces(q, 8);
        if (seq === searchSeq.current) {
          setPlaceSuggestions(response.places);
          setPlaceSearchError(response.places.length === 0 ? "NO MATCHING PLACE IN THE CATALOGUE" : null);
        }
      } catch (error) {
        if (seq === searchSeq.current) {
          setPlaceSuggestions([]);
          setPlaceSearchError("PLACE SEARCH UNAVAILABLE");
        }
      } finally {
        if (seq === searchSeq.current) setIsSearchingPlaces(false);
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [homeQuery, home]);

  const choosePlace = (place: ApiPlace) => {
    setHome(place);
    setHomeQuery(place.name);
    setPlaceSuggestions([]);
    setPlaceSearchError(null);
  };

  const buildProfilePayload = (): ApiProfileCreate => {
    const payload: ApiProfileCreate = {
      fitness: answers.fitness,
      type: answers.type || [],
      narrative: answers.narrative,
    };
    if (home) payload.home = { name: home.name, lat: home.lat, lon: home.lon };
    // A radius only means something relative to a home
    if (travelRadiusKm !== null && home) payload.travel_radius_km = travelRadiusKm;
    if (distanceCapKm !== null) payload.distance_cap_km = distanceCapKm;
    const trimmedIntent = intent.trim();
    if (trimmedIntent) payload.intent = trimmedIntent;
    return payload;
  };

  const profileSlots = () => {
    const form = {
      home: home ? { name: home.name, lat: home.lat, lon: home.lon } : null,
      travelRadiusKm: home ? travelRadiusKm : null,
      distanceCapKm,
      intent: intent.trim() || null,
      activity: null as UserProfile["activity"],
    };
    if (!chatState) return form;
    const s = chatState.slots;
    return {
      home: s.place ? { name: s.place.name, lat: s.place.lat, lon: s.place.lon } : null,
      travelRadiusKm: s.travel_radius_km,
      distanceCapKm: s.distance_cap_km,
      intent: s.intent,
      activity: s.activity,
    };
  };

  const handleAnswer = (
    questionId: string,
    value: any,
    isMultiple: boolean
  ) => {
    if (isMultiple) {
      const current = answers[questionId] || [];
      const updated = current.includes(value)
        ? current.filter((v: any) => v !== value)
        : [...current, value];
      setAnswers({ ...answers, [questionId]: updated });
    } else {
      setAnswers({ ...answers, [questionId]: value });
    }
  };

  const handleNext = () => {
    if (currentQuestion < totalSteps - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      generateExplorerType();
    }
  };

  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const generateExplorerType = async () => {
    // Simple logic to generate explorer type based on answers
    const fitness = answers.fitness;
    const narrative = answers.narrative;

    let type = "";
    if (fitness === "beginner") {
      type = narrative === "history" ? "Urban Historian" : "Nature Novice";
    } else if (fitness === "intermediate") {
      type = narrative === "adventure" ? "Trail Adventurer" : "City Explorer";
    } else if (fitness === "advanced") {
      type = narrative === "adventure" ? "Mountain Challenger" : "Peak Seeker";
    } else {
      type = "Summit Conqueror";
    }

    setExplorerType(type);
    setShowSummary(true);

    // Fetch welcome summary from backend immediately
    setIsLoadingSummary(true);
    try {
      const response = await apiClient.post<ApiProfileCreateResponse>(
        "api/profiles",
        buildProfilePayload()
      );

      setBackendProfileId(response.id);
      setWelcomeSummary(response.welcome_summary);
    } catch (error) {
      console.error("Failed to fetch welcome summary:", error);
      // welcomeSummary remains null, fallback will be shown
    } finally {
      setIsLoadingSummary(false);
    }
  };

  const handleComplete = async () => {
    setIsCreatingProfile(true);

    try {
      // Ensure we have a backend profile ID
      let finalBackendProfileId = backendProfileId;

      // If profile wasn't created during generateExplorerType, create it now
      if (!finalBackendProfileId) {
        const response = await apiClient.post<ApiProfileCreateResponse>(
          "api/profiles",
          buildProfilePayload()
        );

        finalBackendProfileId = response.id;
        setBackendProfileId(response.id);
        setWelcomeSummary(response.welcome_summary);
      }

      // Create frontend profile with backend ID
      // IMPORTANT: Use finalBackendProfileId (from response or state) instead of relying on async state update
      const profile: UserProfile = {
        ...defaultUserProfile,
        id: finalBackendProfileId
          ? String(finalBackendProfileId)
          : defaultUserProfile.id,
        explorerType,
        fitnessLevel: answers.fitness,
        preferredTypes: answers.type || [],
        narrativeStyle: answers.narrative,
        ...profileSlots(),
      };

      console.log(
        "[Questionnaire] Profile created with ID:",
        profile.id,
        "Backend ID:",
        finalBackendProfileId
      );
      onComplete(profile);
    } catch (error) {
      console.error("Failed to complete profile creation:", error);

      // Fallback: create local-only profile with timestamp-based ID to avoid conflicts
      const localId = `local-${Date.now()}`;
      const profile: UserProfile = {
        ...defaultUserProfile,
        id: localId,
        explorerType,
        fitnessLevel: answers.fitness,
        preferredTypes: answers.type || [],
        narrativeStyle: answers.narrative,
        ...profileSlots(),
      };

      console.warn(
        "[Questionnaire] Using fallback local profile with ID:",
        localId
      );
      onComplete(profile);
    } finally {
      setIsCreatingProfile(false);
    }
  };

  const question = isDetailsStep ? null : questionnaireQuestions[currentQuestion];
  const isAnswered = !question
    ? true // the details step is optional
    : question.type === "multiple"
    ? answers[question.id]?.length > 0
    : answers[question.id];

  if (showSummary) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4 relative overflow-hidden">
        <div
          className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: `
            repeating-linear-gradient(0deg, transparent, transparent 8px, currentColor 8px, currentColor 9px),
            repeating-linear-gradient(90deg, transparent, transparent 8px, currentColor 8px, currentColor 9px)
          `,
          }}
        />

        <Card className="max-w-2xl w-full bg-card pixel-border relative z-10">
          <div className="bg-accent text-accent-foreground p-4 border-b-4 border-border">
            <h2 className="text-sm font-pixel text-pixel-shadow-sm text-center">
              CLASS ASSIGNED!
            </h2>
          </div>

          <div className="p-8 md:p-12 space-y-6 text-center">
            <div className="text-6xl mb-4 animate-bounce">🎖️</div>

            <div className="bg-primary text-primary-foreground p-4 pixel-border-sm">
              <h2 className="text-sm md:text-base font-pixel text-pixel-shadow-sm">
                {explorerType.toUpperCase()}
              </h2>
            </div>

            <div className="bg-muted p-6 pixel-border-sm">
              <p className="text-[10px] md:text-xs leading-relaxed text-foreground">
                {isLoadingSummary ? (
                  // Loading state
                  <span className="flex items-center justify-center gap-2">
                    <Sparkles className="w-4 h-4 animate-pulse" />
                    <span>GENERATING YOUR PERSONALIZED WELCOME...</span>
                  </span>
                ) : welcomeSummary ? (
                  // Display AI-generated welcome summary from backend
                  welcomeSummary
                ) : (
                  // Fallback message if backend generation failed
                  "YOUR ADVENTURE PROFILE HAS BEEN CREATED. ROUTES HAVE BEEN PERSONALIZED TO MATCH YOUR EXPLORER SPIRIT."
                )}
              </p>
            </div>

            <div className="bg-card border-4 border-border p-4 space-y-3 text-left">
              <h3 className="font-pixel text-xs text-center mb-4">STATS</h3>
              <div className="grid grid-cols-2 gap-4 text-[10px]">
                <div className="space-y-1">
                  <p className="text-muted-foreground">FITNESS</p>
                  <p className="font-pixel text-foreground uppercase">
                    {answers.fitness}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-muted-foreground">STYLE</p>
                  <p className="font-pixel text-foreground uppercase">
                    {answers.narrative}
                  </p>
                </div>
              </div>
              <div className="pt-2 border-t-2 border-dashed border-border space-y-1">
                <p className="text-muted-foreground text-[10px]">ACTIVITIES</p>
                <div className="flex flex-wrap gap-2">
                  {(answers.type || ["ALL"]).map((t: string) => (
                    <span
                      key={t}
                      className="px-2 py-1 bg-primary text-primary-foreground text-[8px] font-pixel"
                    >
                      {t.toUpperCase().replace("-", " ")}
                    </span>
                  ))}
                </div>
              </div>
              {(home || travelRadiusKm || distanceCapKm || intent.trim()) && (
                <div className="pt-2 border-t-2 border-dashed border-border grid grid-cols-2 gap-4 text-[10px]">
                  {home && (
                    <div className="space-y-1">
                      <p className="text-muted-foreground">HOME</p>
                      <p className="font-pixel text-foreground uppercase">{home.name}</p>
                    </div>
                  )}
                  {home && travelRadiusKm && (
                    <div className="space-y-1">
                      <p className="text-muted-foreground">TRAVEL RADIUS</p>
                      <p className="font-pixel text-foreground">{travelRadiusKm} KM</p>
                    </div>
                  )}
                  {distanceCapKm && (
                    <div className="space-y-1">
                      <p className="text-muted-foreground">MAX ROUTE LENGTH</p>
                      <p className="font-pixel text-foreground">{distanceCapKm} KM</p>
                    </div>
                  )}
                  {intent.trim() && (
                    <div className="space-y-1 col-span-2">
                      <p className="text-muted-foreground">LOOKING FOR</p>
                      <p className="text-foreground leading-relaxed">&ldquo;{intent.trim()}&rdquo;</p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {backendProfileId && (
              <PreferenceChat
                profileId={backendProfileId}
                variant="pixel"
                title="TELL ME MORE IN YOUR OWN WORDS (OPTIONAL)"
                placeholder="e.g. easy loop near Tegernsee with a hut, max 12 km"
                onUpdated={setChatState}
              />
            )}

            <button
              onClick={handleComplete}
              disabled={isCreatingProfile || isLoadingSummary}
              className="w-full px-6 py-4 bg-accent text-accent-foreground font-pixel text-xs pixel-button hover:bg-accent/90 pixel-blink disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isCreatingProfile
                ? "⏳ CREATING PROFILE..."
                : isLoadingSummary
                ? "⏳ LOADING..."
                : "▶ START EXPLORING"}
            </button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4 relative overflow-hidden">
      <div
        className="absolute inset-0 opacity-5"
        style={{
          backgroundImage: `
          repeating-linear-gradient(0deg, transparent, transparent 8px, currentColor 8px, currentColor 9px),
          repeating-linear-gradient(90deg, transparent, transparent 8px, currentColor 8px, currentColor 9px)
        `,
        }}
      />

      <Card className="max-w-2xl w-full bg-card pixel-border relative z-10">
        <div className="bg-primary text-primary-foreground p-4 border-b-4 border-border">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs md:text-sm font-pixel text-pixel-shadow-sm">
              CHARACTER CREATION
            </h2>
            <span className="text-xs font-pixel">
              Q {currentQuestion + 1}/{totalSteps}
            </span>
          </div>
          <div className="health-bar-container">
            <div
              className="health-bar-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="p-6 md:p-8 space-y-6">
          <div className="space-y-4">
            <div className="bg-muted p-4 pixel-border-sm">
              <p className="text-[10px] md:text-xs leading-relaxed text-foreground font-pixel">
                {question
                  ? question.question
                  : "Where do you start from, and what are you looking for? (optional)"}
              </p>
            </div>
          </div>

          {isDetailsStep && (
            <div className="space-y-5">
              {/* Home location with autocomplete */}
              <div className="space-y-2 relative">
                <label htmlFor="home-place" className="block text-[10px] font-pixel text-foreground">
                  🏠 HOME
                </label>
                <div className="flex items-center gap-2 bg-card pixel-border-sm px-3">
                  <MapPin className="w-4 h-4 text-muted-foreground shrink-0" />
                  <input
                    id="home-place"
                    type="text"
                    autoComplete="off"
                    value={homeQuery}
                    onChange={(e) => {
                      setHomeQuery(e.target.value);
                      if (home && e.target.value !== home.name) setHome(null);
                    }}
                    placeholder="Town or village in Bavaria, e.g. Garmisch"
                    className="w-full py-3 bg-transparent text-xs text-foreground outline-none placeholder:text-muted-foreground"
                    aria-autocomplete="list"
                    aria-controls="home-place-suggestions"
                  />
                  {isSearchingPlaces && <span className="text-[10px] animate-pulse">…</span>}
                  {home && <span className="text-lg" aria-label="place selected">✓</span>}
                </div>
                {placeSuggestions.length > 0 && (
                  <ul
                    id="home-place-suggestions"
                    role="listbox"
                    className="absolute z-20 left-0 right-0 bg-card pixel-border-sm max-h-64 overflow-y-auto"
                  >
                    {placeSuggestions.map((place) => (
                      <li key={place.id} role="option" aria-selected={home?.id === place.id}>
                        <button
                          type="button"
                          onClick={() => choosePlace(place)}
                          className="w-full text-left px-3 py-2 text-xs hover:bg-muted flex items-center justify-between gap-2"
                        >
                          <span className="text-foreground">
                            {place.name}
                            <span className="text-muted-foreground"> · {place.admin_district}</span>
                          </span>
                          <span className="text-[10px] text-muted-foreground shrink-0">
                            {place.route_count} {place.route_count === 1 ? "route" : "routes"}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                {placeSearchError && !home && (
                  <p className="text-[10px] text-muted-foreground font-pixel">{placeSearchError}</p>
                )}
                {!home && homeQuery.trim().length >= 2 && placeSuggestions.length > 0 && (
                  <p className="text-[10px] text-muted-foreground">Pick a place from the list to use it.</p>
                )}
              </div>

              {/* Travel radius */}
              <div className="space-y-2">
                <p className="text-[10px] font-pixel text-foreground">🚗 HOW FAR WILL YOU TRAVEL TO THE START?</p>
                <div className="flex flex-wrap gap-2">
                  {RADIUS_OPTIONS.map((opt) => (
                    <button
                      key={opt.label}
                      type="button"
                      onClick={() => setTravelRadiusKm(opt.value)}
                      disabled={opt.value !== null && !home}
                      title={opt.value !== null && !home ? "Choose a home first" : undefined}
                      className={`px-3 py-2 pixel-border-sm text-[10px] font-pixel disabled:opacity-40 disabled:cursor-not-allowed ${
                        travelRadiusKm === opt.value ? "bg-primary text-primary-foreground" : "bg-card hover:bg-muted"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Distance cap */}
              <div className="space-y-2">
                <p className="text-[10px] font-pixel text-foreground">📏 LONGEST ROUTE YOU WANT</p>
                <div className="flex flex-wrap gap-2">
                  {DISTANCE_CAP_OPTIONS.map((opt) => (
                    <button
                      key={opt.label}
                      type="button"
                      onClick={() => setDistanceCapKm(opt.value)}
                      className={`px-3 py-2 pixel-border-sm text-[10px] font-pixel ${
                        distanceCapKm === opt.value ? "bg-primary text-primary-foreground" : "bg-card hover:bg-muted"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Free-text intent */}
              <div className="space-y-2">
                <label htmlFor="intent" className="block text-[10px] font-pixel text-foreground">
                  💬 WHAT ARE YOU LOOKING FOR?
                </label>
                <textarea
                  id="intent"
                  value={intent}
                  maxLength={INTENT_MAX_LENGTH}
                  onChange={(e) => setIntent(e.target.value)}
                  rows={3}
                  placeholder="e.g. a quiet lake loop with a hut stop, fine for a seven-year-old"
                  className="w-full p-3 bg-card pixel-border-sm text-xs text-foreground outline-none placeholder:text-muted-foreground resize-none"
                />
                <p className="text-[10px] text-muted-foreground text-right">
                  {intent.length}/{INTENT_MAX_LENGTH}
                </p>
              </div>
            </div>
          )}

          <div className="space-y-3">
            {(question?.options ?? []).map((option) => {
              const isSelected =
                question!.type === "multiple"
                  ? (answers[question!.id] || []).includes(option.value)
                  : answers[question!.id] === option.value;

              return (
                <button
                  key={option.value}
                  onClick={() =>
                    handleAnswer(
                      question!.id,
                      option.value,
                      question!.type === "multiple"
                    )
                  }
                  className={`w-full p-4 text-left pixel-border-sm text-[10px] md:text-xs transition-all flex items-center gap-3 ${
                    isSelected
                      ? "bg-primary text-primary-foreground"
                      : "bg-card hover:bg-muted"
                  }`}
                >
                  <span className="text-2xl">{option.icon}</span>
                  <span className="flex-1 font-pixel">
                    {isSelected && <span className="mr-2">▶</span>}
                    {option.label.toUpperCase()}
                  </span>
                  {isSelected && <span className="text-lg">✓</span>}
                </button>
              );
            })}
          </div>

          <div className="flex gap-3 pt-4">
            {currentQuestion > 0 && (
              <button
                onClick={handlePrevious}
                className="px-6 py-3 bg-muted text-foreground font-pixel text-[10px] md:text-xs pixel-button"
              >
                ← BACK
              </button>
            )}
            <button
              onClick={handleNext}
              disabled={!isAnswered}
              className={`flex-1 px-6 py-3 font-pixel text-[10px] md:text-xs pixel-button ${
                !isAnswered
                  ? "bg-muted text-muted-foreground opacity-50 cursor-not-allowed"
                  : "bg-primary text-primary-foreground hover:bg-primary/90"
              }`}
            >
              {currentQuestion === totalSteps - 1
                ? "✓ FINISH"
                : "NEXT →"}
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
}
