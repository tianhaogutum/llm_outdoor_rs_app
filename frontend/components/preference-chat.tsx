"use client";

/**
 * Preference chat (docs/CHAT.md): one message in, one assistant turn out. Shows what the system
 * understood as chips (removable; enum chips cycle through their values), the intent it will search
 * with, anything it ignored, and the clarifying question with quick replies.
 *
 * Used in onboarding (variant "pixel") and as the "refine" box on the results page (variant "plain").
 */

import { useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api-client";
import type {
  ApiChatResponse,
  ApiChatSlotEdit,
  ApiChatSlots,
  ChatActivity,
  ChatDifficulty,
  ChatDistanceBand,
} from "@/lib/api-types";
import { MessageCircle, RotateCcw, Send, X } from "lucide-react";

const ACTIVITY_LABEL: Record<ChatActivity, string> = {
  hiking: "Hiking",
  cycling: "Cycling",
  mtb: "Mountain biking",
  running: "Running",
};
const ACTIVITIES: ChatActivity[] = ["hiking", "cycling", "mtb", "running"];
const DIFFICULTIES: ChatDifficulty[] = ["easy", "moderate", "hard"];
const BANDS: ChatDistanceBand[] = ["short", "medium", "long"];
const MAX_MESSAGE = 1000;

function next<T>(values: T[], current: T | null): T {
  const i = current === null ? -1 : values.indexOf(current);
  return values[(i + 1) % values.length];
}

interface Chip {
  key: string;
  label: string;
  tone: "want" | "avoid" | "slot" | "default";
  title?: string;
  onRemove?: ApiChatSlotEdit;
  onCycle?: ApiChatSlotEdit;
}

function chipsFor(slots: ApiChatSlots): Chip[] {
  const src = slots.sources || {};
  const tone = (slot: string): Chip["tone"] => (src[slot] === "default" ? "default" : "slot");
  const chips: Chip[] = [];
  if (slots.activity)
    chips.push({
      key: "activity",
      label: ACTIVITY_LABEL[slots.activity] + (src.activity === "default" ? " (default)" : ""),
      tone: tone("activity"),
      title: "Click to change the activity",
      onCycle: { activity: next(ACTIVITIES, slots.activity) },
      onRemove: { clear: ["activity"] },
    });
  if (slots.difficulty)
    chips.push({
      key: "difficulty",
      label: `Difficulty: ${slots.difficulty}`,
      tone: "slot",
      title: "Click to change; × goes back to your questionnaire answer",
      onCycle: { difficulty: next(DIFFICULTIES, slots.difficulty) },
      onRemove: { clear: ["difficulty"] },
    });
  if (slots.distance_band) {
    const r = slots.distance_range_km;
    chips.push({
      key: "distance_band",
      label: `Length: ${slots.distance_band}${r ? ` (${r[0]}–${r[1]} km)` : ""}`,
      tone: "slot",
      title: "Click to change; × goes back to your questionnaire answer",
      onCycle: { distance_band: next(BANDS, slots.distance_band) },
      onRemove: { clear: ["distance_band"] },
    });
  }
  if (slots.duration_hours != null)
    chips.push({ key: "duration", label: `About ${slots.duration_hours} h`, tone: "slot",
                 onRemove: { clear: ["duration_hours"] } });
  if (slots.distance_cap_km != null)
    chips.push({ key: "cap", label: `Max ${slots.distance_cap_km} km`, tone: "slot",
                 onRemove: { clear: ["distance_cap_km"] } });
  if (slots.place)
    chips.push({ key: "place", label: `Near ${slots.place.name}`, tone: "slot",
                 title: "Coordinates from the catalogue's place list", onRemove: { clear: ["place"] } });
  if (slots.travel_radius_km != null && slots.place)
    chips.push({
      key: "radius",
      label: `Within ${slots.travel_radius_km} km` + (src.travel_radius_km === "default" ? " (default)" : ""),
      tone: tone("travel_radius_km"),
      onRemove: { clear: ["travel_radius_km"] },
    });
  for (const t of slots.tags_preferred)
    chips.push({ key: `+${t}`, label: `+ ${t}`, tone: "want", onRemove: { tags_remove: [t] } });
  for (const t of slots.tags_avoided)
    chips.push({ key: `-${t}`, label: `no ${t}`, tone: "avoid", onRemove: { tags_remove: [t] } });
  return chips;
}

interface PreferenceChatProps {
  profileId: number;
  variant?: "pixel" | "plain";
  title?: string;
  placeholder?: string;
  /** Called after every turn or chip edit that changed something (refresh recommendations here). */
  onUpdated?: (response: ApiChatResponse) => void;
}

export function PreferenceChat({
  profileId,
  variant = "plain",
  title = "Refine your recommendations",
  placeholder = "e.g. an easy loop near Tegernsee with a hut, max 12 km",
  onUpdated,
}: PreferenceChatProps) {
  const [state, setState] = useState<ApiChatResponse | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const pixel = variant === "pixel";

  useEffect(() => {
    let alive = true;
    apiClient
      .getChat(profileId)
      .then((r) => alive && setState(r))
      .catch(() => alive && setState(null));
    return () => {
      alive = false;
    };
  }, [profileId]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [state?.turns.length]);

  const apply = (r: ApiChatResponse) => {
    setState(r);
    if (r.changes.length > 0) onUpdated?.(r);
  };

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      apply(await apiClient.sendChatMessage(profileId, trimmed.slice(0, MAX_MESSAGE)));
      setMessage("");
    } catch (e: any) {
      setError(e?.message || "The chat is not available right now.");
    } finally {
      setBusy(false);
    }
  };

  const edit = async (change: ApiChatSlotEdit) => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      apply(await apiClient.editChatSlots(profileId, change));
    } catch (e: any) {
      setError(e?.message || "Could not change that.");
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    setBusy(true);
    try {
      setState(await apiClient.resetChat(profileId));
    } catch {
      /* keep the old state */
    } finally {
      setBusy(false);
    }
  };

  const slots = state?.slots;
  const chips = slots ? chipsFor(slots) : [];
  const turns = state?.turns ?? [];
  const question = state?.status === "clarify" ? state.question : null;

  const box = pixel ? "bg-card pixel-border-sm" : "bg-card border-2 border-border rounded-lg";
  const small = pixel ? "text-[10px]" : "text-xs";
  const heading = pixel ? "font-pixel text-[10px] md:text-xs" : "text-sm font-semibold";
  const chipTone: Record<Chip["tone"], string> = {
    want: "bg-primary text-primary-foreground",
    avoid: "bg-destructive/15 text-destructive border border-destructive/40",
    slot: "bg-secondary text-secondary-foreground",
    default: "bg-muted text-muted-foreground border border-dashed border-border",
  };

  return (
    <div className={`${box} p-4 space-y-3 text-left`} data-testid="preference-chat">
      <div className="flex items-center justify-between gap-2">
        <h3 className={`${heading} flex items-center gap-2 text-foreground`}>
          <MessageCircle className="w-4 h-4" /> {title}
        </h3>
        {turns.length > 0 && (
          <button type="button" onClick={reset} disabled={busy} title="Start a new conversation (keeps your preferences)"
                  className={`${small} text-muted-foreground hover:text-foreground flex items-center gap-1`}>
            <RotateCcw className="w-3 h-3" /> new conversation
          </button>
        )}
      </div>

      {turns.length > 0 && (
        <div ref={logRef} className="max-h-48 overflow-y-auto space-y-2 pr-1" aria-live="polite">
          {turns.map((t, i) => (
            <div key={i} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
              <p className={`${small} leading-relaxed px-3 py-2 max-w-[85%] whitespace-pre-wrap ${
                t.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
              } ${pixel ? "" : "rounded-lg"}`}>
                {t.text}
              </p>
            </div>
          ))}
        </div>
      )}

      {question && (
        <div className={`${small} space-y-2`}>
          <p className="font-medium text-foreground">
            {question.text}
            <span className="text-muted-foreground">
              {" "}(question {state!.clarify_turns_used} of {state!.clarify_turns_max})
            </span>
          </p>
          {question.options.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {question.options.map((o) => (
                <button key={o} type="button" disabled={busy}
                        onClick={() => send(ACTIVITY_LABEL[o as ChatActivity] ?? o)}
                        className={`px-3 py-1 ${pixel ? "pixel-border-sm font-pixel" : "border-2 border-border rounded-md"} bg-card hover:bg-muted`}>
                  {ACTIVITY_LABEL[o as ChatActivity] ?? o}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); send(message); }}>
        <input
          type="text"
          value={message}
          maxLength={MAX_MESSAGE}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={question ? "Your answer…" : placeholder}
          aria-label="Message to the route assistant"
          className={`flex-1 px-3 py-2 ${small} bg-background text-foreground outline-none placeholder:text-muted-foreground ${
            pixel ? "pixel-border-sm" : "border-2 border-border rounded-md focus:ring-2 focus:ring-primary"
          }`}
        />
        <button type="submit" disabled={busy || !message.trim()} aria-label="Send"
                className={`px-3 py-2 bg-primary text-primary-foreground disabled:opacity-50 ${
                  pixel ? "pixel-button font-pixel text-[10px]" : "rounded-md"
                }`}>
          {busy ? "…" : <Send className="w-4 h-4" />}
        </button>
      </form>
      {error && <p className={`${small} text-destructive`}>⚠️ {error}</p>}

      {slots && (chips.length > 0 || slots.intent || slots.questionnaire_tags.length > 0) && (
        <div className="space-y-2 pt-2 border-t-2 border-dashed border-border">
          <p className={`${small} text-muted-foreground`}>What I will use (click a chip to change it, × to remove):</p>
          <div className="flex flex-wrap gap-2">
            {chips.map((c) => (
              <span key={c.key} className={`inline-flex items-center gap-1 px-2 py-1 ${small} ${chipTone[c.tone]} ${
                pixel ? "" : "rounded-md"
              }`}>
                {c.onCycle ? (
                  <button type="button" title={c.title} disabled={busy} onClick={() => edit(c.onCycle!)}
                          className="hover:underline">{c.label}</button>
                ) : (
                  <span title={c.title}>{c.label}</span>
                )}
                {c.onRemove && (
                  <button type="button" aria-label={`remove ${c.label}`} disabled={busy}
                          onClick={() => edit(c.onRemove!)} className="opacity-70 hover:opacity-100">
                    <X className="w-3 h-3" />
                  </button>
                )}
              </span>
            ))}
            {slots.questionnaire_tags.map((t) => (
              <span key={`q-${t}`} title="From your questionnaire answers"
                    className={`px-2 py-1 ${small} text-muted-foreground border border-border ${pixel ? "" : "rounded-md"}`}>
                {t}
              </span>
            ))}
          </div>
          {slots.intent && (
            <p className={`${small} text-foreground flex items-start gap-2`}>
              <span className="text-muted-foreground shrink-0">Searching for:</span>
              <span className="italic">&ldquo;{slots.intent}&rdquo;</span>
              <button type="button" aria-label="remove the search text" disabled={busy}
                      onClick={() => edit({ clear: ["intent"] })} className="opacity-60 hover:opacity-100 shrink-0">
                <X className="w-3 h-3" />
              </button>
            </p>
          )}
          {state && state.dropped.length > 0 && (
            <p className={`${small} text-muted-foreground`}>
              Ignored: {state.dropped.map((d) => `${d.field}${d.value ? ` ${d.value}` : ""} (${d.reason})`).join("; ")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
