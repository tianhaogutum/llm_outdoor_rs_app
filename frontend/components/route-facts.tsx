"use client";

import { Route } from "@/lib/mock-data";
import { featureLabel } from "@/lib/api-transforms";
import { Repeat, MoveRight } from "lucide-react";

/** "45 min", "2h 05m", "3 days (19h 14m)" */
export function formatDuration(minutes: number, estDays?: number): string {
  if (!minutes || minutes <= 0) return "–";
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  const hm = h === 0 ? `${m} min` : `${h}h ${String(m).padStart(2, "0")}m`;
  return estDays && estDays > 1 ? `${estDays} days (${hm})` : hm;
}

export function formatKm(km: number): string {
  return km >= 100 ? `${Math.round(km)} km` : `${km} km`;
}

/** Loop / linear badge; nothing for routes without catalogue facts */
export function RouteShapeBadge({ route, className = "" }: { route: Route; className?: string }) {
  const isLoop = route.facts?.isLoop;
  if (isLoop === undefined) return null;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border border-border bg-card/90 px-2 py-0.5 text-xs font-semibold text-foreground ${className}`}
    >
      {isLoop ? <Repeat className="w-3 h-3" /> : <MoveRight className="w-3 h-3" />}
      {isLoop ? "Loop" : "Linear"}
    </span>
  );
}

/** "Start → finish" place line for OSM routes, else the legacy location string */
export function routePlaceLine(route: Route): string {
  const f = route.facts;
  if (!f?.locality) return route.location;
  const where = f.district ? `${f.locality}, ${f.district}` : f.locality;
  if (f.isLoop === false && f.endLocality && f.endLocality !== f.locality) {
    return `${f.locality} → ${f.endLocality}${f.district ? ` · ${f.district}` : ""}`;
  }
  return where;
}

/** Up to `limit` named highlights as small chips */
export function HighlightChips({ route, limit = 3 }: { route: Route; limit?: number }) {
  const items = route.facts?.highlights?.slice(0, limit) ?? [];
  if (items.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((h) => (
        <span
          key={`${h.type}-${h.name}`}
          className="rounded-md border border-border bg-muted/40 px-2 py-0.5 text-xs text-foreground"
          title={`${featureLabel(h.type)} at km ${h.atKm}`}
        >
          <span className="text-muted-foreground">{featureLabel(h.type)}:</span> {h.name}
        </span>
      ))}
    </div>
  );
}

/** ODbL attribution with a link to the OSM relation; required wherever OSM routes are shown */
export function OsmAttribution({ route, className = "" }: { route: Route; className?: string }) {
  const f = route.facts;
  if (f?.source !== "osm") return null;
  return (
    <p className={`text-[11px] text-muted-foreground ${className}`}>
      Route data{" "}
      <a
        href="https://www.openstreetmap.org/copyright"
        target="_blank"
        rel="noopener noreferrer"
        className="underline hover:text-foreground"
      >
        {f.attribution || "© OpenStreetMap contributors, ODbL 1.0"}
      </a>
      {f.sourceUrl && (
        <>
          {" · "}
          <a href={f.sourceUrl} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
            view on OpenStreetMap
          </a>
        </>
      )}
    </p>
  );
}
