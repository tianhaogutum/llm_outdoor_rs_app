"use client";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Route, UserProfile } from "@/lib/mock-data";
import {
  X,
  MapPin,
  Clock,
  TrendingUp,
  TrendingDown,
  Mountain,
  Sparkles,
  Lock,
  Repeat,
  MoveRight,
  Train,
  Bus,
  Car,
  Star,
  Footprints,
} from "lucide-react";
import { featureLabel } from "@/lib/api-transforms";
import { OsmAttribution, formatDuration, formatKm, routePlaceLine } from "@/components/route-facts";

const ACCESS_ICONS = { train: Train, tram: Train, bus: Bus, parking: Car } as const;

function pct(share?: number): string {
  return share === undefined ? "–" : `${Math.round(share * 100)}%`;
}

interface RouteDetailModalProps {
  route: Route;
  userProfile: UserProfile;
  isLoggedIn: boolean;
  onClose: () => void;
  onStartRoute: (route: Route) => void;
  onGoToQuestionnaire: () => void;
}

export function RouteDetailModal({
  route,
  userProfile,
  isLoggedIn,
  onClose,
  onStartRoute,
  onGoToQuestionnaire,
}: RouteDetailModalProps) {
  const isLocked = route.isLocked && userProfile.xp < (route.xpRequired || 0);
  const isCompleted = userProfile.completedRoutes.includes(route.id);

  const hasCompletedQuestionnaire = isLoggedIn && userProfile.explorerType;
  const facts = route.facts;
  const isOsm = facts?.source === "osm";
  const tags = facts?.allTags?.length ? facts.allTags : route.tags;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-start justify-center p-2 sm:p-4 overflow-y-auto">
      <Card className="max-w-3xl w-full mt-2 sm:mt-4 mb-auto border-4 border-border shadow-2xl max-h-[calc(100vh-1rem)] sm:max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
        {/* Header Image */}
        <div className="relative h-64 bg-muted flex-shrink-0">
          <img
            src={route.imageUrl || "/placeholder.svg"}
            alt={route.name}
            className="w-full h-full object-cover"
          />
          <Button
            variant="secondary"
            size="icon"
            className="absolute top-4 right-4"
            onClick={onClose}
          >
            <X className="w-5 h-5" />
          </Button>

          {/* XP Reward Badge */}
          <div className="absolute bottom-4 left-4 bg-accent/90 backdrop-blur-sm text-accent-foreground px-4 py-2 rounded-full flex items-center gap-2 font-semibold shadow-lg">
            <Sparkles className="w-5 h-5" />
            <span>{route.xpReward} XP</span>
          </div>

          {isLocked && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
              <div className="text-center text-white">
                <Lock className="w-16 h-16 mx-auto mb-3" />
                <p className="text-xl font-semibold">Locked Route</p>
                <p className="text-sm mt-1">
                  Requires {route.xpRequired} XP to unlock
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto flex-1">
          {/* Title & Location */}
          <div>
            <div className="flex items-start justify-between gap-4 mb-2">
              <h2 className="text-3xl font-bold text-foreground">
                {route.name}
              </h2>
              <Badge
                variant={
                  route.difficulty === "easy"
                    ? "secondary"
                    : route.difficulty === "expert"
                    ? "destructive"
                    : "default"
                }
                className="text-sm"
              >
                {route.difficulty}
              </Badge>
            </div>
            <p className="text-muted-foreground flex items-center gap-2">
              <MapPin className="w-4 h-4" />
              {routePlaceLine(route)}
              {facts?.region && <span className="text-xs">· {facts.region}</span>}
            </p>
          </div>

          {/* Description: OSM text is plain text (never injected as HTML); legacy texts carry markup */}
          {isOsm ? (
            <div className="space-y-1">
              <p className="text-foreground leading-relaxed whitespace-pre-line line-clamp-9">
                {route.description}
              </p>
              {facts?.descriptionGenerated && (
                <p className="text-xs italic text-muted-foreground">
                  Description generated from OpenStreetMap facts about this route.
                </p>
              )}
            </div>
          ) : (
            <div
              className="text-foreground leading-relaxed line-clamp-9"
              dangerouslySetInnerHTML={{ __html: route.description }}
            />
          )}

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 p-4 bg-muted/50 rounded-lg border-2 border-border">
            <div className="text-center">
              <Mountain className="w-6 h-6 mx-auto mb-1 text-primary" />
              <p className="text-sm text-muted-foreground">Distance</p>
              <p className="text-lg font-bold text-foreground">
                {formatKm(route.distance)}
              </p>
            </div>
            <div className="text-center">
              <TrendingUp className="w-6 h-6 mx-auto mb-1 text-primary" />
              <p className="text-sm text-muted-foreground">Ascent</p>
              <p className="text-lg font-bold text-foreground">
                {route.elevation} m
              </p>
            </div>
            {facts?.descent !== undefined && (
              <div className="text-center">
                <TrendingDown className="w-6 h-6 mx-auto mb-1 text-primary" />
                <p className="text-sm text-muted-foreground">Descent</p>
                <p className="text-lg font-bold text-foreground">{facts.descent} m</p>
              </div>
            )}
            <div className="text-center">
              <Clock className="w-6 h-6 mx-auto mb-1 text-primary" />
              <p className="text-sm text-muted-foreground">Duration</p>
              <p className="text-lg font-bold text-foreground">
                {formatDuration(route.duration, facts?.estDays)}
              </p>
            </div>
            {facts?.isLoop !== undefined && (
              <div className="text-center">
                {facts.isLoop ? (
                  <Repeat className="w-6 h-6 mx-auto mb-1 text-primary" />
                ) : (
                  <MoveRight className="w-6 h-6 mx-auto mb-1 text-primary" />
                )}
                <p className="text-sm text-muted-foreground">Shape</p>
                <p className="text-lg font-bold text-foreground">
                  {facts.isLoop ? "Loop" : "Linear"}
                </p>
              </div>
            )}
            {facts?.minEle !== undefined && facts?.maxEle !== undefined && (
              <div className="text-center">
                <Mountain className="w-6 h-6 mx-auto mb-1 text-primary" />
                <p className="text-sm text-muted-foreground">Elevation range</p>
                <p className="text-lg font-bold text-foreground">
                  {facts.minEle}–{facts.maxEle} m
                </p>
              </div>
            )}
            {facts?.maxGrade && (
              <div className="text-center">
                <Footprints className="w-6 h-6 mx-auto mb-1 text-primary" />
                <p className="text-sm text-muted-foreground">Hardest grade</p>
                <p className="text-lg font-bold text-foreground">{facts.maxGrade}</p>
              </div>
            )}
          </div>

          {/* Highlights: named OSM features on the route */}
          {facts && facts.highlights.length > 0 && (
            <div>
              <h3 className="text-lg font-bold text-foreground mb-3">Highlights</h3>
              <ul className="grid sm:grid-cols-2 gap-2">
                {facts.highlights.map((h) => (
                  <li
                    key={`${h.type}-${h.name}`}
                    className="flex items-center gap-2 p-2 bg-muted/30 rounded-lg border border-border text-sm"
                  >
                    <Star className="w-4 h-4 text-accent shrink-0" />
                    <span className="flex-1 text-foreground">
                      {h.name}
                      <span className="text-muted-foreground"> · {featureLabel(h.type)}</span>
                    </span>
                    <span className="text-xs text-muted-foreground">km {h.atKm}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Getting there and terrain */}
          {facts && (facts.access.length > 0 || facts.endAccess.length > 0 || facts.surface) && (
            <div className="grid md:grid-cols-2 gap-4">
              {(facts.access.length > 0 || facts.endAccess.length > 0) && (
                <div className="p-3 rounded-lg border border-border space-y-2">
                  <h4 className="font-semibold text-foreground">Getting there</h4>
                  {facts.access.map((a) => {
                    const Icon = ACCESS_ICONS[a.kind];
                    return (
                      <p key={`start-${a.kind}`} className="text-sm text-foreground flex items-center gap-2">
                        <Icon className="w-4 h-4 text-primary shrink-0" />
                        {a.label}
                      </p>
                    );
                  })}
                  {facts.endAccess.length > 0 && (
                    <>
                      <p className="text-xs text-muted-foreground pt-1">At the finish</p>
                      {facts.endAccess.map((a) => {
                        const Icon = ACCESS_ICONS[a.kind];
                        return (
                          <p key={`end-${a.kind}`} className="text-sm text-foreground flex items-center gap-2">
                            <Icon className="w-4 h-4 text-primary shrink-0" />
                            {a.label}
                          </p>
                        );
                      })}
                    </>
                  )}
                </div>
              )}
              {facts.surface && (
                <div className="p-3 rounded-lg border border-border space-y-2">
                  <h4 className="font-semibold text-foreground">Surface</h4>
                  <div className="flex h-3 w-full overflow-hidden rounded-full border border-border">
                    <div className="bg-primary" style={{ width: pct(facts.surface.paved) }} />
                    <div className="bg-accent" style={{ width: pct(facts.surface.unpaved) }} />
                  </div>
                  <p className="text-sm text-foreground">
                    Paved {pct(facts.surface.paved)} · Unpaved {pct(facts.surface.unpaved)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Paths {pct(facts.surface.path)} · Tracks {pct(facts.surface.track)} · Roads {pct(facts.surface.road)}
                  </p>
                </div>
              )}
            </div>
          )}

          {facts && facts.protectedAreas.length > 0 && (
            <p className="text-sm text-muted-foreground">
              Passes through: {facts.protectedAreas.join(", ")}
            </p>
          )}

          {/* Breakpoints Preview */}
          <div>
            <h3 className="text-lg font-bold text-foreground mb-3">
              Journey Breakpoints
            </h3>
            <div className="space-y-2">
              {route.breakpoints.map((bp, index) => (
                <div
                  key={bp.id}
                  className="flex items-center gap-3 p-3 bg-muted/30 rounded-lg border border-border"
                >
                  <div className="shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold text-foreground">{bp.name}</p>
                    {bp.poiType && (
                      <p className="text-sm text-muted-foreground">
                        {bp.poiType !== "start" && bp.poiType !== "finish" && `${featureLabel(bp.poiType)} · `}
                        km {bp.distance}
                      </p>
                    )}
                  </div>
                  {bp.type === "quest" && (
                    <Badge variant="secondary" className="text-xs">
                      Mini Quest
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Tags (all of them for catalogue routes) */}
          {tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {tags.map((tag) => (
                <Badge key={tag} variant="outline" className="text-xs">
                  #{tag}
                </Badge>
              ))}
            </div>
          )}

          {isOsm && (
            <div className="space-y-1">
              <OsmAttribution route={route} />
              {facts?.website && /^https?:\/\//i.test(facts.website) && (
                <p className="text-[11px] text-muted-foreground">
                  Route website (from OSM):{" "}
                  <a href={facts.website} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                    {facts.website.replace(/^https?:\/\//, "").slice(0, 60)}
                  </a>
                </p>
              )}
            </div>
          )}

          <div className="pt-4 border-t-2 border-border">
            {!hasCompletedQuestionnaire ? (
              <div className="text-center">
                <p className="text-muted-foreground mb-4">
                  Complete your quick setup to start this adventure and track
                  your progress.
                </p>
                <Button
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold"
                  onClick={onGoToQuestionnaire}
                >
                  Take 2-minute setup →
                </Button>
              </div>
            ) : isLocked ? (
              <div className="text-center">
                <p className="text-muted-foreground mb-2">
                  You need {(route.xpRequired || 0) - userProfile.xp} more XP to
                  unlock this route
                </p>
                <Button className="w-full" disabled>
                  <Lock className="mr-2 w-4 h-4" />
                  Locked
                </Button>
              </div>
            ) : (
              <Button
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-lg py-6"
                onClick={() => onStartRoute(route)}
              >
                <Sparkles className="mr-2 w-5 h-5" />
                {isCompleted ? "Replay Adventure" : "Start Adventure →"}
              </Button>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
