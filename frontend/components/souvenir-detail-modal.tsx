"use client";

import React from "react";
import { DigitalSouvenir } from "@/lib/mock-data";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatDate, getRouteTypeStyle, getDifficultyStyle } from "@/lib/utils";
import {
  X,
  MapPin,
  Calendar,
  Sparkles,
  TrendingUp,
  Award,
  Star,
  Footprints,
  Clock,
  Copy,
  CheckCircle2,
} from "lucide-react";

interface SouvenirDetailModalProps {
  souvenir: DigitalSouvenir | null;
  isOpen: boolean;
  onClose: () => void;
}

export function SouvenirDetailModal({
  souvenir,
  isOpen,
  onClose,
}: SouvenirDetailModalProps) {
  if (!souvenir) return null;

  const [copied, setCopied] = React.useState(false);

  // Parse XP breakdown
  let xpBreakdown: any = null;
  if (souvenir.xpBreakdown) {
    try {
      xpBreakdown =
        typeof souvenir.xpBreakdown === "string"
          ? JSON.parse(souvenir.xpBreakdown)
          : souvenir.xpBreakdown;
    } catch (e) {
      // Ignore parse errors
    }
  }

  const routeTypeStyle = souvenir.routeType
    ? getRouteTypeStyle(souvenir.routeType)
    : null;

  const difficultyStyle = getDifficultyStyle(souvenir.difficulty);

  // Generate share text and URL
  const shareText = `🏆 I completed ${souvenir.routeName} and earned ${
    souvenir.xpGained
  } XP on TrailSaga!\n\n📍 ${souvenir.location}\n📏 ${souvenir.distance.toFixed(
    1
  )}km\n⭐ ${souvenir.xpGained} XP\n\nJoin me on this adventure!`;
  const shareUrl = typeof window !== "undefined" ? window.location.origin : "";

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const handleWhatsAppShare = () => {
    const whatsappText = encodeURIComponent(shareText + "\n" + shareUrl);
    window.open(`https://wa.me/?text=${whatsappText}`, "_blank");
  };

  const handleTelegramShare = () => {
    const telegramText = encodeURIComponent(shareText);
    window.open(
      `https://t.me/share/url?url=${encodeURIComponent(
        shareUrl
      )}&text=${telegramText}`,
      "_blank"
    );
  };

  const handleTwitterShare = () => {
    const tweetText = encodeURIComponent(
      `🏆 Completed ${souvenir.routeName} - ${souvenir.xpGained} XP earned! 🎉`
    );
    window.open(
      `https://twitter.com/intent/tweet?text=${tweetText}&url=${encodeURIComponent(
        shareUrl
      )}`,
      "_blank"
    );
  };

  const handleFacebookShare = () => {
    window.open(
      `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(
        shareUrl
      )}`,
      "_blank"
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col p-0">
        <DialogHeader className="px-6 pt-6 pb-4 border-b">
          <DialogTitle className="flex items-center justify-between">
            <span className="text-2xl font-bold">{souvenir.routeName}</span>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-5 h-5" />
            </Button>
          </DialogTitle>
        </DialogHeader>

        {/* Scrollable content area */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <div className="space-y-4">
            {/* Route Type and Date */}
            <div className="flex items-center justify-between">
              {routeTypeStyle && (
                <Badge
                  className={`${routeTypeStyle.textColor} bg-transparent border-2 ${routeTypeStyle.borderColor} font-semibold text-sm`}
                >
                  <span className="mr-1">{routeTypeStyle.icon}</span>
                  {routeTypeStyle.label}
                </Badge>
              )}

              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Calendar className="w-4 h-4" />
                {formatDate(souvenir.completedAt)}
              </div>
            </div>

            {/* Pixel Art - Large Display */}
            {souvenir.pixelImageSvg && (
              <div
                className="w-full flex items-center justify-center border-4 border-border rounded-lg overflow-hidden"
                style={{
                  imageRendering: "pixelated",
                  aspectRatio: "4/3",
                }}
              >
                <div
                  dangerouslySetInnerHTML={{ __html: souvenir.pixelImageSvg }}
                  style={{
                    width: "100%",
                    height: "auto",
                    display: "block",
                  }}
                />
              </div>
            )}

            {/* Location */}
            <div className="flex items-center gap-2 text-muted-foreground">
              <MapPin className="w-4 h-4" />
              <span>{souvenir.location}</span>
            </div>

            {/* Stats Grid - Enhanced */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-muted/30 rounded-lg border-2 border-border">
              {/* XP Gained */}
              <div className="text-center">
                <Star className="w-5 h-5 text-accent mx-auto mb-2" />
                <div className="text-2xl font-bold text-accent">
                  {souvenir.xpGained}
                </div>
                <div className="text-xs text-muted-foreground">XP Earned</div>
              </div>

              {/* Distance */}
              <div className="text-center">
                <Footprints className="w-5 h-5 text-primary mx-auto mb-2" />
                <div className="text-2xl font-bold text-foreground">
                  {souvenir.distance.toFixed(1)}
                </div>
                <div className="text-xs text-muted-foreground">Kilometers</div>
              </div>

              {/* Duration */}
              {souvenir.duration && (
                <div className="text-center">
                  <Clock className="w-5 h-5 text-secondary mx-auto mb-2" />
                  <div className="text-2xl font-bold text-foreground">
                    {souvenir.duration}
                  </div>
                  <div className="text-xs text-muted-foreground">Minutes</div>
                </div>
              )}

              {/* Difficulty */}
              <div className="text-center">
                <div className="text-3xl mx-auto mb-2">
                  {difficultyStyle.emoji}
                </div>
                <div className={`text-lg font-bold ${difficultyStyle.color}`}>
                  {difficultyStyle.label}
                </div>
                <div className="text-xs text-muted-foreground">Difficulty</div>
              </div>
            </div>

            {/* Unlocked Achievements */}
            {souvenir.unlockedAchievements &&
              souvenir.unlockedAchievements.length > 0 && (
                <div className="p-4 bg-gradient-to-r from-accent/20 to-primary/20 border-2 border-accent/40 rounded-lg">
                  <div className="flex items-center gap-2 mb-3">
                    <Award className="w-5 h-5 text-accent" />
                    <h3 className="font-bold text-foreground text-lg">
                      🎉 Achievements Unlocked
                    </h3>
                  </div>
                  <div className="grid gap-3">
                    {souvenir.unlockedAchievements.map((achievement) => (
                      <div
                        key={achievement.id}
                        className="flex items-center gap-3 p-3 bg-card/60 backdrop-blur-sm rounded-lg border border-accent/30"
                      >
                        <span className="text-3xl">{achievement.icon}</span>
                        <div className="flex-1">
                          <h4 className="font-bold text-foreground">
                            {achievement.name}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {achievement.description}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            {/* AI Summary */}
            {souvenir.genaiSummary && (
              <div className="p-4 bg-muted/50 rounded-lg border-2 border-border">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles className="w-5 h-5 text-accent" />
                  <h3 className="font-bold text-foreground">AI Trip Summary</h3>
                </div>
                <p className="text-sm text-muted-foreground italic leading-relaxed">
                  {souvenir.genaiSummary}
                </p>
              </div>
            )}

            {/* XP Breakdown */}
            {xpBreakdown && (
              <div className="p-4 bg-muted/50 rounded-lg border-2 border-border">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-5 h-5 text-accent" />
                  <h3 className="font-bold text-foreground">XP Breakdown</h3>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">
                      Base Route XP
                    </span>
                    <span className="font-semibold text-foreground">
                      {xpBreakdown.base || 0} XP
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">
                      Quest Bonuses
                    </span>
                    <span className="font-semibold text-foreground">
                      {xpBreakdown.quests || 0} XP
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">
                      Difficulty Multiplier
                    </span>
                    <span className="font-semibold text-foreground">
                      ×{xpBreakdown.difficulty_multiplier || 1.0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center pt-2 border-t-2 border-border">
                    <span className="font-bold text-foreground">
                      Total XP Earned
                    </span>
                    <span className="font-bold text-accent text-lg">
                      {xpBreakdown.total || souvenir.xpGained} XP
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Fixed Share Section at Bottom */}
        <div className="border-t bg-background px-6 py-4">
          <div className="space-y-3">
            <h3 className="font-bold text-foreground text-center">
              Share This Adventure
            </h3>

            {/* Primary Share Buttons */}
            <div className="grid grid-cols-2 gap-3">
              {/* WhatsApp */}
              <Button
                onClick={handleWhatsAppShare}
                className="bg-[#25D366] hover:bg-[#20BA5A] text-white font-semibold"
              >
                <svg
                  className="w-5 h-5 mr-2"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z" />
                </svg>
                WhatsApp
              </Button>

              {/* Telegram */}
              <Button
                onClick={handleTelegramShare}
                className="bg-[#0088cc] hover:bg-[#0077b5] text-white font-semibold"
              >
                <svg
                  className="w-5 h-5 mr-2"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z" />
                </svg>
                Telegram
              </Button>
            </div>

            {/* Secondary Share Buttons */}
            <div className="grid grid-cols-2 gap-3">
              {/* Twitter/X */}
              <Button
                onClick={handleTwitterShare}
                variant="outline"
                className="font-semibold border-2"
              >
                <svg
                  className="w-4 h-4 mr-2"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
                Twitter
              </Button>

              {/* Facebook */}
              <Button
                onClick={handleFacebookShare}
                variant="outline"
                className="font-semibold border-2"
              >
                <svg
                  className="w-4 h-4 mr-2"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
                </svg>
                Facebook
              </Button>
            </div>

            {/* Copy Link Button */}
            <Button
              onClick={handleCopyLink}
              variant="outline"
              className="w-full font-semibold border-2"
            >
              {copied ? (
                <>
                  <CheckCircle2 className="w-4 h-4 mr-2 text-green-500" />
                  Link Copied!
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4 mr-2" />
                  Copy Link
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
