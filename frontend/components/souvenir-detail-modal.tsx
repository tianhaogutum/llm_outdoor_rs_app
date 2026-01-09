'use client';

import { DigitalSouvenir } from '@/lib/mock-data';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatDate, getRouteTypeStyle, getDifficultyStyle } from '@/lib/utils';
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
  Share2
} from 'lucide-react';

interface SouvenirDetailModalProps {
  souvenir: DigitalSouvenir | null;
  isOpen: boolean;
  onClose: () => void;
}

export function SouvenirDetailModal({ souvenir, isOpen, onClose }: SouvenirDetailModalProps) {
  if (!souvenir) return null;

  // Parse XP breakdown
  let xpBreakdown: any = null;
  if (souvenir.xpBreakdown) {
    try {
      xpBreakdown = typeof souvenir.xpBreakdown === 'string' 
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

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `${souvenir.routeName} - TrailSaga`,
          text: `I completed ${souvenir.routeName} and earned ${souvenir.xpGained} XP! 🏆`,
          url: window.location.href,
        });
      } catch (err) {
        console.log('Share cancelled or failed:', err);
      }
    } else {
      // Fallback: copy to clipboard
      const text = `I completed ${souvenir.routeName} and earned ${souvenir.xpGained} XP! 🏆`;
      navigator.clipboard.writeText(text);
      alert('Copied to clipboard!');
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <span className="text-2xl font-bold">{souvenir.routeName}</span>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-5 h-5" />
            </Button>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Route Type and Date */}
          <div className="flex items-center justify-between">
            {routeTypeStyle && (
              <Badge className={`${routeTypeStyle.textColor} bg-transparent border-2 ${routeTypeStyle.borderColor} font-semibold text-sm`}>
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
                imageRendering: 'pixelated',
                aspectRatio: '4/3',
              }}
            >
              <div
                dangerouslySetInnerHTML={{ __html: souvenir.pixelImageSvg }}
                style={{
                  width: '100%',
                  height: 'auto',
                  display: 'block',
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
              <div className="text-2xl font-bold text-accent">{souvenir.xpGained}</div>
              <div className="text-xs text-muted-foreground">XP Earned</div>
            </div>
            
            {/* Distance */}
            <div className="text-center">
              <Footprints className="w-5 h-5 text-primary mx-auto mb-2" />
              <div className="text-2xl font-bold text-foreground">{souvenir.distance.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">Kilometers</div>
            </div>
            
            {/* Duration */}
            {souvenir.duration && (
              <div className="text-center">
                <Clock className="w-5 h-5 text-secondary mx-auto mb-2" />
                <div className="text-2xl font-bold text-foreground">{souvenir.duration}</div>
                <div className="text-xs text-muted-foreground">Minutes</div>
              </div>
            )}
            
            {/* Difficulty */}
            <div className="text-center">
              <div className="text-3xl mx-auto mb-2">{difficultyStyle.emoji}</div>
              <div className={`text-lg font-bold ${difficultyStyle.color}`}>
                {difficultyStyle.label}
              </div>
              <div className="text-xs text-muted-foreground">Difficulty</div>
            </div>
          </div>

          {/* Unlocked Achievements */}
          {souvenir.unlockedAchievements && souvenir.unlockedAchievements.length > 0 && (
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
                      <h4 className="font-bold text-foreground">{achievement.name}</h4>
                      <p className="text-sm text-muted-foreground">{achievement.description}</p>
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
                  <span className="text-sm text-muted-foreground">Base Route XP</span>
                  <span className="font-semibold text-foreground">{xpBreakdown.base || 0} XP</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Quest Bonuses</span>
                  <span className="font-semibold text-foreground">{xpBreakdown.quests || 0} XP</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Difficulty Multiplier</span>
                  <span className="font-semibold text-foreground">×{xpBreakdown.difficulty_multiplier || 1.0}</span>
                </div>
                <div className="flex justify-between items-center pt-2 border-t-2 border-border">
                  <span className="font-bold text-foreground">Total XP Earned</span>
                  <span className="font-bold text-accent text-lg">{xpBreakdown.total || souvenir.xpGained} XP</span>
                </div>
              </div>
            </div>
          )}

          {/* Share Button */}
          <Button 
            onClick={handleShare}
            className="w-full bg-accent hover:bg-accent/90 text-accent-foreground font-semibold"
          >
            <Share2 className="w-4 h-4 mr-2" />
            Share This Adventure
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
