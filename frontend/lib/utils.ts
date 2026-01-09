import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format date to relative time (e.g., "2 days ago", "Just now")
 */
export function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  const diffWeeks = Math.floor(diffDays / 7);
  const diffMonths = Math.floor(diffDays / 30);
  const diffYears = Math.floor(diffDays / 365);

  if (diffSeconds < 60) return 'Just now';
  if (diffMinutes < 60) return `${diffMinutes} ${diffMinutes === 1 ? 'minute' : 'minutes'} ago`;
  if (diffHours < 24) return `${diffHours} ${diffHours === 1 ? 'hour' : 'hours'} ago`;
  if (diffDays < 7) return `${diffDays} ${diffDays === 1 ? 'day' : 'days'} ago`;
  if (diffWeeks < 4) return `${diffWeeks} ${diffWeeks === 1 ? 'week' : 'weeks'} ago`;
  if (diffMonths < 12) return `${diffMonths} ${diffMonths === 1 ? 'month' : 'months'} ago`;
  return `${diffYears} ${diffYears === 1 ? 'year' : 'years'} ago`;
}

/**
 * Format date to readable format (e.g., "Jan 9, 2025")
 */
export function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

/**
 * Get route type icon and color
 */
export function getRouteTypeStyle(type: 'hiking' | 'running' | 'cycling') {
  const styles = {
    hiking: {
      icon: '⛰️',
      emoji: '🥾',
      color: 'from-green-500/20 to-emerald-500/20',
      borderColor: 'border-green-500',
      textColor: 'text-green-600',
      label: 'Hiking',
    },
    running: {
      icon: '🏃',
      emoji: '⚡',
      color: 'from-red-500/20 to-orange-500/20',
      borderColor: 'border-red-500',
      textColor: 'text-red-600',
      label: 'Running',
    },
    cycling: {
      icon: '🚴',
      emoji: '🚴',
      color: 'from-blue-500/20 to-cyan-500/20',
      borderColor: 'border-blue-500',
      textColor: 'text-blue-600',
      label: 'Cycling',
    },
  };
  return styles[type] || styles.hiking;
}

/**
 * Get difficulty emoji and style
 */
export function getDifficultyStyle(difficulty: string) {
  const lower = difficulty.toLowerCase();
  if (lower.includes('easy')) {
    return { emoji: '😊', color: 'text-green-600', label: 'Easy' };
  }
  if (lower.includes('medium') || lower.includes('moderate')) {
    return { emoji: '😐', color: 'text-yellow-600', label: 'Medium' };
  }
  if (lower.includes('hard') || lower.includes('difficult')) {
    return { emoji: '😰', color: 'text-red-600', label: 'Hard' };
  }
  if (lower.includes('expert')) {
    return { emoji: '🔥', color: 'text-purple-600', label: 'Expert' };
  }
  return { emoji: '😊', color: 'text-gray-600', label: difficulty };
}
