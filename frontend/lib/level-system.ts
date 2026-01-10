/**
 * Level and XP calculation system for frontend.
 *
 * This module provides a progressive leveling system where higher levels require more XP.
 * The system uses a linear progression formula where each level requires incrementally more XP.
 *
 * Formula:
 * - Level 1 → 2: BASE_XP (100 XP)
 * - Level 2 → 3: BASE_XP + XP_INCREMENT (150 XP)
 * - Level 3 → 4: BASE_XP + 2*XP_INCREMENT (200 XP)
 * - And so on...
 *
 * Example progression:
 *     Level   XP Needed   Cumulative XP
 *     1       -           0
 *     2       100         100
 *     3       150         250
 *     4       200         450
 *     5       250         700
 *     6       300         1000
 *     7       350         1350
 *     8       400         1750
 *     9       450         2200
 *     10      500         2700
 */

// Base XP required to reach level 2
export const BASE_XP = 100;

// XP increment per level (each level requires this much more XP than the previous)
export const XP_INCREMENT = 50;

/**
 * Calculate the XP required to advance FROM target_level TO target_level+1.
 *
 * @param targetLevel - The current level (e.g., 1 means XP needed to reach level 2)
 * @returns XP required to reach the next level
 *
 * @example
 * xpForLevel(1)  // Level 1 → 2: 100
 * xpForLevel(2)  // Level 2 → 3: 150
 * xpForLevel(5)  // Level 5 → 6: 300
 */
export function xpForLevel(targetLevel: number): number {
  if (targetLevel < 1) {
    return 0;
  }

  // Formula: BASE_XP + (level - 1) * XP_INCREMENT
  return BASE_XP + (targetLevel - 1) * XP_INCREMENT;
}

/**
 * Calculate the TOTAL cumulative XP required to reach a specific level.
 *
 * @param targetLevel - The target level (e.g., 3 means total XP to reach level 3)
 * @returns Total cumulative XP required
 *
 * @example
 * cumulativeXpForLevel(1)  // Starting level: 0
 * cumulativeXpForLevel(2)  // Total XP to reach level 2: 100
 * cumulativeXpForLevel(3)  // Total XP to reach level 3: 250
 * cumulativeXpForLevel(5)  // Total XP to reach level 5: 700
 */
export function cumulativeXpForLevel(targetLevel: number): number {
  if (targetLevel <= 1) {
    return 0;
  }

  // Sum of arithmetic sequence: n/2 * (first_term + last_term)
  // For levels 1→2, 2→3, ..., (target_level-1)→target_level
  // first_term = BASE_XP
  // last_term = BASE_XP + (target_level - 2) * XP_INCREMENT
  // n = target_level - 1

  const n = targetLevel - 1;
  const first = BASE_XP;
  const last = BASE_XP + (targetLevel - 2) * XP_INCREMENT;

  return Math.floor((n * (first + last)) / 2);
}

/**
 * Calculate the current level based on total XP using binary search.
 *
 * This is the inverse of cumulativeXpForLevel().
 *
 * @param totalXp - Total XP accumulated
 * @returns Current level (minimum 1)
 *
 * @example
 * calculateLevelFromXp(0)    // 1
 * calculateLevelFromXp(50)   // 1 (Less than 100)
 * calculateLevelFromXp(100)  // 2 (Exactly 100)
 * calculateLevelFromXp(250)  // 3 (Exactly 250)
 * calculateLevelFromXp(260)  // 3 (Between 250 and 450)
 * calculateLevelFromXp(700)  // 5 (Exactly 700)
 */
export function calculateLevelFromXp(totalXp: number): number {
  if (totalXp < 0) {
    return 1;
  }

  // Binary search for the level
  // We need to find the largest level L such that cumulativeXpForLevel(L) <= totalXp

  let left = 1;
  let right = 100; // Assume max level of 100
  let currentLevel = 1;

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    const xpNeeded = cumulativeXpForLevel(mid);

    if (xpNeeded <= totalXp) {
      currentLevel = mid;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return currentLevel;
}

/**
 * Calculate how much more XP is needed to reach the next level.
 *
 * @param totalXp - Current total XP
 * @returns XP needed to reach next level
 *
 * @example
 * xpToNextLevel(0)    // 100
 * xpToNextLevel(50)   // 50 (50/100 to level 2)
 * xpToNextLevel(100)  // 150 (Just reached level 2, now need 150 to reach level 3)
 * xpToNextLevel(200)  // 100 (At level 2 with 50 extra XP, need 100 more for level 3)
 */
export function xpToNextLevel(totalXp: number): number {
  const currentLevel = calculateLevelFromXp(totalXp);
  const nextLevel = currentLevel + 1;

  const xpNeededForNext = cumulativeXpForLevel(nextLevel);

  return xpNeededForNext - totalXp;
}

/**
 * Calculate XP progress within the current level.
 *
 * @param totalXp - Current total XP
 * @returns Object with currentXpInLevel and xpRequiredForLevel
 *
 * @example
 * xpProgressInCurrentLevel(0)    // { current: 0, required: 100 }
 * xpProgressInCurrentLevel(50)   // { current: 50, required: 100 }
 * xpProgressInCurrentLevel(100)  // { current: 0, required: 150 } (Just reached level 2)
 * xpProgressInCurrentLevel(150)  // { current: 50, required: 150 } (50 XP into level 2)
 */
export function xpProgressInCurrentLevel(totalXp: number): {
  current: number;
  required: number;
} {
  const currentLevel = calculateLevelFromXp(totalXp);
  const xpAtCurrentLevelStart = cumulativeXpForLevel(currentLevel);
  const xpInCurrentLevel = totalXp - xpAtCurrentLevelStart;
  const xpRequiredForNext = xpForLevel(currentLevel);

  return {
    current: xpInCurrentLevel,
    required: xpRequiredForNext,
  };
}

/**
 * Pre-generate a lookup table for quick reference (levels 1-100).
 */
export const LEVEL_XP_TABLE: Record<
  number,
  {
    xpForNextLevel: number;
    cumulativeXp: number;
  }
> = {};

// Populate the table
for (let level = 1; level <= 100; level++) {
  LEVEL_XP_TABLE[level] = {
    xpForNextLevel: xpForLevel(level),
    cumulativeXp: cumulativeXpForLevel(level),
  };
}
