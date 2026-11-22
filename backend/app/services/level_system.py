"""
Level and XP calculation system.

This module provides a progressive leveling system where higher levels require more XP.
The system uses a linear progression formula where each level requires incrementally more XP.

Formula:
- Level 1 → 2: BASE_XP (100 XP)
- Level 2 → 3: BASE_XP + XP_INCREMENT (150 XP)
- Level 3 → 4: BASE_XP + 2*XP_INCREMENT (200 XP)
- And so on...

Example progression:
    Level   XP Needed   Cumulative XP
    1       -           0
    2       100         100
    3       150         250
    4       200         450
    5       250         700
    6       300         1000
    7       350         1350
    8       400         1750
    9       450         2200
    10      500         2700
"""

# Base XP required to reach level 2
BASE_XP = 100

# XP increment per level (each level requires this much more XP than the previous)
XP_INCREMENT = 50


def xp_for_level(target_level: int) -> int:
    """
    Calculate the XP required to advance FROM target_level TO target_level+1.

    Args:
        target_level: The current level (e.g., 1 means XP needed to reach level 2)

    Returns:
        XP required to reach the next level

    Examples:
        >>> xp_for_level(1)  # Level 1 → 2
        100
        >>> xp_for_level(2)  # Level 2 → 3
        150
        >>> xp_for_level(5)  # Level 5 → 6
        300
    """
    if target_level < 1:
        return 0

    # Formula: BASE_XP + (level - 1) * XP_INCREMENT
    return BASE_XP + (target_level - 1) * XP_INCREMENT


def cumulative_xp_for_level(target_level: int) -> int:
    """
    Calculate the TOTAL cumulative XP required to reach a specific level.

    Args:
        target_level: The target level (e.g., 3 means total XP to reach level 3)

    Returns:
        Total cumulative XP required

    Examples:
        >>> cumulative_xp_for_level(1)  # Starting level
        0
        >>> cumulative_xp_for_level(2)  # Total XP to reach level 2
        100
        >>> cumulative_xp_for_level(3)  # Total XP to reach level 3
        250
        >>> cumulative_xp_for_level(5)  # Total XP to reach level 5
        700
    """
    if target_level <= 1:
        return 0

    # Sum of arithmetic sequence: n/2 * (first_term + last_term)
    # For levels 1→2, 2→3, ..., (target_level-1)→target_level
    # first_term = BASE_XP
    # last_term = BASE_XP + (target_level - 2) * XP_INCREMENT
    # n = target_level - 1

    n = target_level - 1
    first = BASE_XP
    last = BASE_XP + (target_level - 2) * XP_INCREMENT

    return n * (first + last) // 2


def calculate_level_from_xp(total_xp: int) -> int:
    """
    Calculate the current level based on total XP using binary search.

    This is the inverse of cumulative_xp_for_level().

    Args:
        total_xp: Total XP accumulated

    Returns:
        Current level (minimum 1)

    Examples:
        >>> calculate_level_from_xp(0)
        1
        >>> calculate_level_from_xp(50)   # Less than 100
        1
        >>> calculate_level_from_xp(100)  # Exactly 100
        2
        >>> calculate_level_from_xp(250)  # Exactly 250
        3
        >>> calculate_level_from_xp(260)  # Between 250 and 450
        3
        >>> calculate_level_from_xp(700)  # Exactly 700
        5
    """
    if total_xp < 0:
        return 1

    # Binary search for the level
    # We need to find the largest level L such that cumulative_xp_for_level(L) <= total_xp

    left, right = 1, 100  # Assume max level of 100
    current_level = 1

    while left <= right:
        mid = (left + right) // 2
        xp_needed = cumulative_xp_for_level(mid)

        if xp_needed <= total_xp:
            current_level = mid
            left = mid + 1
        else:
            right = mid - 1

    return current_level


def xp_to_next_level(total_xp: int) -> int:
    """
    Calculate how much more XP is needed to reach the next level.

    Args:
        total_xp: Current total XP

    Returns:
        XP needed to reach next level

    Examples:
        >>> xp_to_next_level(0)
        100
        >>> xp_to_next_level(50)   # 50/100 to level 2
        50
        >>> xp_to_next_level(100)  # Just reached level 2, now need 150 to reach level 3
        150
        >>> xp_to_next_level(200)  # At level 2 with 50 extra XP (need 100 more for level 3)
        100
    """
    current_level = calculate_level_from_xp(total_xp)
    next_level = current_level + 1

    xp_needed_for_next = cumulative_xp_for_level(next_level)

    return xp_needed_for_next - total_xp


def xp_progress_in_current_level(total_xp: int) -> tuple[int, int]:
    """
    Calculate XP progress within the current level.

    Args:
        total_xp: Current total XP

    Returns:
        Tuple of (current_xp_in_level, xp_required_for_level)

    Examples:
        >>> xp_progress_in_current_level(0)
        (0, 100)
        >>> xp_progress_in_current_level(50)
        (50, 100)
        >>> xp_progress_in_current_level(100)  # Just reached level 2
        (0, 150)
        >>> xp_progress_in_current_level(150)  # 50 XP into level 2
        (50, 150)
    """
    current_level = calculate_level_from_xp(total_xp)
    xp_at_current_level_start = cumulative_xp_for_level(current_level)
    xp_in_current_level = total_xp - xp_at_current_level_start
    xp_required_for_next = xp_for_level(current_level)

    return (xp_in_current_level, xp_required_for_next)


# Pre-generate a lookup table for quick reference (levels 1-100)
LEVEL_XP_TABLE = {
    level: {
        "xp_for_next_level": xp_for_level(level),
        "cumulative_xp": cumulative_xp_for_level(level),
    }
    for level in range(1, 101)
}


if __name__ == "__main__":
    # Test and display level progression
    print("=" * 60)
    print("PROGRESSIVE LEVEL SYSTEM - XP REQUIREMENTS")
    print("=" * 60)
    print(f"{'Level':<8} {'XP Needed':<12} {'Cumulative XP':<15} {'Progress':<15}")
    print("-" * 60)

    for level in range(1, 21):
        cumulative = cumulative_xp_for_level(level)

        if level == 1:
            print(f"{level:<8} {'START':<12} {cumulative:<15} {'-':<15}")
        else:
            xp_needed = xp_for_level(level - 1)  # XP needed to reach THIS level from previous
            print(f"{level:<8} {xp_needed:<12} {cumulative:<15} {f'+{xp_needed - BASE_XP}':<15}")

    print("=" * 60)
    print("\nTest cases:")
    print(f"Level at 0 XP: {calculate_level_from_xp(0)} (Expected: 1)")
    print(f"Level at 100 XP: {calculate_level_from_xp(100)} (Expected: 2)")
    print(f"Level at 250 XP: {calculate_level_from_xp(250)} (Expected: 3)")
    print(f"Level at 700 XP: {calculate_level_from_xp(700)} (Expected: 5)")
    print(f"XP to next level at 50 XP: {xp_to_next_level(50)} (Expected: 50)")
    print(f"XP to next level at 100 XP: {xp_to_next_level(100)} (Expected: 150)")

    progress, required = xp_progress_in_current_level(150)
    print(f"Progress at 150 XP: {progress}/{required} (Expected: 50/150)")
