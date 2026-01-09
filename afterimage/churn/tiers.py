"""
Tier calculation system for file stability classification.

Determines Gold/Silver/Bronze/Red tiers based on edit frequency
and patterns over time windows.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

from .storage import ChurnTier, FileChurnStats, FunctionChurnStats


# Tier thresholds (edits in 30-day window)
TIER_THRESHOLDS = {
    ChurnTier.GOLD: (0, 2),      # 0-2 edits = stable
    ChurnTier.SILVER: (3, 10),   # 3-10 edits = normal
    ChurnTier.BRONZE: (11, 20),  # 11-20 edits = high activity
    ChurnTier.RED: (21, float('inf')),  # >20 edits = excessive
}

# Red flag: same function edited this many times in 24h
REPETITIVE_FUNCTION_THRESHOLD = 3

# Red flag: file touched in 24h while being high-churn
HIGH_CHURN_24H_THRESHOLD = 5


def calculate_tier(stats: FileChurnStats) -> ChurnTier:
    """
    Calculate the stability tier for a file.

    Tier Rules:
    - Gold: <3 edits in 30 days (stable, rarely changed)
    - Silver: 3-10 edits in 30 days (normal activity)
    - Bronze: 11-20 edits in 30 days (high activity)
    - Red: >20 edits in 30 days OR excessive 24h activity

    Args:
        stats: File churn statistics

    Returns:
        Calculated ChurnTier
    """
    edits_30d = stats.edits_last_30d

    # Check for red-tier conditions first
    if edits_30d > 20:
        return ChurnTier.RED

    # Additional red flag: too many edits in 24h
    if stats.edits_last_24h >= HIGH_CHURN_24H_THRESHOLD:
        return ChurnTier.RED

    # Check tier thresholds
    for tier, (low, high) in TIER_THRESHOLDS.items():
        if low <= edits_30d <= high:
            return tier

    return ChurnTier.SILVER


def should_warn_gold_tier(
    stats: FileChurnStats,
    current_session_id: Optional[str] = None,
    last_session_id: Optional[str] = None
) -> bool:
    """
    Check if modifying a Gold-tier file should trigger a warning.

    Gold-tier files are stable and rarely modified. A warning helps
    prevent accidental changes to well-tested, stable code.

    Args:
        stats: File churn statistics
        current_session_id: Current Claude session ID
        last_session_id: Session ID of the last edit

    Returns:
        True if a warning should be shown
    """
    if stats.tier != ChurnTier.GOLD:
        return False

    # Always warn for gold tier files
    return True


def should_warn_repetitive_function(
    func_stats: FunctionChurnStats,
    edits_in_24h: int
) -> bool:
    """
    Check if a function is being modified too frequently.

    Repetitive modifications to the same function often indicate:
    - Bug not fully fixed
    - Requirements unclear
    - Function needs redesign

    Args:
        func_stats: Function churn statistics
        edits_in_24h: Number of times this function was edited in last 24h

    Returns:
        True if a warning should be shown
    """
    return edits_in_24h >= REPETITIVE_FUNCTION_THRESHOLD


def should_warn_red_tier(stats: FileChurnStats) -> bool:
    """
    Check if a Red-tier file should trigger a warning.

    Red-tier files have excessive churn and may indicate:
    - Poorly designed code
    - Frequently changing requirements
    - Technical debt

    Args:
        stats: File churn statistics

    Returns:
        True if a warning should be shown
    """
    return stats.tier == ChurnTier.RED


def get_tier_description(tier: ChurnTier) -> str:
    """Get a human-readable description of a tier."""
    descriptions = {
        ChurnTier.GOLD: "Stable - rarely changed",
        ChurnTier.SILVER: "Normal - moderate activity",
        ChurnTier.BRONZE: "Active - high modification rate",
        ChurnTier.RED: "Hot - excessive churn detected",
    }
    return descriptions.get(tier, "Unknown tier")


def get_tier_emoji(tier: ChurnTier) -> str:
    """Get emoji representation of a tier."""
    emojis = {
        ChurnTier.GOLD: "🥇",
        ChurnTier.SILVER: "🥈",
        ChurnTier.BRONZE: "🥉",
        ChurnTier.RED: "🔴",
    }
    return emojis.get(tier, "⚪")


def format_tier_badge(tier: ChurnTier) -> str:
    """Format tier as a display badge."""
    emoji = get_tier_emoji(tier)
    desc = get_tier_description(tier)
    return f"{emoji} {tier.value.upper()} ({desc})"


def calculate_churn_velocity(
    edits_24h: int,
    edits_7d: int,
    edits_30d: int
) -> float:
    """
    Calculate churn velocity - rate of change acceleration.

    Higher values indicate increasing churn (bad trend).
    Lower values indicate stabilizing churn (good trend).

    Returns:
        Velocity factor (>1 = accelerating, <1 = decelerating)
    """
    # Avoid division by zero
    if edits_30d == 0:
        return 0.0

    # Normalize to daily rates
    daily_rate_24h = edits_24h
    daily_rate_7d = edits_7d / 7
    daily_rate_30d = edits_30d / 30

    # Calculate trend
    if daily_rate_30d == 0:
        return 1.0 if daily_rate_24h > 0 else 0.0

    # Recent activity compared to baseline
    velocity = daily_rate_24h / daily_rate_30d
    return min(velocity, 10.0)  # Cap at 10x


def suggest_action(stats: FileChurnStats) -> Optional[str]:
    """
    Suggest an action based on file churn statistics.

    Returns:
        Suggested action string, or None if no action needed
    """
    if stats.tier == ChurnTier.RED:
        return "Consider refactoring this file - it has excessive churn"

    if stats.tier == ChurnTier.BRONZE:
        velocity = calculate_churn_velocity(
            stats.edits_last_24h,
            stats.edits_last_7d,
            stats.edits_last_30d
        )
        if velocity > 2.0:
            return "Churn is accelerating - consider pausing to stabilize"

    if stats.tier == ChurnTier.GOLD:
        return "This file is stable - consider if changes are truly necessary"

    return None


def rank_hotspots(
    file_stats: List[FileChurnStats],
    limit: int = 20
) -> List[Tuple[FileChurnStats, float]]:
    """
    Rank files by churn hotspot score.

    Considers both total edits and recent velocity.

    Args:
        file_stats: List of file statistics
        limit: Maximum files to return

    Returns:
        List of (stats, score) tuples, sorted by score descending
    """
    scored = []
    for stats in file_stats:
        # Base score from 30-day edits
        base_score = stats.edits_last_30d

        # Boost for recent activity
        velocity = calculate_churn_velocity(
            stats.edits_last_24h,
            stats.edits_last_7d,
            stats.edits_last_30d
        )
        recency_boost = 1.0 + (velocity * 0.5)

        # Tier penalty for gold (we don't want gold files in hotspots)
        tier_factor = {
            ChurnTier.GOLD: 0.1,
            ChurnTier.SILVER: 0.5,
            ChurnTier.BRONZE: 1.0,
            ChurnTier.RED: 1.5,
        }.get(stats.tier, 1.0)

        final_score = base_score * recency_boost * tier_factor
        scored.append((stats, final_score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]
