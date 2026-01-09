"""
AfterImage Churn Tracking Module.

Provides code churn statistics, file stability tiers, and change classification
for intelligent pre-write warnings.

Key Components:
- ChurnTracker: Main orchestrator for tracking and querying churn stats
- ChangeClassifier: AST/regex-based code change classification
- ChurnTier: Gold/Silver/Bronze/Red stability tiers
- ChurnWarning: Warning generation for concerning patterns

Usage:
    from afterimage.churn import ChurnTracker, ChurnTier

    tracker = ChurnTracker()
    tracker.initialize()

    # Check for warnings before write
    warning = tracker.get_warning(file_path, new_code, old_code, session_id)
    if warning:
        print(warning.format_message())

    # Record edit after write
    result = tracker.record_edit(file_path, old_code, new_code, session_id)

    # Get statistics
    stats = tracker.get_file_stats(file_path)
    print(f"Tier: {stats.tier.value}, Edits: {stats.total_edits}")

    # Get hotspots
    hotspots = tracker.get_hotspots(limit=10)
    for stats, score in hotspots:
        print(f"{stats.file_path}: {score:.1f}")
"""

from .storage import (
    ChurnTier,
    ChangeType,
    FileChurnStats,
    FunctionChurnStats,
    FunctionInfo,
    EditRecord,
    ChurnWarning,
    ChangeResult,
)

from .classifier import ChangeClassifier

from .tiers import (
    calculate_tier,
    should_warn_gold_tier,
    should_warn_repetitive_function,
    should_warn_red_tier,
    get_tier_description,
    get_tier_emoji,
    format_tier_badge,
    calculate_churn_velocity,
    suggest_action,
    rank_hotspots,
    TIER_THRESHOLDS,
    REPETITIVE_FUNCTION_THRESHOLD,
    HIGH_CHURN_24H_THRESHOLD,
)

from .tracker import ChurnTracker


__all__ = [
    # Main classes
    "ChurnTracker",
    "ChangeClassifier",

    # Data types
    "ChurnTier",
    "ChangeType",
    "FileChurnStats",
    "FunctionChurnStats",
    "FunctionInfo",
    "EditRecord",
    "ChurnWarning",
    "ChangeResult",

    # Tier functions
    "calculate_tier",
    "should_warn_gold_tier",
    "should_warn_repetitive_function",
    "should_warn_red_tier",
    "get_tier_description",
    "get_tier_emoji",
    "format_tier_badge",
    "calculate_churn_velocity",
    "suggest_action",
    "rank_hotspots",

    # Constants
    "TIER_THRESHOLDS",
    "REPETITIVE_FUNCTION_THRESHOLD",
    "HIGH_CHURN_24H_THRESHOLD",
]
