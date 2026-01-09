# AfterImage Churn Tracking

## Overview

The Churn Tracking module provides intelligent pre-write warnings based on file and function-level edit patterns. It helps prevent excessive modifications to stable code and identifies problematic churn patterns.

**Version:** 0.3.0

## Features

- **File Stability Tiers**: Gold/Silver/Bronze/Red classification based on edit frequency
- **Function-Level Tracking**: AST/regex-based detection of individual function modifications
- **Pre-Write Warnings**: Contextual alerts before modifying stable or high-churn files
- **Churn Hotspots**: Identify files with the highest modification frequency
- **CLI Integration**: `afterimage churn`, `afterimage hotspots`, `afterimage files --tier`
- **SQLite Storage**: Persistent edit history with time-windowed statistics

## Quick Start

```python
from afterimage.churn import ChurnTracker, ChurnTier

# Initialize tracker
tracker = ChurnTracker()
tracker.initialize()

# Check for warnings before write
warning = tracker.get_warning(
    file_path="/project/src/auth.py",
    new_code="def login(): pass",
    old_code=None,
    session_id="session_123"
)
if warning:
    print(warning.format_message())

# Record edit after write
tracker.record_edit(
    file_path="/project/src/auth.py",
    old_code=None,
    new_code="def login(): pass",
    session_id="session_123"
)

# Get file statistics
stats = tracker.get_file_stats("/project/src/auth.py")
print(f"Tier: {stats.tier.value}, Edits: {stats.total_edits}")

# Get hotspots
hotspots = tracker.get_hotspots(limit=10)
for stats, score in hotspots:
    print(f"{stats.file_path}: {score:.1f}")
```

## Stability Tiers

Files are classified into four stability tiers based on edit frequency over a 30-day window:

| Tier | Edits (30d) | Description | Emoji |
|------|-------------|-------------|-------|
| **Gold** | 0-2 | Stable, rarely changed | :1st_place_medal: |
| **Silver** | 3-10 | Normal activity | :2nd_place_medal: |
| **Bronze** | 11-20 | High activity | :3rd_place_medal: |
| **Red** | >20 OR >5 in 24h | Excessive churn | :red_circle: |

### Tier Warnings

- **Gold Tier**: "This file is stable - consider if changes are truly necessary"
- **Bronze Tier** (accelerating): "Churn is accelerating - consider pausing to stabilize"
- **Red Tier**: "Consider refactoring this file - it has excessive churn"

## Warning Types

### 1. Gold Tier Warning

Triggered when modifying a file with <3 edits in the last 30 days. These files are considered stable and well-tested.

```
============================================================
AFTERIMAGE CHURN ALERT: Stable File Modification
============================================================

This file (/project/src/auth.py) is GOLD tier - stable code
that hasn't been changed recently.

Stats: 2 edits in 30 days
Last edit: 2026-01-01T12:00:00Z

Consider if this change is truly necessary.
============================================================
```

### 2. Repetitive Function Warning

Triggered when the same function is modified 3+ times within 24 hours. This often indicates:
- Bug not fully fixed
- Requirements unclear
- Function needs redesign

```
============================================================
AFTERIMAGE CHURN ALERT: Repetitive Function Modification
============================================================

The function 'validate_email' in /project/src/validators.py
has been modified 4 times in the last 24 hours.

This may indicate a bug that isn't fully fixed or unclear
requirements. Consider stepping back to understand the issue.
============================================================
```

### 3. Red Tier Warning

Triggered for files with excessive churn (>20 edits in 30 days OR >5 edits in 24 hours).

```
============================================================
AFTERIMAGE CHURN ALERT: High Churn File
============================================================

This file (/project/src/utils.py) is RED tier with excessive
modification frequency.

Stats: 25 edits in 30 days, 6 edits today
Velocity: 2.5x (accelerating)

Consider refactoring or splitting this file.
============================================================
```

## CLI Commands

### Check File Churn

```bash
# View churn stats for a specific file
afterimage churn /project/src/auth.py

# JSON output
afterimage churn /project/src/auth.py --json
```

Output:
```
File: /project/src/auth.py
Tier: :2nd_place_medal: SILVER (Normal - moderate activity)
Total Edits: 7
Last 24h: 1 | Last 7d: 3 | Last 30d: 7
Last Edit: 2026-01-09T10:30:00Z
```

### View Hotspots

```bash
# Top 10 churn hotspots
afterimage hotspots

# Limit results
afterimage hotspots --limit 20

# JSON output
afterimage hotspots --json
```

Output:
```
Churn Hotspots (Top 10)
━━━━━━━━━━━━━━━━━━━━━━━━
1. :red_circle: /project/src/utils.py          Score: 45.2  (25 edits)
2. :3rd_place_medal: /project/src/api.py             Score: 22.1  (15 edits)
3. :2nd_place_medal: /project/src/models.py          Score: 10.5  (8 edits)
...
```

### List Files by Tier

```bash
# List all gold-tier files
afterimage files --tier gold

# List red-tier files
afterimage files --tier red

# JSON output
afterimage files --tier bronze --json
```

## API Reference

### ChurnTracker

Main orchestrator for tracking and querying churn statistics.

```python
class ChurnTracker:
    def initialize(self) -> None:
        """Initialize the SQLite database."""

    def record_edit(
        self,
        file_path: str,
        old_code: Optional[str],
        new_code: str,
        session_id: str
    ) -> ChangeResult:
        """Record an edit operation."""

    def get_warning(
        self,
        file_path: str,
        new_code: str,
        old_code: Optional[str],
        session_id: str
    ) -> Optional[ChurnWarning]:
        """Check if a write should trigger a warning."""

    def get_file_stats(self, file_path: str) -> FileChurnStats:
        """Get statistics for a file."""

    def get_function_stats(
        self,
        file_path: str,
        function_name: str
    ) -> Optional[FunctionChurnStats]:
        """Get statistics for a specific function."""

    def get_hotspots(
        self,
        limit: int = 20
    ) -> List[Tuple[FileChurnStats, float]]:
        """Get files ranked by churn score."""

    def get_files_by_tier(
        self,
        tier: ChurnTier
    ) -> List[FileChurnStats]:
        """Get all files with a specific tier."""
```

### Data Models

#### ChurnTier (Enum)

```python
class ChurnTier(Enum):
    GOLD = "gold"      # <3 edits/30d
    SILVER = "silver"  # 3-10 edits/30d
    BRONZE = "bronze"  # 11-20 edits/30d
    RED = "red"        # >20 edits/30d OR >5/24h
```

#### FileChurnStats

```python
@dataclass
class FileChurnStats:
    file_path: str
    total_edits: int
    edits_last_24h: int
    edits_last_7d: int
    edits_last_30d: int
    first_edit: Optional[str]  # ISO timestamp
    last_edit: Optional[str]   # ISO timestamp
    tier: ChurnTier
```

#### FunctionChurnStats

```python
@dataclass
class FunctionChurnStats:
    file_path: str
    function_name: str
    signature_hash: str
    edit_count: int
    last_edit: Optional[str]
    change_types: List[str]
```

#### ChurnWarning

```python
@dataclass
class ChurnWarning:
    warning_type: str  # "gold_tier", "repetitive_function", "red_tier"
    file_path: str
    message: str
    details: Dict[str, Any]
    severity: str  # "warn", "alert", "critical"

    def format_message(self) -> str:
        """Format as human-readable message."""
```

#### ChangeResult

```python
@dataclass
class ChangeResult:
    change_type: ChangeType
    functions_added: List[FunctionInfo]
    functions_modified: List[FunctionInfo]
    functions_deleted: List[FunctionInfo]
    is_new_session_edit: bool
```

### Tier Functions

```python
from afterimage.churn import (
    calculate_tier,
    format_tier_badge,
    get_tier_description,
    get_tier_emoji,
    calculate_churn_velocity,
    suggest_action,
    rank_hotspots,
)

# Calculate tier from stats
tier = calculate_tier(file_stats)

# Format for display
badge = format_tier_badge(ChurnTier.GOLD)
# :1st_place_medal: GOLD (Stable - rarely changed)

# Get velocity (>1 = accelerating, <1 = decelerating)
velocity = calculate_churn_velocity(
    edits_24h=3,
    edits_7d=10,
    edits_30d=25
)

# Get suggested action
action = suggest_action(file_stats)
```

## Configuration

The churn tracker uses a SQLite database stored at `~/.afterimage/churn.db`.

### Thresholds

Default thresholds can be customized by modifying the constants in `afterimage/churn/tiers.py`:

```python
# Tier thresholds (edits in 30-day window)
TIER_THRESHOLDS = {
    ChurnTier.GOLD: (0, 2),
    ChurnTier.SILVER: (3, 10),
    ChurnTier.BRONZE: (11, 20),
    ChurnTier.RED: (21, float('inf')),
}

# Repetitive function warning threshold (edits in 24h)
REPETITIVE_FUNCTION_THRESHOLD = 3

# High churn 24h threshold
HIGH_CHURN_24H_THRESHOLD = 5
```

## Hook Integration

The churn tracking is automatically integrated into the AfterImage hook (`hooks/afterimage_hook.py`):

1. **Pre-Write**: Checks for churn warnings before allowing writes
2. **Post-Write**: Records the edit in the churn tracker

```python
# In hooks/afterimage_hook.py (simplified)

# Pre-Write: Check for warnings
if hook_event == "PreToolUse":
    warning = check_churn_warning(file_path, content, old_content)
    if warning:
        # Deny first attempt with warning context
        return {"permissionDecision": "deny", "reason": warning}

# Post-Write: Record edit
elif hook_event == "PostToolUse":
    record_churn_edit(file_path, new_code, old_code)
```

## Best Practices

1. **Gold Tier Files**: Think twice before modifying. If changes are needed, ensure thorough testing.

2. **Repetitive Function Edits**: If you're editing the same function multiple times:
   - Step back and understand the root cause
   - Consider if the function needs redesign
   - Document unclear requirements

3. **Red Tier Files**: Files with excessive churn often indicate:
   - Poor separation of concerns (split the file)
   - Missing abstractions (extract common patterns)
   - Technical debt (schedule refactoring time)

4. **Monitor Hotspots**: Regularly check `afterimage hotspots` to identify problematic areas before they become critical.

## ChangeClassifier

The `ChangeClassifier` uses AST parsing (Python) and regex patterns (other languages) to detect:

- **Functions Added**: New function/method definitions
- **Functions Modified**: Changes to existing function bodies
- **Functions Deleted**: Removed function definitions

Supported languages:
- Python (AST-based)
- JavaScript/TypeScript (regex-based)
- Go, Rust, C/C++ (regex-based)

## Performance

Churn checks are designed to be fast:
- Target latency: <50ms
- SQLite queries are indexed by file path
- Time-window calculations use efficient SQL aggregations
