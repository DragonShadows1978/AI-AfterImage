"""
Churn tracker - the main orchestrator for code churn tracking.

Coordinates between storage, classification, and tier calculation
to track and report on code modification patterns.
"""

import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .storage import (
    ChurnTier, ChangeType, FileChurnStats, FunctionChurnStats,
    FunctionInfo, EditRecord, ChurnWarning, ChangeResult
)
from .classifier import ChangeClassifier
from .tiers import (
    calculate_tier, should_warn_gold_tier, should_warn_repetitive_function,
    should_warn_red_tier, format_tier_badge, get_tier_description,
    calculate_churn_velocity, suggest_action, rank_hotspots
)


class ChurnTracker:
    """
    Main tracker for code churn statistics.

    Tracks file and function-level edit history, calculates tiers,
    and generates warnings for concerning patterns.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize churn tracker.

        Args:
            db_path: Path to churn database. Defaults to ~/.afterimage/churn.db
        """
        if db_path is None:
            afterimage_dir = Path.home() / ".afterimage"
            afterimage_dir.mkdir(exist_ok=True)
            db_path = afterimage_dir / "churn.db"

        self.db_path = Path(db_path)
        self.classifier = ChangeClassifier()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database schema for churn tracking."""
        if self._initialized:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # File-level churn stats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_churn_stats (
                file_path TEXT PRIMARY KEY,
                total_edits INTEGER DEFAULT 0,
                first_edit TEXT,
                last_edit TEXT,
                tier TEXT DEFAULT 'silver'
            )
        """)

        # Function-level churn stats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS function_churn_stats (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                function_name TEXT,
                signature_hash TEXT,
                edit_count INTEGER DEFAULT 0,
                last_edit TEXT,
                UNIQUE(file_path, signature_hash)
            )
        """)

        # Edit history (for time-window queries)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edit_history (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                function_name TEXT,
                signature_hash TEXT,
                change_type TEXT,
                session_id TEXT,
                timestamp TEXT
            )
        """)

        # Indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edit_history_file_time
            ON edit_history(file_path, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_edit_history_func
            ON edit_history(signature_hash, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_function_churn_file
            ON function_churn_stats(file_path)
        """)

        conn.commit()
        conn.close()
        self._initialized = True

    def _ensure_init(self):
        """Ensure database is initialized."""
        if not self._initialized:
            self.initialize()

    def record_edit(
        self,
        file_path: str,
        old_code: Optional[str],
        new_code: str,
        session_id: str
    ) -> ChangeResult:
        """
        Record an edit and update statistics.

        Args:
            file_path: Path to the file being edited
            old_code: Previous content (None for new files)
            new_code: New content
            session_id: Claude session ID

        Returns:
            ChangeResult with classification details
        """
        self._ensure_init()

        now = datetime.now(timezone.utc).isoformat()

        # Classify the change
        result = self.classifier.classify_change(file_path, old_code, new_code)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Update file stats
            self._update_file_stats(cursor, file_path, now)

            # Record function-level edits
            all_functions = (
                result.functions_added +
                result.functions_modified +
                result.functions_deleted
            )

            for func in all_functions:
                change_type = (
                    ChangeType.ADD if func in result.functions_added else
                    ChangeType.DELETE if func in result.functions_deleted else
                    ChangeType.MODIFY
                )
                self._record_function_edit(
                    cursor, file_path, func, change_type, session_id, now
                )

            # If no functions detected, still record file-level edit
            if not all_functions:
                self._record_file_edit(
                    cursor, file_path, result.change_type, session_id, now
                )

            conn.commit()
        finally:
            conn.close()

        return result

    def _update_file_stats(
        self,
        cursor: sqlite3.Cursor,
        file_path: str,
        timestamp: str
    ):
        """Update file-level statistics."""
        # Check if file exists
        cursor.execute(
            "SELECT total_edits, first_edit FROM file_churn_stats WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()

        if row:
            total_edits = row[0] + 1
            first_edit = row[1]
            cursor.execute("""
                UPDATE file_churn_stats
                SET total_edits = ?, last_edit = ?
                WHERE file_path = ?
            """, (total_edits, timestamp, file_path))
        else:
            cursor.execute("""
                INSERT INTO file_churn_stats (file_path, total_edits, first_edit, last_edit, tier)
                VALUES (?, 1, ?, ?, 'silver')
            """, (file_path, timestamp, timestamp))

        # Recalculate tier
        stats = self._get_file_stats_from_cursor(cursor, file_path)
        new_tier = calculate_tier(stats)
        cursor.execute(
            "UPDATE file_churn_stats SET tier = ? WHERE file_path = ?",
            (new_tier.value, file_path)
        )

    def _record_function_edit(
        self,
        cursor: sqlite3.Cursor,
        file_path: str,
        func: FunctionInfo,
        change_type: ChangeType,
        session_id: str,
        timestamp: str
    ):
        """Record a function-level edit."""
        sig_hash = func.signature_hash()
        edit_id = str(uuid.uuid4())

        # Update function stats
        cursor.execute("""
            INSERT INTO function_churn_stats (id, file_path, function_name, signature_hash, edit_count, last_edit)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(file_path, signature_hash) DO UPDATE SET
                edit_count = edit_count + 1,
                last_edit = excluded.last_edit
        """, (str(uuid.uuid4()), file_path, func.name, sig_hash, timestamp))

        # Record in history
        cursor.execute("""
            INSERT INTO edit_history (id, file_path, function_name, signature_hash, change_type, session_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (edit_id, file_path, func.name, sig_hash, change_type.value, session_id, timestamp))

    def _record_file_edit(
        self,
        cursor: sqlite3.Cursor,
        file_path: str,
        change_type: ChangeType,
        session_id: str,
        timestamp: str
    ):
        """Record a file-level edit (no function detected)."""
        edit_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO edit_history (id, file_path, function_name, signature_hash, change_type, session_id, timestamp)
            VALUES (?, ?, NULL, NULL, ?, ?, ?)
        """, (edit_id, file_path, change_type.value, session_id, timestamp))

    def _get_file_stats_from_cursor(
        self,
        cursor: sqlite3.Cursor,
        file_path: str
    ) -> FileChurnStats:
        """Get file stats with time-window calculations."""
        cursor.execute(
            "SELECT * FROM file_churn_stats WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()

        if not row:
            return FileChurnStats(file_path=file_path)

        # Calculate time-window edits
        now = datetime.now(timezone.utc)
        day_ago = (now - timedelta(days=1)).isoformat()
        week_ago = (now - timedelta(days=7)).isoformat()
        month_ago = (now - timedelta(days=30)).isoformat()

        cursor.execute("""
            SELECT
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END) as edits_24h,
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END) as edits_7d,
                SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END) as edits_30d
            FROM edit_history
            WHERE file_path = ?
        """, (day_ago, week_ago, month_ago, file_path))
        time_stats = cursor.fetchone()

        tier = ChurnTier(row[4]) if row[4] else ChurnTier.SILVER

        return FileChurnStats(
            file_path=file_path,
            total_edits=row[1] or 0,
            edits_last_24h=time_stats[0] or 0,
            edits_last_7d=time_stats[1] or 0,
            edits_last_30d=time_stats[2] or 0,
            first_edit=row[2],
            last_edit=row[3],
            tier=tier,
        )

    def get_file_stats(self, file_path: str) -> FileChurnStats:
        """Get churn statistics for a file."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            stats = self._get_file_stats_from_cursor(cursor, file_path)
            # Recalculate tier
            stats.tier = calculate_tier(stats)
            return stats
        finally:
            conn.close()

    def get_function_stats(self, file_path: str) -> List[FunctionChurnStats]:
        """Get per-function churn statistics for a file."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT function_name, signature_hash, edit_count, last_edit
            FROM function_churn_stats
            WHERE file_path = ?
            ORDER BY edit_count DESC
        """, (file_path,))

        results = []
        for row in cursor.fetchall():
            # Get change types from history
            cursor.execute("""
                SELECT change_type FROM edit_history
                WHERE file_path = ? AND signature_hash = ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (file_path, row[1]))
            change_types = [r[0] for r in cursor.fetchall()]

            results.append(FunctionChurnStats(
                file_path=file_path,
                function_name=row[0],
                signature_hash=row[1],
                edit_count=row[2],
                last_edit=row[3],
                change_types=change_types,
            ))

        conn.close()
        return results

    def get_function_edits_24h(
        self,
        file_path: str,
        signature_hash: str
    ) -> int:
        """Get number of edits to a function in the last 24 hours."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        cursor.execute("""
            SELECT COUNT(*) FROM edit_history
            WHERE file_path = ? AND signature_hash = ? AND timestamp >= ?
        """, (file_path, signature_hash, day_ago))

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_warning(
        self,
        file_path: str,
        new_code: str,
        old_code: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[ChurnWarning]:
        """
        Check if an edit should trigger a churn warning.

        Warns if:
        - Gold tier file being modified
        - Same function modified >3x in 24h
        - Red tier file
        - New session editing high-churn file

        Args:
            file_path: Path to file being edited
            new_code: New content
            old_code: Previous content (for classification)
            session_id: Current session ID

        Returns:
            ChurnWarning if warning should be shown, None otherwise
        """
        self._ensure_init()

        stats = self.get_file_stats(file_path)

        # Check Gold tier warning
        if should_warn_gold_tier(stats):
            return self._format_gold_tier_warning(stats)

        # Check Red tier warning
        if should_warn_red_tier(stats):
            return self._format_red_tier_warning(stats)

        # Check repetitive function edits
        if old_code is not None:
            result = self.classifier.classify_change(file_path, old_code, new_code)
            for func in result.functions_modified:
                edits_24h = self.get_function_edits_24h(file_path, func.signature_hash())
                if should_warn_repetitive_function(
                    FunctionChurnStats(
                        file_path=file_path,
                        function_name=func.name,
                        signature_hash=func.signature_hash(),
                        edit_count=edits_24h,
                    ),
                    edits_24h
                ):
                    return self._format_repetitive_warning(file_path, func, edits_24h)

        return None

    def _format_gold_tier_warning(self, stats: FileChurnStats) -> ChurnWarning:
        """Format warning for Gold tier file modification."""
        lines = [
            f"File: {stats.file_path}",
            f"Tier: {format_tier_badge(ChurnTier.GOLD)}",
            "",
            "This file is stable and rarely modified.",
            f"  - Total edits: {stats.total_edits}",
            f"  - Last edit: {stats.last_edit[:19] if stats.last_edit else 'Unknown'}",
            "",
            "Are you sure this change is necessary?",
            "",
            "If intentional, retry your write.",
        ]

        return ChurnWarning(
            warning_type="gold_tier",
            file_path=stats.file_path,
            message="\n".join(lines),
            details=stats.to_dict(),
            severity="warn",
        )

    def _format_red_tier_warning(self, stats: FileChurnStats) -> ChurnWarning:
        """Format warning for Red tier file."""
        func_stats = self.get_function_stats(stats.file_path)
        top_functions = func_stats[:3]

        lines = [
            f"File: {stats.file_path}",
            f"Tier: {format_tier_badge(ChurnTier.RED)}",
            "",
            f"This file has excessive churn ({stats.edits_last_30d} edits in 30 days).",
            "",
        ]

        if top_functions:
            lines.append("Top modified functions:")
            for i, f in enumerate(top_functions, 1):
                lines.append(f"  {i}. {f.function_name}() - {f.edit_count} edits")
            lines.append("")

        suggestion = suggest_action(stats)
        if suggestion:
            lines.append(f"Suggestion: {suggestion}")
            lines.append("")

        lines.append("Retry your write if this is intentional.")

        return ChurnWarning(
            warning_type="red_tier",
            file_path=stats.file_path,
            message="\n".join(lines),
            details={
                "file_stats": stats.to_dict(),
                "top_functions": [f.to_dict() for f in top_functions],
            },
            severity="alert",
        )

    def _format_repetitive_warning(
        self,
        file_path: str,
        func: FunctionInfo,
        edits_24h: int
    ) -> ChurnWarning:
        """Format warning for repetitive function modification."""
        lines = [
            f"File: {file_path}",
            f"Function: {func.name}()",
            "",
            f"This function has been modified {edits_24h} times in the last 24 hours.",
            "",
            "Possible issues:",
            "  - Bug not fully fixed",
            "  - Requirements unclear",
            "  - Function needs redesign",
            "",
            "Consider stepping back to understand the root cause.",
            "",
            "Retry your write if this is intentional.",
        ]

        return ChurnWarning(
            warning_type="repetitive_function",
            file_path=file_path,
            message="\n".join(lines),
            details={
                "function_name": func.name,
                "edits_24h": edits_24h,
            },
            severity="warn",
        )

    def get_hotspots(self, limit: int = 20) -> List[Tuple[FileChurnStats, float]]:
        """Get files ranked by churn hotspot score."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT file_path FROM file_churn_stats")
        all_files = [row[0] for row in cursor.fetchall()]
        conn.close()

        all_stats = [self.get_file_stats(fp) for fp in all_files]
        return rank_hotspots(all_stats, limit)

    def get_files_by_tier(self, tier: ChurnTier) -> List[FileChurnStats]:
        """Get all files in a specific tier."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT file_path FROM file_churn_stats WHERE tier = ?",
            (tier.value,)
        )
        file_paths = [row[0] for row in cursor.fetchall()]
        conn.close()

        return [self.get_file_stats(fp) for fp in file_paths]

    def get_edit_history(
        self,
        file_path: str,
        limit: int = 20
    ) -> List[EditRecord]:
        """Get recent edit history for a file."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, file_path, function_name, signature_hash, change_type, session_id, timestamp
            FROM edit_history
            WHERE file_path = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (file_path, limit))

        results = []
        for row in cursor.fetchall():
            results.append(EditRecord(
                id=row[0],
                file_path=row[1],
                function_name=row[2],
                signature_hash=row[3],
                change_type=ChangeType(row[4]),
                session_id=row[5],
                timestamp=row[6],
            ))

        conn.close()
        return results

    def clear(self) -> int:
        """Clear all churn data. Returns count of deleted records."""
        self._ensure_init()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM edit_history")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM edit_history")
        cursor.execute("DELETE FROM function_churn_stats")
        cursor.execute("DELETE FROM file_churn_stats")

        conn.commit()
        conn.close()
        return count

    def close(self):
        """Close tracker (no-op for SQLite, but here for interface consistency)."""
        pass
