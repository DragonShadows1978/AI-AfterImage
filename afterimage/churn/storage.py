"""
Churn statistics data models and storage interface.

Defines dataclasses for tracking file and function-level churn metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
import hashlib


class ChurnTier(Enum):
    """File stability tiers based on edit frequency."""
    GOLD = "gold"      # Stable, rarely changed (<3 edits/30d)
    SILVER = "silver"  # Normal activity (3-10 edits/30d)
    BRONZE = "bronze"  # High activity (11-20 edits/30d)
    RED = "red"        # Excessive churn (>20 edits OR >5 same-func edits/24h)


class ChangeType(Enum):
    """Types of code changes."""
    ADD = "add"           # New code added (new functions/classes)
    MODIFY = "modify"     # Existing code modified
    DELETE = "delete"     # Code removed
    REFACTOR = "refactor" # Structure changed but logic preserved


@dataclass
class FileChurnStats:
    """Statistics for a single file's edit history."""
    file_path: str
    total_edits: int = 0
    edits_last_24h: int = 0
    edits_last_7d: int = 0
    edits_last_30d: int = 0
    first_edit: Optional[str] = None  # ISO timestamp
    last_edit: Optional[str] = None   # ISO timestamp
    tier: ChurnTier = ChurnTier.SILVER

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "total_edits": self.total_edits,
            "edits_last_24h": self.edits_last_24h,
            "edits_last_7d": self.edits_last_7d,
            "edits_last_30d": self.edits_last_30d,
            "first_edit": self.first_edit,
            "last_edit": self.last_edit,
            "tier": self.tier.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileChurnStats":
        """Create from dictionary."""
        tier = ChurnTier(data.get("tier", "silver"))
        return cls(
            file_path=data["file_path"],
            total_edits=data.get("total_edits", 0),
            edits_last_24h=data.get("edits_last_24h", 0),
            edits_last_7d=data.get("edits_last_7d", 0),
            edits_last_30d=data.get("edits_last_30d", 0),
            first_edit=data.get("first_edit"),
            last_edit=data.get("last_edit"),
            tier=tier,
        )


@dataclass
class FunctionInfo:
    """Information about a function/method/class definition."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    kind: str  # "function", "class", "method"
    signature: str  # Full signature for hashing

    def signature_hash(self) -> str:
        """Generate stable hash of function signature.

        Normalizes whitespace to ensure stability.
        """
        # Normalize: lowercase, strip, collapse whitespace
        normalized = " ".join(self.signature.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class FunctionChurnStats:
    """Statistics for a single function's edit history."""
    file_path: str
    function_name: str
    signature_hash: str
    edit_count: int = 0
    last_edit: Optional[str] = None
    change_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "function_name": self.function_name,
            "signature_hash": self.signature_hash,
            "edit_count": self.edit_count,
            "last_edit": self.last_edit,
            "change_types": self.change_types,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionChurnStats":
        """Create from dictionary."""
        return cls(
            file_path=data["file_path"],
            function_name=data["function_name"],
            signature_hash=data["signature_hash"],
            edit_count=data.get("edit_count", 0),
            last_edit=data.get("last_edit"),
            change_types=data.get("change_types", []),
        )


@dataclass
class EditRecord:
    """Record of a single edit operation."""
    id: str
    file_path: str
    function_name: Optional[str]
    signature_hash: Optional[str]
    change_type: ChangeType
    session_id: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "signature_hash": self.signature_hash,
            "change_type": self.change_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ChurnWarning:
    """Warning generated when churn thresholds are exceeded."""
    warning_type: str  # "gold_tier", "repetitive_function", "red_tier"
    file_path: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warn"  # "warn", "alert", "critical"

    def format_message(self) -> str:
        """Format warning as a human-readable message."""
        lines = [
            "",
            "=" * 60,
            f"AFTERIMAGE CHURN ALERT: {self._get_title()}",
            "=" * 60,
            "",
        ]
        lines.append(self.message)
        lines.append("")
        lines.append("=" * 60)
        lines.append("")
        return "\n".join(lines)

    def _get_title(self) -> str:
        """Get warning title based on type."""
        titles = {
            "gold_tier": "Stable File Modification",
            "repetitive_function": "Repetitive Function Modification",
            "red_tier": "High Churn File",
            "new_session": "New Session Edit",
        }
        return titles.get(self.warning_type, "Code Churn Warning")


@dataclass
class ChangeResult:
    """Result of classifying a code change."""
    change_type: ChangeType
    functions_added: List[FunctionInfo] = field(default_factory=list)
    functions_modified: List[FunctionInfo] = field(default_factory=list)
    functions_deleted: List[FunctionInfo] = field(default_factory=list)
    is_new_session_edit: bool = False

    @property
    def is_purely_additive(self) -> bool:
        """True if change only adds code (no modifications/deletions)."""
        return (
            len(self.functions_added) > 0 and
            len(self.functions_modified) == 0 and
            len(self.functions_deleted) == 0
        )

    @property
    def has_modifications(self) -> bool:
        """True if existing code was modified."""
        return len(self.functions_modified) > 0

    @property
    def total_changes(self) -> int:
        """Total number of function-level changes."""
        return (
            len(self.functions_added) +
            len(self.functions_modified) +
            len(self.functions_deleted)
        )
