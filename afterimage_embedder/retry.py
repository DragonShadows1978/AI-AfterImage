"""Retry Logic for AfterImage Embedding Daemon with exponential backoff."""

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class FailedEntry:
    entry_id: str
    file_path: str
    first_failure: str
    last_failure: str
    last_error: str
    attempt_count: int = 1
    next_retry_time: float = 0.0
    permanently_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailedEntry":
        return cls(**data)


class RetryManager:
    """Manages retry logic for failed embedding operations with exponential backoff."""

    def __init__(
        self,
        state_file: Path,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ):
        self.state_file = state_file
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._lock = Lock()
        self._failed_entries: Dict[str, FailedEntry] = {}
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                for entry_id, entry_data in data.get("failed_entries", {}).items():
                    self._failed_entries[entry_id] = FailedEntry.from_dict(entry_data)
                logger.info(f"Loaded {len(self._failed_entries)} entries from retry state")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load retry state: {e}")
                self._failed_entries = {}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "failed_entries": {
                entry_id: entry.to_dict()
                for entry_id, entry in self._failed_entries.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def calculate_backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay

    def record_failure(self, entry_id: str, file_path: str, error: str) -> bool:
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            current_time = time.time()
            if entry_id in self._failed_entries:
                entry = self._failed_entries[entry_id]
                entry.attempt_count += 1
                entry.last_failure = now
                entry.last_error = error
            else:
                entry = FailedEntry(
                    entry_id=entry_id, file_path=file_path,
                    first_failure=now, last_failure=now,
                    last_error=error, attempt_count=1
                )
                self._failed_entries[entry_id] = entry
            if entry.attempt_count >= self.max_attempts:
                entry.permanently_failed = True
                entry.next_retry_time = 0
                logger.warning(f"Entry {entry_id} permanently failed after {entry.attempt_count} attempts")
                self._save_state()
                return False
            backoff = self.calculate_backoff(entry.attempt_count)
            entry.next_retry_time = current_time + backoff
            logger.debug(f"Entry {entry_id} scheduled for retry in {backoff:.1f}s")
            self._save_state()
            return True

    def record_success(self, entry_id: str):
        with self._lock:
            if entry_id in self._failed_entries:
                del self._failed_entries[entry_id]
                self._save_state()

    def get_entries_due_for_retry(self) -> List[str]:
        current_time = time.time()
        with self._lock:
            return [
                entry_id for entry_id, entry in self._failed_entries.items()
                if not entry.permanently_failed and entry.next_retry_time <= current_time
            ]

    def is_in_retry_queue(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._failed_entries.get(entry_id)
            return entry is not None and not entry.permanently_failed

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            retrying = sum(1 for e in self._failed_entries.values() if not e.permanently_failed)
            permanently_failed = sum(1 for e in self._failed_entries.values() if e.permanently_failed)
            total_attempts = sum(e.attempt_count for e in self._failed_entries.values())
            return {
                "entries_retrying": retrying,
                "entries_permanently_failed": permanently_failed,
                "total_retry_attempts": total_attempts,
                "total_tracked": len(self._failed_entries)
            }

    def clear_permanently_failed(self) -> int:
        with self._lock:
            to_remove = [eid for eid, e in self._failed_entries.items() if e.permanently_failed]
            for entry_id in to_remove:
                del self._failed_entries[entry_id]
            if to_remove:
                self._save_state()
            return len(to_remove)

    def get_permanently_failed_entries(self) -> List[FailedEntry]:
        with self._lock:
            return [e for e in self._failed_entries.values() if e.permanently_failed]
