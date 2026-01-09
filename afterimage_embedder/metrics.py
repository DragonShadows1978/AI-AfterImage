"""
Metrics collection for the AfterImage Embedding Daemon.

Tracks processing progress, throughput, embedding coverage, and rate limiter stats.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from threading import Lock


@dataclass
class CycleMetrics:
    """Metrics for a single reindex cycle."""
    cycle_id: int
    started_at: str
    completed_at: Optional[str] = None
    entries_processed: int = 0
    entries_skipped: int = 0
    entries_failed: int = 0
    priority_entries: int = 0
    batch_count: int = 0
    duration_seconds: float = 0.0
    avg_embedding_time_ms: float = 0.0
    device_used: str = "cpu"


@dataclass
class DaemonMetrics:
    """Overall daemon metrics."""
    daemon_started_at: str = ""
    last_cycle_at: Optional[str] = None
    total_cycles: int = 0
    total_entries_processed: int = 0
    total_entries_failed: int = 0
    current_coverage_percent: float = 0.0
    total_entries_in_kb: int = 0
    entries_with_embeddings: int = 0
    avg_entries_per_cycle: float = 0.0
    avg_cycle_duration_seconds: float = 0.0
    device: str = "cpu"
    model_name: str = ""
    active_embedding_dim: int = 0
    model_fallback_count: int = 0
    warmup_time_ms: float = 0.0
    warmup_count: int = 0
    entries_retrying: int = 0
    entries_permanently_failed: int = 0
    retry_attempts_total: int = 0
    retry_successes_total: int = 0
    retry_enabled: bool = False
    rate_limit_enabled: bool = False
    rate_limit_waits_total: int = 0
    rate_limit_timeouts_total: int = 0
    recent_cycles: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MetricsCollector:
    """Collects and persists daemon metrics."""

    def __init__(self, metrics_file: Path, max_recent_cycles: int = 20):
        self.metrics_file = metrics_file
        self.max_recent_cycles = max_recent_cycles
        self._lock = Lock()
        self._metrics = DaemonMetrics()
        self._current_cycle: Optional[CycleMetrics] = None
        self._cycle_start_time: float = 0
        self._embedding_times: List[float] = []
        self._load()

    def _load(self):
        """Load existing metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    data = json.load(f)
                    self._metrics = DaemonMetrics(
                        daemon_started_at=data.get("daemon_started_at", ""),
                        last_cycle_at=data.get("last_cycle_at"),
                        total_cycles=data.get("total_cycles", 0),
                        total_entries_processed=data.get("total_entries_processed", 0),
                        total_entries_failed=data.get("total_entries_failed", 0),
                        current_coverage_percent=data.get("current_coverage_percent", 0.0),
                        total_entries_in_kb=data.get("total_entries_in_kb", 0),
                        entries_with_embeddings=data.get("entries_with_embeddings", 0),
                        avg_entries_per_cycle=data.get("avg_entries_per_cycle", 0.0),
                        avg_cycle_duration_seconds=data.get("avg_cycle_duration_seconds", 0.0),
                        device=data.get("device", "cpu"),
                        model_name=data.get("model_name", ""),
                        active_embedding_dim=data.get("active_embedding_dim", 0),
                        model_fallback_count=data.get("model_fallback_count", 0),
                        recent_cycles=data.get("recent_cycles", []),
                        warmup_time_ms=data.get("warmup_time_ms", 0.0),
                        warmup_count=data.get("warmup_count", 0),
                        entries_retrying=data.get("entries_retrying", 0),
                        entries_permanently_failed=data.get("entries_permanently_failed", 0),
                        retry_attempts_total=data.get("retry_attempts_total", 0),
                        retry_successes_total=data.get("retry_successes_total", 0),
                        retry_enabled=data.get("retry_enabled", False),
                        rate_limit_enabled=data.get("rate_limit_enabled", False),
                        rate_limit_waits_total=data.get("rate_limit_waits_total", 0),
                        rate_limit_timeouts_total=data.get("rate_limit_timeouts_total", 0)
                    )
            except (json.JSONDecodeError, KeyError):
                self._metrics = DaemonMetrics()

    def _save(self):
        """Persist metrics to file."""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, "w") as f:
            json.dump(self._metrics.to_dict(), f, indent=2)

    def daemon_started(self, device: str, model_name: str, embedding_dim: int = 0, fallback_count: int = 0):
        """Record daemon startup."""
        with self._lock:
            self._metrics.daemon_started_at = datetime.now(timezone.utc).isoformat()
            self._metrics.device = device
            self._metrics.model_name = model_name
            self._metrics.active_embedding_dim = embedding_dim
            self._metrics.model_fallback_count = fallback_count
            self._save()

    def start_cycle(self) -> int:
        """Start a new reindex cycle. Returns cycle ID."""
        with self._lock:
            cycle_id = self._metrics.total_cycles + 1
            self._current_cycle = CycleMetrics(
                cycle_id=cycle_id,
                started_at=datetime.now(timezone.utc).isoformat(),
                device_used=self._metrics.device
            )
            self._cycle_start_time = time.time()
            self._embedding_times = []
            return cycle_id

    def record_batch(self, processed: int, skipped: int, failed: int,
                     priority: int, embedding_time_ms: float):
        """Record metrics for a processed batch."""
        with self._lock:
            if self._current_cycle:
                self._current_cycle.entries_processed += processed
                self._current_cycle.entries_skipped += skipped
                self._current_cycle.entries_failed += failed
                self._current_cycle.priority_entries += priority
                self._current_cycle.batch_count += 1
                if embedding_time_ms > 0:
                    self._embedding_times.append(embedding_time_ms)

    def end_cycle(self, total_entries: int, entries_with_embeddings: int):
        """Complete the current cycle and update totals."""
        with self._lock:
            if not self._current_cycle:
                return

            # Finalize cycle metrics
            self._current_cycle.completed_at = datetime.now(timezone.utc).isoformat()
            self._current_cycle.duration_seconds = time.time() - self._cycle_start_time
            if self._embedding_times:
                self._current_cycle.avg_embedding_time_ms = sum(self._embedding_times) / len(self._embedding_times)

            # Update daemon totals
            self._metrics.total_cycles += 1
            self._metrics.total_entries_processed += self._current_cycle.entries_processed
            self._metrics.total_entries_failed += self._current_cycle.entries_failed
            self._metrics.last_cycle_at = self._current_cycle.completed_at
            self._metrics.total_entries_in_kb = total_entries
            self._metrics.entries_with_embeddings = entries_with_embeddings

            # Calculate coverage
            if total_entries > 0:
                self._metrics.current_coverage_percent = (entries_with_embeddings / total_entries) * 100

            # Calculate averages
            if self._metrics.total_cycles > 0:
                self._metrics.avg_entries_per_cycle = (
                    self._metrics.total_entries_processed / self._metrics.total_cycles
                )

            # Store recent cycles (as dicts for JSON)
            cycle_dict = asdict(self._current_cycle)
            self._metrics.recent_cycles.append(cycle_dict)
            if len(self._metrics.recent_cycles) > self.max_recent_cycles:
                self._metrics.recent_cycles = self._metrics.recent_cycles[-self.max_recent_cycles:]

            # Update average cycle duration
            durations = [c["duration_seconds"] for c in self._metrics.recent_cycles if c.get("duration_seconds")]
            if durations:
                self._metrics.avg_cycle_duration_seconds = sum(durations) / len(durations)

            self._current_cycle = None
            self._save()

    def get_metrics(self) -> DaemonMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return DaemonMetrics(**asdict(self._metrics))

    def get_current_cycle(self) -> Optional[CycleMetrics]:
        """Get current cycle metrics (if in progress)."""
        with self._lock:
            return self._current_cycle

    def format_status(self) -> str:
        """Format metrics as human-readable status string."""
        m = self.get_metrics()
        lines = [
            "=" * 50,
            "AfterImage Embedding Daemon Status",
            "=" * 50,
            f"Device: {m.device}",
            f"Model: {m.model_name}",
            f"Embedding Dim: {m.active_embedding_dim}",
            f"Started: {m.daemon_started_at[:19] if m.daemon_started_at else 'N/A'}",
            "",
            f"Total KB Entries: {m.total_entries_in_kb:,}",
            f"With Embeddings:  {m.entries_with_embeddings:,}",
            f"Coverage:         {m.current_coverage_percent:.1f}%",
            "",
            f"Total Cycles:     {m.total_cycles}",
            f"Total Processed:  {m.total_entries_processed:,}",
            f"Total Failed:     {m.total_entries_failed:,}",
            f"Avg/Cycle:        {m.avg_entries_per_cycle:.1f}",
            f"Avg Duration:     {m.avg_cycle_duration_seconds:.1f}s",
            "",
        ]

        if m.rate_limit_enabled:
            lines.append(f"Rate Limit Waits: {m.rate_limit_waits_total}")
            lines.append(f"Rate Limit Timeouts: {m.rate_limit_timeouts_total}")
            lines.append("")

        if m.retry_enabled:
            lines.append(f"Entries Retrying: {m.entries_retrying}")
            lines.append(f"Permanently Failed: {m.entries_permanently_failed}")
            lines.append("")

        if m.recent_cycles:
            lines.append("Recent Cycles:")
            for cycle in m.recent_cycles[-5:]:
                lines.append(
                    f"  #{cycle['cycle_id']}: {cycle['entries_processed']} processed, "
                    f"{cycle['duration_seconds']:.1f}s"
                )

        lines.append("=" * 50)
        return "\n".join(lines)

    def record_warmup(self, warmup_time_ms: float):
        """Record model warmup time."""
        with self._lock:
            self._metrics.warmup_time_ms = warmup_time_ms
            self._metrics.warmup_count += 1
            self._save()

    def update_retry_stats(self, retrying: int, permanently_failed: int, retry_attempts: int = 0, retry_successes: int = 0):
        """Update retry-related metrics."""
        with self._lock:
            self._metrics.entries_retrying = retrying
            self._metrics.entries_permanently_failed = permanently_failed
            self._metrics.retry_attempts_total += retry_attempts
            self._metrics.retry_successes_total += retry_successes
            self._metrics.retry_enabled = True
            self._save()

    def update_rate_limit_stats(self, waits: int = 0, timeouts: int = 0):
        """Update rate limiter metrics."""
        with self._lock:
            self._metrics.rate_limit_waits_total += waits
            self._metrics.rate_limit_timeouts_total += timeouts
            self._metrics.rate_limit_enabled = True
            self._save()

    def set_rate_limit_enabled(self, enabled: bool):
        """Set rate limiting enabled flag."""
        with self._lock:
            self._metrics.rate_limit_enabled = enabled
            self._save()

    def get_metrics_dict(self) -> dict:
        """Get metrics as a plain dictionary for health endpoint."""
        with self._lock:
            return self._metrics.to_dict()
