"""
Webhook Notifications for AfterImage Embedding Daemon.

Sends notifications to Discord and Slack webhooks for critical events.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
from threading import Lock

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not installed. Notifications unavailable.")


class EventType(Enum):
    """Types of notification events."""
    DAEMON_START = "daemon_start"
    DAEMON_SHUTDOWN = "daemon_shutdown"
    COVERAGE_MILESTONE = "coverage_milestone"
    PERMANENT_FAILURE = "permanent_failure"
    CRITICAL_ERROR = "critical_error"
    BATCH_COMPLETE = "batch_complete"


@dataclass
class NotificationConfig:
    """Configuration for webhook notifications."""
    enabled: bool = False
    discord_webhook_url: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    notify_on_startup: bool = True
    notify_on_shutdown: bool = True
    notify_on_milestones: bool = True
    notify_on_failures: bool = True
    min_interval_seconds: int = 60
    milestone_thresholds: List[int] = field(default_factory=lambda: [50, 75, 90, 100])

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Load notification config from environment variables."""
        config = cls()
        config.discord_webhook_url = os.environ.get("EMBEDDER_DISCORD_WEBHOOK")
        config.slack_webhook_url = os.environ.get("EMBEDDER_SLACK_WEBHOOK")
        config.enabled = bool(config.discord_webhook_url or config.slack_webhook_url)

        if val := os.environ.get("EMBEDDER_NOTIFY_STARTUP"):
            config.notify_on_startup = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_NOTIFY_SHUTDOWN"):
            config.notify_on_shutdown = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_NOTIFY_MILESTONES"):
            config.notify_on_milestones = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_NOTIFY_FAILURES"):
            config.notify_on_failures = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_NOTIFY_INTERVAL"):
            config.min_interval_seconds = int(val)

        return config


class WebhookNotifier:
    """Sends webhook notifications to Discord and Slack."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self._last_notification: Dict[str, float] = {}
        self._milestones_reached: set = set()
        self._lock = Lock()
        self._stats = {"notifications_sent": 0, "notifications_failed": 0, "rate_limited": 0}

        if config.enabled:
            logger.info(f"Webhook notifier enabled: Discord={'yes' if config.discord_webhook_url else 'no'}, Slack={'yes' if config.slack_webhook_url else 'no'}")

    def _should_notify(self, event_type: EventType) -> bool:
        if not self.config.enabled or not REQUESTS_AVAILABLE:
            return False

        if event_type == EventType.DAEMON_START and not self.config.notify_on_startup:
            return False
        if event_type == EventType.DAEMON_SHUTDOWN and not self.config.notify_on_shutdown:
            return False
        if event_type == EventType.COVERAGE_MILESTONE and not self.config.notify_on_milestones:
            return False
        if event_type in (EventType.PERMANENT_FAILURE, EventType.CRITICAL_ERROR) and not self.config.notify_on_failures:
            return False

        with self._lock:
            last_time = self._last_notification.get(event_type.value, 0)
            now = time.time()
            if now - last_time < self.config.min_interval_seconds:
                self._stats["rate_limited"] += 1
                return False
        return True

    def _record_notification(self, event_type: EventType, success: bool) -> None:
        with self._lock:
            if success:
                self._last_notification[event_type.value] = time.time()
                self._stats["notifications_sent"] += 1
            else:
                self._stats["notifications_failed"] += 1

    def _send_discord(self, message: str, details: Optional[Dict[str, Any]] = None, color: int = 0x00FF00) -> bool:
        if not self.config.discord_webhook_url:
            return False
        try:
            embed = {"title": "AfterImage Embedder", "description": message, "color": color, "timestamp": datetime.now(timezone.utc).isoformat()}
            if details:
                embed["fields"] = [{"name": k, "value": str(v), "inline": True} for k, v in details.items()]
            response = requests.post(self.config.discord_webhook_url, json={"embeds": [embed]}, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Discord notification failed: {e}")
            return False

    def _send_slack(self, message: str, details: Optional[Dict[str, Any]] = None, color: str = "good") -> bool:
        if not self.config.slack_webhook_url:
            return False
        try:
            attachment = {"color": color, "title": "AfterImage Embedder", "text": message, "ts": int(time.time())}
            if details:
                attachment["fields"] = [{"title": k, "value": str(v), "short": True} for k, v in details.items()]
            response = requests.post(self.config.slack_webhook_url, json={"attachments": [attachment]}, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")
            return False

    def notify(self, event_type: EventType, message: str, details: Optional[Dict[str, Any]] = None,
               color: int = 0x00FF00, slack_color: str = "good") -> bool:
        if not self._should_notify(event_type):
            return False
        success = False
        if self.config.discord_webhook_url:
            if self._send_discord(message, details, color):
                success = True
        if self.config.slack_webhook_url:
            if self._send_slack(message, details, slack_color):
                success = True
        self._record_notification(event_type, success)
        return success

    def notify_startup(self, device: str, model: str, coverage: float) -> bool:
        return self.notify(EventType.DAEMON_START, "Embedding daemon started",
                          {"Device": device, "Model": model, "Coverage": f"{coverage:.1f}%"},
                          color=0x28A745, slack_color="good")

    def notify_shutdown(self, reason: str, cycles_completed: int = 0, entries_processed: int = 0) -> bool:
        return self.notify(EventType.DAEMON_SHUTDOWN, f"Embedding daemon shutting down: {reason}",
                          {"Reason": reason, "Cycles Completed": cycles_completed, "Entries Processed": entries_processed},
                          color=0xFFC107, slack_color="warning")

    def notify_coverage_milestone(self, coverage: float, total_entries: int, entries_with_embeddings: int) -> bool:
        threshold = None
        for t in sorted(self.config.milestone_thresholds):
            if coverage >= t and t not in self._milestones_reached:
                threshold = t
                break
        if threshold is None:
            return False
        with self._lock:
            self._milestones_reached.add(threshold)
        color = 0x28A745 if threshold >= 100 else 0x17A2B8
        slack_color = "good" if threshold >= 100 else "#17a2b8"
        return self.notify(EventType.COVERAGE_MILESTONE, f"Coverage milestone reached: {threshold}%",
                          {"Coverage": f"{coverage:.1f}%", "Entries": f"{entries_with_embeddings:,} / {total_entries:,}"},
                          color=color, slack_color=slack_color)

    def notify_permanent_failure(self, entry_id: str, file_path: str, error: str, attempts: int) -> bool:
        return self.notify(EventType.PERMANENT_FAILURE, "Entry permanently failed after max retries",
                          {"Entry ID": entry_id[:16] + "..." if len(entry_id) > 16 else entry_id,
                           "File": file_path.split("/")[-1] if file_path else "unknown",
                           "Attempts": attempts, "Error": error[:100] if error else "unknown"},
                          color=0xDC3545, slack_color="danger")

    def notify_error(self, error: str, context: Optional[str] = None) -> bool:
        details = {"Error": error}
        if context:
            details["Context"] = context
        return self.notify(EventType.CRITICAL_ERROR, f"Critical error: {error}", details, color=0xDC3545, slack_color="danger")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {**self._stats, "milestones_reached": list(self._milestones_reached), "last_notifications": dict(self._last_notification)}

    def reset_milestones(self) -> None:
        with self._lock:
            self._milestones_reached.clear()
