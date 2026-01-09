"""
AI-AfterImage Background Embedding Service

A background daemon that generates embeddings for code entries stored in
AfterImage's knowledge base, enabling full hybrid search (FTS + semantic)
without impacting real-time hook performance.

Components:
- EmbeddingDaemon: Main daemon process with signal handling
- EmbeddingProcessor: Core processing logic with GPU support
- HealthServer: HTTP health check endpoint (Prometheus compatible)
- RetryManager: Exponential backoff for failed embeddings
- Dashboard: Rich CLI monitoring dashboard
- WebDashboard: Browser-based monitoring dashboard
- RateLimiter: Token bucket rate limiting for GPU/DB
- WebhookNotifier: Discord/Slack notifications
- MetricsCollector: Track progress and coverage
"""

__version__ = "3.0.0"

from .daemon import EmbeddingDaemon
from .processor import EmbeddingProcessor
from .metrics import MetricsCollector, DaemonMetrics
from .config import DaemonConfig
from .health import HealthServer
from .retry import RetryManager
from .dashboard import Dashboard
from .rate_limiter import RateLimiter, TokenBucket
from .notifications import WebhookNotifier, NotificationConfig, EventType

# Web dashboard is optional (requires fastapi/uvicorn)
try:
    from .web_dashboard import WebDashboard
except ImportError:
    WebDashboard = None

__all__ = [
    "EmbeddingDaemon",
    "EmbeddingProcessor",
    "MetricsCollector",
    "DaemonMetrics",
    "DaemonConfig",
    "HealthServer",
    "RetryManager",
    "Dashboard",
    "WebDashboard",
    "RateLimiter",
    "TokenBucket",
    "WebhookNotifier",
    "NotificationConfig",
    "EventType",
]
