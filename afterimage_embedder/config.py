"""
Configuration for the AfterImage Embedding Daemon.

Handles loading config from environment and YAML files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import yaml


@dataclass
class DaemonConfig:
    """Configuration for the embedding daemon."""

    # Processing settings
    batch_size: int = 32
    max_entries_per_cycle: int = 500  # Limit per reindex cycle
    reindex_interval_seconds: int = 300  # 5 minutes between cycles

    # Priority queue settings
    priority_window_hours: int = 24  # Entries newer than this get priority
    priority_batch_first: bool = True  # Process priority entries first

    # GPU settings
    device: str = "auto"  # "auto", "cuda", or "cpu"
    cuda_memory_fraction: float = 0.5  # Max GPU memory to use

    # Backend settings
    backend: str = "auto"  # "auto", "postgresql", "sqlite"
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "afterimage"
    pg_user: str = "afterimage"
    pg_password: str = ""  # From env: AFTERIMAGE_PG_PASSWORD
    sqlite_path: Path = field(default_factory=lambda: Path.home() / ".afterimage" / "memory.db")

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # Metrics
    metrics_enabled: bool = True
    metrics_file: Path = field(default_factory=lambda: Path.home() / ".afterimage" / "embedder_metrics.json")

    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    fallback_models: List[str] = field(default_factory=lambda: [
        "all-MiniLM-L12-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-mpnet-base-v2"
    ])
    model_retry_on_failure: bool = True

    # Health server settings
    health_server_enabled: bool = True
    health_server_host: str = "127.0.0.1"
    health_server_port: int = 9090

    # Retry settings
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_jitter: bool = True
    retry_state_file: Path = field(default_factory=lambda: Path.home() / ".afterimage" / "retry_state.json")

    # Warmup settings
    warmup_enabled: bool = True
    warmup_batch_size: int = 10

    # Rate limiting settings (Cycle 3)
    rate_limit_enabled: bool = True
    rate_limit_gpu_capacity: int = 50  # Max GPU batches per burst
    rate_limit_gpu_refill: float = 10.0  # GPU batches per second
    rate_limit_db_capacity: int = 100  # Max DB writes per burst
    rate_limit_db_refill: float = 50.0  # DB writes per second

    # Web dashboard settings (Cycle 3)
    web_dashboard_enabled: bool = False
    web_dashboard_host: str = "127.0.0.1"
    web_dashboard_port: int = 8080

    # Graceful shutdown settings (Cycle 3)
    shutdown_timeout_seconds: int = 60

    # Notification settings (Cycle 3) - webhook URLs from env only
    notifications_enabled: bool = False
    notify_on_startup: bool = True
    notify_on_shutdown: bool = True
    notify_on_milestones: bool = True
    notify_on_failures: bool = True
    notify_min_interval: int = 60

    @classmethod
    def from_env(cls) -> "DaemonConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Processing
        if val := os.environ.get("EMBEDDER_BATCH_SIZE"):
            config.batch_size = int(val)
        if val := os.environ.get("EMBEDDER_MAX_PER_CYCLE"):
            config.max_entries_per_cycle = int(val)
        if val := os.environ.get("EMBEDDER_INTERVAL_SECONDS"):
            config.reindex_interval_seconds = int(val)

        # Priority queue
        if val := os.environ.get("EMBEDDER_PRIORITY_HOURS"):
            config.priority_window_hours = int(val)

        # GPU
        if val := os.environ.get("EMBEDDER_DEVICE"):
            config.device = val
        if val := os.environ.get("EMBEDDER_CUDA_MEMORY_FRACTION"):
            config.cuda_memory_fraction = float(val)

        # Backend
        if val := os.environ.get("AFTERIMAGE_BACKEND"):
            config.backend = val
        if val := os.environ.get("AFTERIMAGE_PG_HOST"):
            config.pg_host = val
        if val := os.environ.get("AFTERIMAGE_PG_PORT"):
            config.pg_port = int(val)
        if val := os.environ.get("AFTERIMAGE_PG_DATABASE"):
            config.pg_database = val
        if val := os.environ.get("AFTERIMAGE_PG_USER"):
            config.pg_user = val
        if val := os.environ.get("AFTERIMAGE_PG_PASSWORD"):
            config.pg_password = val
        if val := os.environ.get("AFTERIMAGE_SQLITE_PATH"):
            config.sqlite_path = Path(val)

        # Logging
        if val := os.environ.get("EMBEDDER_LOG_LEVEL"):
            config.log_level = val
        if val := os.environ.get("EMBEDDER_LOG_FILE"):
            config.log_file = Path(val)

        # Metrics
        if val := os.environ.get("EMBEDDER_METRICS_ENABLED"):
            config.metrics_enabled = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_METRICS_FILE"):
            config.metrics_file = Path(val)

        # Model
        if val := os.environ.get("EMBEDDER_MODEL_NAME"):
            config.model_name = val

        # Health server
        if val := os.environ.get("EMBEDDER_HEALTH_ENABLED"):
            config.health_server_enabled = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_HEALTH_HOST"):
            config.health_server_host = val
        if val := os.environ.get("EMBEDDER_HEALTH_PORT"):
            config.health_server_port = int(val)

        # Retry
        if val := os.environ.get("EMBEDDER_RETRY_ENABLED"):
            config.retry_enabled = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_RETRY_MAX_ATTEMPTS"):
            config.retry_max_attempts = int(val)
        if val := os.environ.get("EMBEDDER_RETRY_BASE_DELAY"):
            config.retry_base_delay = float(val)
        if val := os.environ.get("EMBEDDER_RETRY_MAX_DELAY"):
            config.retry_max_delay = float(val)

        # Warmup
        if val := os.environ.get("EMBEDDER_WARMUP_ENABLED"):
            config.warmup_enabled = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_WARMUP_BATCH_SIZE"):
            config.warmup_batch_size = int(val)

        # Rate limiting (Cycle 3)
        if val := os.environ.get("EMBEDDER_RATE_LIMIT_ENABLED"):
            config.rate_limit_enabled = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_RATE_LIMIT_GPU_CAPACITY"):
            config.rate_limit_gpu_capacity = int(val)
        if val := os.environ.get("EMBEDDER_RATE_LIMIT_GPU_REFILL"):
            config.rate_limit_gpu_refill = float(val)
        if val := os.environ.get("EMBEDDER_RATE_LIMIT_DB_CAPACITY"):
            config.rate_limit_db_capacity = int(val)
        if val := os.environ.get("EMBEDDER_RATE_LIMIT_DB_REFILL"):
            config.rate_limit_db_refill = float(val)

        # Web dashboard (Cycle 3)
        if val := os.environ.get("EMBEDDER_WEB_DASHBOARD_ENABLED"):
            config.web_dashboard_enabled = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_WEB_DASHBOARD_HOST"):
            config.web_dashboard_host = val
        if val := os.environ.get("EMBEDDER_WEB_DASHBOARD_PORT"):
            config.web_dashboard_port = int(val)

        # Graceful shutdown (Cycle 3)
        if val := os.environ.get("EMBEDDER_SHUTDOWN_TIMEOUT"):
            config.shutdown_timeout_seconds = int(val)

        # Notifications (Cycle 3)
        if os.environ.get("EMBEDDER_DISCORD_WEBHOOK") or os.environ.get("EMBEDDER_SLACK_WEBHOOK"):
            config.notifications_enabled = True
        if val := os.environ.get("EMBEDDER_NOTIFY_STARTUP"):
            config.notify_on_startup = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_NOTIFY_SHUTDOWN"):
            config.notify_on_shutdown = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_NOTIFY_MILESTONES"):
            config.notify_on_milestones = val.lower() in ("true", "1", "yes")
        if val := os.environ.get("EMBEDDER_NOTIFY_FAILURES"):
            config.notify_on_failures = val.lower() in ("true", "1", "yes")

        return config

    @classmethod
    def from_yaml(cls, path: Path) -> "DaemonConfig":
        """Load configuration from YAML file."""
        config = cls()

        if not path.exists():
            return config

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Processing
        if proc := data.get("processing"):
            config.batch_size = proc.get("batch_size", config.batch_size)
            config.max_entries_per_cycle = proc.get("max_entries_per_cycle", config.max_entries_per_cycle)
            config.reindex_interval_seconds = proc.get("reindex_interval_seconds", config.reindex_interval_seconds)

        # Priority
        if prio := data.get("priority"):
            config.priority_window_hours = prio.get("window_hours", config.priority_window_hours)
            config.priority_batch_first = prio.get("batch_first", config.priority_batch_first)

        # GPU
        if gpu := data.get("gpu"):
            config.device = gpu.get("device", config.device)
            config.cuda_memory_fraction = gpu.get("cuda_memory_fraction", config.cuda_memory_fraction)

        # Backend
        if backend := data.get("backend"):
            config.backend = backend.get("type", config.backend)
            if pg := backend.get("postgresql"):
                config.pg_host = pg.get("host", config.pg_host)
                config.pg_port = pg.get("port", config.pg_port)
                config.pg_database = pg.get("database", config.pg_database)
                config.pg_user = pg.get("user", config.pg_user)
                config.pg_password = pg.get("password", config.pg_password)
            if sqlite := backend.get("sqlite"):
                config.sqlite_path = Path(sqlite.get("path", config.sqlite_path))

        # Logging
        if log := data.get("logging"):
            config.log_level = log.get("level", config.log_level)
            if file_path := log.get("file"):
                config.log_file = Path(file_path)

        # Metrics
        if metrics := data.get("metrics"):
            config.metrics_enabled = metrics.get("enabled", config.metrics_enabled)
            if file_path := metrics.get("file"):
                config.metrics_file = Path(file_path)

        # Model
        if model := data.get("model"):
            config.model_name = model.get("name", config.model_name)
            config.embedding_dim = model.get("embedding_dim", config.embedding_dim)
            if fallbacks := model.get("fallback_models"):
                config.fallback_models = fallbacks

        # Health server
        if health := data.get("health"):
            config.health_server_enabled = health.get("enabled", config.health_server_enabled)
            config.health_server_host = health.get("host", config.health_server_host)
            config.health_server_port = health.get("port", config.health_server_port)

        # Retry
        if retry := data.get("retry"):
            config.retry_enabled = retry.get("enabled", config.retry_enabled)
            config.retry_max_attempts = retry.get("max_attempts", config.retry_max_attempts)
            config.retry_base_delay = retry.get("base_delay", config.retry_base_delay)
            config.retry_max_delay = retry.get("max_delay", config.retry_max_delay)
            config.retry_jitter = retry.get("jitter", config.retry_jitter)

        # Warmup
        if warmup := data.get("warmup"):
            config.warmup_enabled = warmup.get("enabled", config.warmup_enabled)
            config.warmup_batch_size = warmup.get("batch_size", config.warmup_batch_size)

        # Rate limiting (Cycle 3)
        if rate_limit := data.get("rate_limit"):
            config.rate_limit_enabled = rate_limit.get("enabled", config.rate_limit_enabled)
            config.rate_limit_gpu_capacity = rate_limit.get("gpu_capacity", config.rate_limit_gpu_capacity)
            config.rate_limit_gpu_refill = rate_limit.get("gpu_refill", config.rate_limit_gpu_refill)
            config.rate_limit_db_capacity = rate_limit.get("db_capacity", config.rate_limit_db_capacity)
            config.rate_limit_db_refill = rate_limit.get("db_refill", config.rate_limit_db_refill)

        # Web dashboard (Cycle 3)
        if web := data.get("web_dashboard"):
            config.web_dashboard_enabled = web.get("enabled", config.web_dashboard_enabled)
            config.web_dashboard_host = web.get("host", config.web_dashboard_host)
            config.web_dashboard_port = web.get("port", config.web_dashboard_port)

        # Graceful shutdown (Cycle 3)
        if shutdown := data.get("shutdown"):
            config.shutdown_timeout_seconds = shutdown.get("timeout_seconds", config.shutdown_timeout_seconds)

        # Notifications (Cycle 3)
        if notif := data.get("notifications"):
            config.notifications_enabled = notif.get("enabled", config.notifications_enabled)
            config.notify_on_startup = notif.get("on_startup", config.notify_on_startup)
            config.notify_on_shutdown = notif.get("on_shutdown", config.notify_on_shutdown)
            config.notify_on_milestones = notif.get("on_milestones", config.notify_on_milestones)
            config.notify_on_failures = notif.get("on_failures", config.notify_on_failures)
            config.notify_min_interval = notif.get("min_interval", config.notify_min_interval)

        return config

    @classmethod
    def load(cls) -> "DaemonConfig":
        """Load configuration from default locations, with env overrides."""
        # Start with YAML config if exists
        yaml_paths = [
            Path.home() / ".afterimage" / "embedder.yaml",
            Path.home() / ".afterimage" / "embedder.yml",
            Path("/etc/afterimage/embedder.yaml"),
        ]

        config = cls()
        for path in yaml_paths:
            if path.exists():
                config = cls.from_yaml(path)
                break

        # Apply environment overrides
        env_config = cls.from_env()

        # Merge - env takes precedence
        for field_name in [
            "batch_size", "max_entries_per_cycle", "reindex_interval_seconds",
            "priority_window_hours", "device", "cuda_memory_fraction",
            "backend", "pg_host", "pg_port", "pg_database", "pg_user",
            "log_level", "metrics_enabled", "model_name",
            "rate_limit_enabled", "rate_limit_gpu_capacity", "rate_limit_gpu_refill",
            "rate_limit_db_capacity", "rate_limit_db_refill",
            "web_dashboard_enabled", "web_dashboard_host", "web_dashboard_port",
            "shutdown_timeout_seconds", "notifications_enabled"
        ]:
            env_val = getattr(env_config, field_name)
            default_val = getattr(cls(), field_name)
            if env_val != default_val:
                setattr(config, field_name, env_val)

        # Password from env always overrides
        if env_config.pg_password:
            config.pg_password = env_config.pg_password

        return config

    def detect_device(self) -> str:
        """Auto-detect the best available device."""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"
