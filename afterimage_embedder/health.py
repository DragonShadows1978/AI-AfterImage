"""
HTTP Health Check Server for AfterImage Embedding Daemon.

Provides Prometheus-compatible metrics endpoint and health/readiness probes.
Runs in a background thread without blocking the main daemon loop.
"""

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class HealthRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints."""

    # These will be set by the server before starting
    _metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None
    _ready_check: Optional[Callable[[], bool]] = None
    _server_start_time: float = 0

    def log_message(self, format: str, *args):
        logger.debug(f"Health endpoint: {format % args}")

    def _send_response(self, status: int, content_type: str, body: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        if self.path == "/health" or self.path == "/healthz":
            self._handle_health()
        elif self.path == "/ready" or self.path == "/readyz":
            self._handle_ready()
        elif self.path == "/metrics":
            self._handle_metrics()
        elif self.path == "/":
            self._handle_root()
        else:
            self._send_response(404, "text/plain", "Not Found")

    def _handle_root(self):
        info = {
            "service": "afterimage-embedder",
            "endpoints": ["/health", "/ready", "/metrics"],
            "uptime_seconds": time.time() - HealthRequestHandler._server_start_time
        }
        self._send_response(200, "application/json", json.dumps(info, indent=2))

    def _handle_health(self):
        response = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - HealthRequestHandler._server_start_time
        }
        self._send_response(200, "application/json", json.dumps(response))

    def _handle_ready(self):
        ready_check = HealthRequestHandler._ready_check
        if ready_check and ready_check():
            response = {"status": "ready", "timestamp": time.time()}
            self._send_response(200, "application/json", json.dumps(response))
        else:
            response = {"status": "not_ready", "timestamp": time.time()}
            self._send_response(503, "application/json", json.dumps(response))

    def _handle_metrics(self):
        metrics_provider = HealthRequestHandler._metrics_provider
        if not metrics_provider:
            self._send_response(503, "text/plain", "# Metrics not available\n")
            return
        try:
            metrics = metrics_provider()
            prometheus_output = self._format_prometheus(metrics)
            self._send_response(200, "text/plain; version=0.0.4", prometheus_output)
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            self._send_response(500, "text/plain", f"# Error: {e}\n")

    def _format_prometheus(self, metrics: Dict[str, Any]) -> str:
        lines = []
        lines.append("# HELP afterimage_embedder_up Whether the embedder daemon is running")
        lines.append("# TYPE afterimage_embedder_up gauge")
        lines.append("afterimage_embedder_up 1")
        lines.append("")
        coverage = metrics.get("current_coverage_percent", 0)
        lines.append("# HELP afterimage_embedder_coverage_percent Percentage of entries with embeddings")
        lines.append("# TYPE afterimage_embedder_coverage_percent gauge")
        lines.append(f"afterimage_embedder_coverage_percent {coverage:.2f}")
        lines.append("")
        total = metrics.get("total_entries_in_kb", 0)
        lines.append("# HELP afterimage_embedder_entries_total Total entries in knowledge base")
        lines.append("# TYPE afterimage_embedder_entries_total gauge")
        lines.append(f"afterimage_embedder_entries_total {total}")
        lines.append("")
        with_embeddings = metrics.get("entries_with_embeddings", 0)
        lines.append("# HELP afterimage_embedder_entries_with_embeddings Entries that have embeddings")
        lines.append("# TYPE afterimage_embedder_entries_with_embeddings gauge")
        lines.append(f"afterimage_embedder_entries_with_embeddings {with_embeddings}")
        lines.append("")
        cycles = metrics.get("total_cycles", 0)
        lines.append("# HELP afterimage_embedder_cycles_total Total reindex cycles completed")
        lines.append("# TYPE afterimage_embedder_cycles_total counter")
        lines.append(f"afterimage_embedder_cycles_total {cycles}")
        lines.append("")
        processed = metrics.get("total_entries_processed", 0)
        lines.append("# HELP afterimage_embedder_entries_processed_total Total entries processed")
        lines.append("# TYPE afterimage_embedder_entries_processed_total counter")
        lines.append(f"afterimage_embedder_entries_processed_total {processed}")
        lines.append("")
        failed = metrics.get("total_entries_failed", 0)
        lines.append("# HELP afterimage_embedder_entries_failed_total Total entries that failed processing")
        lines.append("# TYPE afterimage_embedder_entries_failed_total counter")
        lines.append(f"afterimage_embedder_entries_failed_total {failed}")
        lines.append("")
        avg_duration = metrics.get("avg_cycle_duration_seconds", 0)
        lines.append("# HELP afterimage_embedder_cycle_duration_seconds Average cycle duration")
        lines.append("# TYPE afterimage_embedder_cycle_duration_seconds gauge")
        lines.append(f"afterimage_embedder_cycle_duration_seconds {avg_duration:.3f}")
        lines.append("")
        warmup_time = metrics.get("warmup_time_ms", 0)
        if warmup_time > 0:
            lines.append("# HELP afterimage_embedder_warmup_time_seconds Model warmup duration")
            lines.append("# TYPE afterimage_embedder_warmup_time_seconds gauge")
            lines.append(f"afterimage_embedder_warmup_time_seconds {warmup_time / 1000:.3f}")
            lines.append("")
        retrying = metrics.get("entries_retrying", 0)
        if retrying > 0 or metrics.get("retry_enabled", False):
            lines.append("# HELP afterimage_embedder_entries_retrying Entries currently in retry queue")
            lines.append("# TYPE afterimage_embedder_entries_retrying gauge")
            lines.append(f"afterimage_embedder_entries_retrying {retrying}")
            lines.append("")
            perm_failed = metrics.get("entries_permanently_failed", 0)
            lines.append("# HELP afterimage_embedder_entries_permanently_failed Entries that exhausted retries")
            lines.append("# TYPE afterimage_embedder_entries_permanently_failed gauge")
            lines.append(f"afterimage_embedder_entries_permanently_failed {perm_failed}")
            lines.append("")
        device = metrics.get("device", "unknown")
        model = metrics.get("model_name", "unknown")
        lines.append("# HELP afterimage_embedder_info Daemon information")
        lines.append("# TYPE afterimage_embedder_info gauge")
        lines.append(f'afterimage_embedder_info{{device="{device}",model="{model}"}} 1')
        lines.append("")
        return "\n".join(lines)


class HealthServer:
    """HTTP server for health checks and metrics in background thread."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        ready_check: Optional[Callable[[], bool]] = None
    ):
        self.host = host
        self.port = port
        self._metrics_provider = metrics_provider
        self._ready_check = ready_check
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._start_time = time.time()

    def start(self) -> bool:
        if self._running:
            return True
        try:
            # Store callables as class attributes (functions, not methods)
            HealthRequestHandler._metrics_provider = self._metrics_provider
            HealthRequestHandler._ready_check = self._ready_check
            HealthRequestHandler._server_start_time = self._start_time
            self._server = HTTPServer((self.host, self.port), HealthRequestHandler)
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            self._running = True
            logger.info(f"Health server started on http://{self.host}:{self.port}")
            return True
        except OSError as e:
            if e.errno == 98:
                logger.warning(f"Health server port {self.port} already in use")
            else:
                logger.error(f"Failed to start health server: {e}")
            return False

    def _serve(self):
        try:
            self._server.serve_forever()
        except Exception as e:
            if self._running:
                logger.error(f"Health server error: {e}")

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Health server stopped")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
