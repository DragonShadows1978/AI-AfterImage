"""
Web Dashboard for AfterImage Embedding Daemon.

Provides a browser-based dashboard with auto-refresh for monitoring metrics.
Uses FastAPI with inline HTML/CSS (no external template files).
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

# Check for FastAPI availability
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI/uvicorn not installed. Web dashboard unavailable.")


def check_fastapi_available() -> bool:
    """Check if FastAPI dependencies are installed."""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
        return False
    return True


# Dashboard HTML template (inline, no external files)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>AfterImage Embedder Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            color: #00d4ff;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .status-running { background: #28a745; }
        .status-stopped { background: #dc3545; }

        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            color: #00d4ff;
            font-size: 1.1em;
            margin-bottom: 15px;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
        }

        .progress-container {
            background: #0f3460;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #28a745);
            transition: width 0.5s ease;
            border-radius: 10px;
        }
        .progress-text {
            position: absolute;
            width: 100%;
            text-align: center;
            line-height: 30px;
            font-weight: bold;
            color: #fff;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00d4ff;
        }
        .stat-label {
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #0f3460;
        }
        th { color: #00d4ff; font-weight: 600; }
        tr:hover { background: #0f3460; }

        .links {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        .links a {
            color: #00d4ff;
            text-decoration: none;
            padding: 8px 16px;
            background: #0f3460;
            border-radius: 6px;
            transition: background 0.3s;
        }
        .links a:hover { background: #1a4080; }

        .rate-limiter-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .rate-box {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
        }
        .rate-box h3 {
            color: #00d4ff;
            font-size: 1em;
            margin-bottom: 10px;
        }
        .rate-stat { margin: 5px 0; }
        .rate-stat span { color: #888; }

        .footer {
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }
        .failed { color: #dc3545; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            AfterImage Embedder
            <span class="status-badge status-running" id="status">RUNNING</span>
        </h1>

        <!-- Header Info -->
        <div class="card">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="device">-</div>
                    <div class="stat-label">Device</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="model">-</div>
                    <div class="stat-label">Model</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="uptime">-</div>
                    <div class="stat-label">Uptime</div>
                </div>
            </div>
        </div>

        <!-- Coverage Progress -->
        <div class="card">
            <h2>Coverage Progress</h2>
            <div class="progress-container">
                <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
                <div class="progress-text" id="progress-text">0%</div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #888;">
                <span id="entries-count">0 / 0 entries</span>
                <span id="remaining">0 remaining</span>
            </div>
        </div>

        <!-- Stats Grid -->
        <div class="card">
            <h2>Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value success" id="total-processed">0</div>
                    <div class="stat-label">Total Processed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value failed" id="total-failed">0</div>
                    <div class="stat-label">Total Failed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="total-cycles">0</div>
                    <div class="stat-label">Total Cycles</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="throughput">0/s</div>
                    <div class="stat-label">Throughput</div>
                </div>
            </div>
        </div>

        <!-- Rate Limiter -->
        <div class="card" id="rate-limiter-card">
            <h2>Rate Limiter</h2>
            <div class="rate-limiter-grid">
                <div class="rate-box">
                    <h3>GPU Bucket</h3>
                    <div class="rate-stat"><span>Tokens:</span> <strong id="gpu-tokens">-</strong></div>
                    <div class="rate-stat"><span>Waits:</span> <strong id="gpu-waits">0</strong></div>
                    <div class="rate-stat"><span>Capacity:</span> <strong id="gpu-capacity">-</strong></div>
                </div>
                <div class="rate-box">
                    <h3>DB Bucket</h3>
                    <div class="rate-stat"><span>Tokens:</span> <strong id="db-tokens">-</strong></div>
                    <div class="rate-stat"><span>Waits:</span> <strong id="db-waits">0</strong></div>
                    <div class="rate-stat"><span>Capacity:</span> <strong id="db-capacity">-</strong></div>
                </div>
            </div>
        </div>

        <!-- Retry Queue -->
        <div class="card" id="retry-card">
            <h2>Retry Queue</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value warning" id="retrying">0</div>
                    <div class="stat-label">Retrying</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value failed" id="perm-failed">0</div>
                    <div class="stat-label">Permanently Failed</div>
                </div>
            </div>
        </div>

        <!-- Recent Cycles -->
        <div class="card">
            <h2>Recent Cycles</h2>
            <table>
                <thead>
                    <tr>
                        <th>Cycle</th>
                        <th>Processed</th>
                        <th>Failed</th>
                        <th>Priority</th>
                        <th>Duration</th>
                        <th>Rate</th>
                    </tr>
                </thead>
                <tbody id="cycles-table">
                    <tr><td colspan="6" style="text-align: center; color: #888;">No cycles yet</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Links -->
        <div class="links">
            <a href="/health">Health</a>
            <a href="/ready">Ready</a>
            <a href="/metrics">Prometheus Metrics</a>
            <a href="/api/metrics">JSON API</a>
        </div>

        <div class="footer">
            Auto-refreshes every 5 seconds | AfterImage Embedding Daemon v3.0
        </div>
    </div>

    <script>
        function formatDuration(seconds) {
            if (seconds < 60) return seconds.toFixed(1) + 's';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins + 'm ' + secs + 's';
        }

        function formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toString();
        }

        async function updateDashboard() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();

                // Header info
                document.getElementById('device').textContent = data.device || '-';
                document.getElementById('model').textContent = (data.model_name || '-').replace('all-', '');

                // Uptime
                if (data.daemon_started_at) {
                    const started = new Date(data.daemon_started_at);
                    const uptime = (Date.now() - started.getTime()) / 1000;
                    document.getElementById('uptime').textContent = formatDuration(uptime);
                }

                // Coverage
                const total = data.total_entries_in_kb || 0;
                const withEmbed = data.entries_with_embeddings || 0;
                const coverage = data.current_coverage_percent || 0;
                document.getElementById('progress-bar').style.width = coverage + '%';
                document.getElementById('progress-text').textContent = coverage.toFixed(1) + '%';
                document.getElementById('entries-count').textContent =
                    formatNumber(withEmbed) + ' / ' + formatNumber(total) + ' entries';
                document.getElementById('remaining').textContent =
                    formatNumber(total - withEmbed) + ' remaining';

                // Stats
                document.getElementById('total-processed').textContent = formatNumber(data.total_entries_processed || 0);
                document.getElementById('total-failed').textContent = formatNumber(data.total_entries_failed || 0);
                document.getElementById('total-cycles').textContent = data.total_cycles || 0;

                const avgDuration = data.avg_cycle_duration_seconds || 1;
                const avgPerCycle = data.avg_entries_per_cycle || 0;
                const throughput = avgPerCycle / avgDuration;
                document.getElementById('throughput').textContent = throughput.toFixed(1) + '/s';

                // Rate limiter
                if (data.rate_limiter) {
                    const rl = data.rate_limiter;
                    document.getElementById('gpu-tokens').textContent = rl.gpu?.current_tokens || '-';
                    document.getElementById('gpu-waits').textContent = rl.gpu?.waits_total || 0;
                    document.getElementById('gpu-capacity').textContent = rl.gpu?.capacity || '-';
                    document.getElementById('db-tokens').textContent = rl.db?.current_tokens || '-';
                    document.getElementById('db-waits').textContent = rl.db?.waits_total || 0;
                    document.getElementById('db-capacity').textContent = rl.db?.capacity || '-';
                    document.getElementById('rate-limiter-card').style.display = 'block';
                } else {
                    document.getElementById('rate-limiter-card').style.display = 'none';
                }

                // Retry queue
                document.getElementById('retrying').textContent = data.entries_retrying || 0;
                document.getElementById('perm-failed').textContent = data.entries_permanently_failed || 0;

                // Recent cycles
                const cycles = (data.recent_cycles || []).slice(-10).reverse();
                const tbody = document.getElementById('cycles-table');
                if (cycles.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No cycles yet</td></tr>';
                } else {
                    tbody.innerHTML = cycles.map(c => {
                        const rate = c.duration_seconds > 0 ? (c.entries_processed / c.duration_seconds).toFixed(1) : '0';
                        const failedClass = c.entries_failed > 0 ? 'failed' : '';
                        return '<tr>' +
                            '<td>#' + c.cycle_id + '</td>' +
                            '<td>' + c.entries_processed + '</td>' +
                            '<td class="' + failedClass + '">' + c.entries_failed + '</td>' +
                            '<td>' + (c.priority_entries || 0) + '</td>' +
                            '<td>' + (c.duration_seconds ? c.duration_seconds.toFixed(1) : '0') + 's</td>' +
                            '<td>' + rate + '/s</td>' +
                        '</tr>';
                    }).join('');
                }

            } catch (error) {
                console.error('Failed to update dashboard:', error);
            }
        }

        // Initial load
        updateDashboard();
    </script>
</body>
</html>
"""


class WebDashboard:
    """
    FastAPI-based web dashboard for monitoring the embedding daemon.

    Runs in a background thread, non-blocking to the main daemon.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        rate_limiter_stats_provider: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        """
        Initialize web dashboard.

        Args:
            host: Host to bind to
            port: Port to bind to
            metrics_provider: Function that returns current metrics dict
            rate_limiter_stats_provider: Function that returns rate limiter stats
        """
        self.host = host
        self.port = port
        self._metrics_provider = metrics_provider
        self._rate_limiter_stats_provider = rate_limiter_stats_provider
        self._app: Optional[FastAPI] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._start_time = time.time()

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with routes."""
        app = FastAPI(title="AfterImage Embedder Dashboard")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Main dashboard page."""
            return HTMLResponse(content=DASHBOARD_HTML)

        @app.get("/api/metrics")
        async def api_metrics():
            """JSON API endpoint for metrics."""
            metrics = {}
            if self._metrics_provider:
                metrics = self._metrics_provider()

            # Add rate limiter stats if available
            if self._rate_limiter_stats_provider:
                metrics["rate_limiter"] = self._rate_limiter_stats_provider()

            return JSONResponse(content=metrics)

        @app.get("/api/cycles")
        async def api_cycles():
            """JSON API for recent cycles."""
            metrics = {}
            if self._metrics_provider:
                metrics = self._metrics_provider()
            return JSONResponse(content={"cycles": metrics.get("recent_cycles", [])})

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return JSONResponse(content={
                "status": "healthy",
                "uptime_seconds": time.time() - self._start_time
            })

        @app.get("/ready")
        async def ready():
            """Readiness check endpoint."""
            return JSONResponse(content={"status": "ready"})

        @app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus-compatible metrics endpoint."""
            metrics = {}
            if self._metrics_provider:
                metrics = self._metrics_provider()

            lines = []
            lines.append("# HELP afterimage_embedder_up Whether the embedder is running")
            lines.append("# TYPE afterimage_embedder_up gauge")
            lines.append("afterimage_embedder_up 1")
            lines.append("")

            coverage = metrics.get("current_coverage_percent", 0)
            lines.append("# HELP afterimage_embedder_coverage_percent Coverage percentage")
            lines.append("# TYPE afterimage_embedder_coverage_percent gauge")
            lines.append(f"afterimage_embedder_coverage_percent {coverage:.2f}")
            lines.append("")

            total = metrics.get("total_entries_in_kb", 0)
            lines.append("# HELP afterimage_embedder_entries_total Total entries")
            lines.append("# TYPE afterimage_embedder_entries_total gauge")
            lines.append(f"afterimage_embedder_entries_total {total}")
            lines.append("")

            processed = metrics.get("total_entries_processed", 0)
            lines.append("# HELP afterimage_embedder_processed_total Total processed")
            lines.append("# TYPE afterimage_embedder_processed_total counter")
            lines.append(f"afterimage_embedder_processed_total {processed}")
            lines.append("")

            failed = metrics.get("total_entries_failed", 0)
            lines.append("# HELP afterimage_embedder_failed_total Total failed")
            lines.append("# TYPE afterimage_embedder_failed_total counter")
            lines.append(f"afterimage_embedder_failed_total {failed}")

            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content="\n".join(lines), media_type="text/plain")

        return app

    def start(self) -> bool:
        """Start the web dashboard in a background thread."""
        if not check_fastapi_available():
            return False

        if self._running:
            return True

        try:
            self._app = self._create_app()
            self._start_time = time.time()

            # Configure uvicorn
            config = uvicorn.Config(
                self._app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False
            )
            server = uvicorn.Server(config)

            # Run in background thread
            self._server_thread = threading.Thread(
                target=server.run,
                daemon=True,
                name="web-dashboard"
            )
            self._server_thread.start()
            self._running = True

            # Give server time to start
            time.sleep(0.5)

            logger.info(f"Web dashboard started at http://{self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start web dashboard: {e}")
            return False

    def stop(self) -> None:
        """Stop the web dashboard."""
        self._running = False
        logger.info("Web dashboard stopped")

    @property
    def url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.host}:{self.port}"
