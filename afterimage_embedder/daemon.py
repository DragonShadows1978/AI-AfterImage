"""
AfterImage Embedding Daemon

Background service that periodically generates embeddings for code entries.
Supports systemd/launchd integration, graceful shutdown, web dashboard, and notifications.
"""

import signal
import time
import logging
import sys
import os
from pathlib import Path
from threading import Event
from typing import Optional

from .config import DaemonConfig
from .processor import EmbeddingProcessor
from .metrics import MetricsCollector
from .health import HealthServer
from .notifications import WebhookNotifier, NotificationConfig


logger = logging.getLogger(__name__)


class EmbeddingDaemon:
    """
    Background daemon for generating embeddings.

    Features:
    - Periodic reindex cycles
    - Graceful shutdown on SIGTERM/SIGINT (completes current batch)
    - Systemd notify support
    - HTTP health check endpoint
    - Web dashboard
    - Retry handling for failed embeddings
    - Webhook notifications for critical events
    """

    def __init__(self, config: Optional[DaemonConfig] = None):
        self.config = config or DaemonConfig.load()
        self._setup_logging()

        self.metrics = MetricsCollector(self.config.metrics_file)
        self.processor = EmbeddingProcessor(self.config, self.metrics)

        self._shutdown_event = Event()
        self._running = False
        self._shutdown_reason = None

        self._health_file = Path.home() / ".afterimage" / "embedder.health"
        self._health_server = None
        self._web_dashboard = None
        self._notifier = None

    def _setup_logging(self):
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        handlers = [logging.StreamHandler(sys.stdout)]
        if self.config.log_file:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.config.log_file))
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=handlers
        )

    def _signal_handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_reason = sig_name
        self._shutdown_event.set()

    def _setup_signals(self):
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self._handle_reload)

    def _handle_reload(self, signum, frame):
        logger.info("Received SIGHUP, reloading configuration...")
        try:
            self.config = DaemonConfig.load()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")

    def _update_health(self, status: str = "healthy"):
        try:
            self._health_file.parent.mkdir(parents=True, exist_ok=True)
            self._health_file.write_text(f"{status}\n{time.time()}\n")
        except Exception as e:
            logger.warning(f"Could not update health file: {e}")

    def _notify_systemd(self, message: str):
        notify_socket = os.environ.get("NOTIFY_SOCKET")
        if not notify_socket:
            return
        try:
            import socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            if notify_socket.startswith("@"):
                notify_socket = "\0" + notify_socket[1:]
            sock.connect(notify_socket)
            sock.sendall(message.encode())
            sock.close()
        except Exception as e:
            logger.debug(f"systemd notify failed: {e}")

    def _setup_notifier(self):
        """Initialize webhook notifier if configured."""
        notif_config = NotificationConfig.from_env()
        if notif_config.enabled:
            self._notifier = WebhookNotifier(notif_config)
            logger.info("Webhook notifications enabled")

    def _setup_web_dashboard(self):
        """Initialize web dashboard if enabled."""
        if not self.config.web_dashboard_enabled:
            return

        try:
            from .web_dashboard import WebDashboard
            
            # Provide metrics and rate limiter stats
            def get_metrics_with_rate_limiter():
                metrics = self.metrics.get_metrics_dict()
                if self.processor.rate_limiter:
                    metrics["rate_limiter"] = self.processor.rate_limiter.get_stats()
                return metrics

            self._web_dashboard = WebDashboard(
                host=self.config.web_dashboard_host,
                port=self.config.web_dashboard_port,
                metrics_provider=get_metrics_with_rate_limiter,
                rate_limiter_stats_provider=lambda: self.processor.rate_limiter.get_stats() if self.processor.rate_limiter else None
            )
            
            if self._web_dashboard.start():
                logger.info(f"Web dashboard: {self._web_dashboard.url}")
            else:
                logger.warning("Web dashboard failed to start")
                self._web_dashboard = None
        except ImportError as e:
            logger.warning(f"Web dashboard unavailable: {e}")

    def run(self):
        self._setup_signals()
        self._setup_notifier()

        logger.info("AfterImage Embedding Daemon starting...")
        logger.info(f"Reindex interval: {self.config.reindex_interval_seconds}s")
        logger.info(f"Max entries per cycle: {self.config.max_entries_per_cycle}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Rate limiting: {'enabled' if self.config.rate_limit_enabled else 'disabled'}")
        logger.info(f"Web dashboard: {'enabled' if self.config.web_dashboard_enabled else 'disabled'}")

        if not self.processor.initialize():
            logger.error("Failed to initialize processor, exiting")
            sys.exit(1)

        # Start health server if enabled
        if self.config.health_server_enabled:
            self._health_server = HealthServer(
                host=self.config.health_server_host,
                port=self.config.health_server_port,
                metrics_provider=self.metrics.get_metrics_dict,
                ready_check=lambda: self._running and self.processor._initialized
            )
            if self._health_server.start():
                logger.info(f"Health server: {self._health_server.url}")
            else:
                logger.warning("Health server failed to start")

        # Start web dashboard if enabled
        self._setup_web_dashboard()

        self._running = True
        self._update_health("healthy")
        self._notify_systemd("READY=1")

        # Send startup notification
        if self._notifier:
            status = self.processor.get_status()
            self._notifier.notify_startup(
                device=status.get("device", "unknown"),
                model=status.get("model", "unknown"),
                coverage=status.get("coverage_percent", 0)
            )

        logger.info("Daemon ready, starting main loop")
        last_coverage = 0

        try:
            while not self._shutdown_event.is_set():
                try:
                    result = self.processor.run_cycle()

                    # Process retries if enabled
                    if self.processor.retry_manager:
                        retry_result = self.processor.process_retries()
                        if retry_result["processed"] > 0:
                            logger.info(
                                f"Retries: {retry_result['succeeded']} succeeded, "
                                f"{retry_result['failed']} failed"
                            )

                    if "error" not in result:
                        self._update_health("healthy")
                        status = (
                            f"STATUS=Cycle #{result['cycle_id']}: "
                            f"{result['processed']} processed, "
                            f"{result['coverage_percent']:.1f}% coverage"
                        )
                        self._notify_systemd(status)

                        # Check for coverage milestones
                        if self._notifier and result.get("coverage_percent", 0) > last_coverage:
                            stats = self.processor.get_status()
                            self._notifier.notify_coverage_milestone(
                                coverage=result["coverage_percent"],
                                total_entries=stats.get("total_entries", 0),
                                entries_with_embeddings=stats.get("entries_with_embeddings", 0)
                            )
                            last_coverage = result["coverage_percent"]
                    else:
                        self._update_health(f"error: {result['error']}")
                        if self._notifier:
                            self._notifier.notify_error(result["error"], "Cycle error")

                except Exception as e:
                    logger.error(f"Cycle failed: {e}")
                    self._update_health(f"error: {e}")
                    if self._notifier:
                        self._notifier.notify_error(str(e), "Cycle exception")

                self._shutdown_event.wait(self.config.reindex_interval_seconds)

        except Exception as e:
            logger.error(f"Daemon error: {e}")
        finally:
            self._shutdown()

    def _shutdown(self):
        """Graceful shutdown sequence."""
        logger.info("Starting graceful shutdown...")
        self._notify_systemd("STOPPING=1")

        # 1. Stop accepting new work
        self._running = False

        # 2. Wait for current batch to complete with timeout
        if self.processor.batch_in_progress:
            logger.info("Waiting for current batch to complete...")
            completed = self.processor.wait_for_batch_completion(
                timeout=self.config.shutdown_timeout_seconds
            )
            if not completed:
                logger.warning("Batch completion timed out, forcing shutdown")

        # 3. Flush retry state
        if self.processor.retry_manager:
            try:
                self.processor.retry_manager._save_state()
                logger.info("Retry state saved")
            except Exception as e:
                logger.warning(f"Failed to save retry state: {e}")

        # 4. Flush metrics
        try:
            self.metrics._save()
            logger.info("Metrics flushed")
        except Exception as e:
            logger.warning(f"Failed to flush metrics: {e}")

        # 5. Send shutdown notification
        if self._notifier:
            metrics = self.metrics.get_metrics()
            self._notifier.notify_shutdown(
                reason=self._shutdown_reason or "unknown",
                cycles_completed=metrics.total_cycles,
                entries_processed=metrics.total_entries_processed
            )

        # 6. Stop web dashboard
        if self._web_dashboard:
            self._web_dashboard.stop()

        # 7. Stop health server
        if self._health_server:
            self._health_server.stop()

        # 8. Close processor (backend, model)
        self.processor.close()

        # 9. Clean up health file
        try:
            self._health_file.unlink(missing_ok=True)
        except:
            pass

        logger.info("Graceful shutdown complete")

    def run_once(self):
        logger.info("Running single reindex cycle...")

        if not self.processor.initialize():
            logger.error("Failed to initialize processor")
            return False

        try:
            result = self.processor.run_cycle()

            # Process retries if enabled
            if self.processor.retry_manager:
                retry_result = self.processor.process_retries()
                if retry_result["processed"] > 0:
                    logger.info(
                        f"Retries: {retry_result['succeeded']} succeeded, "
                        f"{retry_result['failed']} failed"
                    )

            if "error" in result:
                logger.error(f"Cycle failed: {result['error']}")
                return False

            logger.info(f"Cycle complete: {result['processed']} entries processed")
            return True

        finally:
            self.processor.close()

    def status(self) -> str:
        return self.metrics.format_status()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AfterImage Embedding Daemon")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--status", action="store_true", help="Show daemon status and exit")
    parser.add_argument("--dashboard", action="store_true", help="Launch CLI monitoring dashboard")
    parser.add_argument("--web-dashboard", action="store_true", help="Enable web dashboard")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--interval", type=int, help="Override reindex interval (seconds)")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--max-entries", type=int, help="Override max entries per cycle")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], help="Override device")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.config:
        config = DaemonConfig.from_yaml(Path(args.config))
    else:
        config = DaemonConfig.load()

    if args.interval:
        config.reindex_interval_seconds = args.interval
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_entries:
        config.max_entries_per_cycle = args.max_entries
    if args.device:
        config.device = args.device
    if args.verbose:
        config.log_level = "DEBUG"
    if args.web_dashboard:
        config.web_dashboard_enabled = True

    if args.dashboard:
        from .dashboard import Dashboard
        dash = Dashboard(metrics_file=config.metrics_file)
        dash.run()
        return 0

    if args.status:
        daemon = EmbeddingDaemon(config)
        print(daemon.status())
        return 0

    if args.once:
        daemon = EmbeddingDaemon(config)
        success = daemon.run_once()
        return 0 if success else 1

    daemon = EmbeddingDaemon(config)
    daemon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
