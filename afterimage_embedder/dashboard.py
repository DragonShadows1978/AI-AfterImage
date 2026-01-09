"""Rich CLI Dashboard for AfterImage Embedding Daemon monitoring."""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def check_rich_available() -> bool:
    if not RICH_AVAILABLE:
        print("Error: 'rich' library not installed.")
        print("Install with: pip install rich")
        return False
    return True


class Dashboard:
    """Real-time CLI dashboard for monitoring the embedding daemon."""

    def __init__(self, metrics_file: Optional[Path] = None, refresh_rate: float = 1.0):
        self.metrics_file = metrics_file or Path.home() / ".afterimage" / "embedder_metrics.json"
        self.refresh_rate = refresh_rate
        self.console = Console() if RICH_AVAILABLE else None
        self._running = False

    def load_metrics(self) -> Dict[str, Any]:
        if not self.metrics_file.exists():
            return {}
        try:
            with open(self.metrics_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _create_header_panel(self, metrics: Dict[str, Any]) -> Panel:
        device = metrics.get("device", "unknown")
        model = metrics.get("model_name", "unknown")
        started = metrics.get("daemon_started_at", "")
        if started:
            started = started[:19].replace("T", " ")
        coverage = metrics.get("current_coverage_percent", 0)
        status_color = "green" if coverage >= 99 else "yellow" if coverage >= 50 else "red"
        header_text = Text()
        header_text.append("Status: ", style="bold")
        header_text.append("RUNNING", style=f"bold {status_color}")
        header_text.append("    Device: ", style="bold")
        header_text.append(f"{device}", style="cyan")
        header_text.append("\nCoverage: ", style="bold")
        header_text.append(f"{coverage:.1f}%", style=status_color)
        header_text.append("    Model: ", style="bold")
        header_text.append(f"{model}", style="cyan")
        header_text.append("\nStarted: ", style="bold")
        header_text.append(f"{started or 'N/A'}", style="dim")
        return Panel(header_text, title="[bold blue]AfterImage Embedding Daemon[/]", border_style="blue")

    def _create_coverage_bar(self, metrics: Dict[str, Any]) -> Panel:
        total = metrics.get("total_entries_in_kb", 0)
        with_embeddings = metrics.get("entries_with_embeddings", 0)
        coverage = metrics.get("current_coverage_percent", 0)
        bar_width = 50
        filled = int(bar_width * coverage / 100)
        bar = "[green]" + "█" * filled + "[/][dim]" + "░" * (bar_width - filled) + "[/]"
        text = f"{bar}  {with_embeddings:,}/{total:,} ({coverage:.1f}%)"
        return Panel(Text.from_markup(text), title="[bold]Coverage Progress[/]", border_style="green")

    def _create_cycles_table(self, metrics: Dict[str, Any]) -> Table:
        table = Table(title="Recent Cycles", border_style="cyan", header_style="bold cyan")
        table.add_column("Cycle", justify="right", style="cyan", width=6)
        table.add_column("Processed", justify="right", width=10)
        table.add_column("Failed", justify="right", width=8)
        table.add_column("Priority", justify="right", width=8)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Rate", justify="right", width=8)
        table.add_column("Time", width=10)
        recent = metrics.get("recent_cycles", [])[-10:]
        for cycle in reversed(recent):
            cycle_id = cycle.get("cycle_id", 0)
            processed = cycle.get("entries_processed", 0)
            failed = cycle.get("entries_failed", 0)
            priority = cycle.get("priority_entries", 0)
            duration = cycle.get("duration_seconds", 0)
            completed = cycle.get("completed_at", "")
            rate = processed / duration if duration > 0 else 0
            time_str = completed[11:19] if len(completed) > 19 else "-"
            failed_style = "red" if failed > 0 else "dim"
            table.add_row(
                f"#{cycle_id}",
                str(processed),
                f"[{failed_style}]{failed}[/]",
                str(priority),
                f"{duration:.1f}s",
                f"{rate:.1f}/s",
                time_str
            )
        return table

    def _create_stats_panel(self, metrics: Dict[str, Any]) -> Panel:
        total_cycles = metrics.get("total_cycles", 0)
        total_processed = metrics.get("total_entries_processed", 0)
        total_failed = metrics.get("total_entries_failed", 0)
        avg_per_cycle = metrics.get("avg_entries_per_cycle", 0)
        avg_duration = metrics.get("avg_cycle_duration_seconds", 0)
        throughput = avg_per_cycle / avg_duration if avg_duration > 0 else 0
        stats_text = Text()
        stats_text.append("Throughput: ", style="bold")
        stats_text.append(f"{throughput:.1f}/s", style="cyan")
        stats_text.append("  │  Avg Cycle: ", style="dim")
        stats_text.append(f"{avg_duration:.1f}s", style="cyan")
        stats_text.append("  │  Total Cycles: ", style="dim")
        stats_text.append(f"{total_cycles}", style="cyan")
        stats_text.append("\nTotal Processed: ", style="bold")
        stats_text.append(f"{total_processed:,}", style="green")
        stats_text.append("  │  Total Failed: ", style="dim")
        failed_style = "red" if total_failed > 0 else "green"
        stats_text.append(f"{total_failed:,}", style=failed_style)
        return Panel(stats_text, title="[bold]Statistics[/]", border_style="yellow")

    def _create_layout(self, metrics: Dict[str, Any]) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="coverage", size=3),
            Layout(name="cycles", size=14),
            Layout(name="stats", size=4)
        )
        layout["header"].update(self._create_header_panel(metrics))
        layout["coverage"].update(self._create_coverage_bar(metrics))
        layout["cycles"].update(self._create_cycles_table(metrics))
        layout["stats"].update(self._create_stats_panel(metrics))
        return layout

    def run(self):
        if not check_rich_available():
            return
        self._running = True
        try:
            with Live(self._create_layout({}), console=self.console, refresh_per_second=1) as live:
                while self._running:
                    metrics = self.load_metrics()
                    live.update(self._create_layout(metrics))
                    time.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.console.print("\n[dim]Dashboard stopped[/]")

    def stop(self):
        self._running = False

    def print_status(self):
        """Print a single status snapshot (non-live mode)."""
        if not check_rich_available():
            return
        metrics = self.load_metrics()
        if not metrics:
            self.console.print("[yellow]No metrics data available[/]")
            return
        self.console.print(self._create_header_panel(metrics))
        self.console.print(self._create_coverage_bar(metrics))
        self.console.print(self._create_cycles_table(metrics))
        self.console.print(self._create_stats_panel(metrics))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AfterImage Embedder Dashboard")
    parser.add_argument("--metrics-file", type=str, help="Path to metrics JSON file")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh rate in seconds")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    args = parser.parse_args()
    metrics_file = Path(args.metrics_file) if args.metrics_file else None
    dashboard = Dashboard(metrics_file=metrics_file, refresh_rate=args.refresh)
    if args.once:
        dashboard.print_status()
    else:
        dashboard.run()


if __name__ == "__main__":
    main()
