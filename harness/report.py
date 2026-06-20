"""Pretty-printed report with regression detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

REPORT_DIR = Path("runs")


@dataclass
class RunReport:
    timestamp: str
    pipeline: str
    n_questions: int
    faithfulness: float
    answer_relevance: float
    retrieval_recall: float
    latency_p50: float
    latency_p95: float
    total_cost_usd: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# Targets — fail CI if any metric drops below these
TARGETS = {
    "faithfulness": 0.90,
    "answer_relevance": 0.85,
    "retrieval_recall": 0.80,
}

# Regression threshold — fail CI if any metric drops by more than this vs prior run
REGRESSION_THRESHOLD = 0.05


def print_report(current: RunReport, prior: RunReport | None) -> bool:
    """Render the report. Returns True if metrics passed, False on any regression / target miss."""
    console = Console()

    table = Table(title=f"RAG Eval Harness — {current.timestamp}", show_lines=True)
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("vs Prior", justify="right")
    table.add_column("Pass", justify="center")

    pass_all = True
    for metric in ("faithfulness", "answer_relevance", "retrieval_recall"):
        score = getattr(current, metric)
        target = TARGETS[metric]
        delta = score - getattr(prior, metric) if prior else None

        meets_target = score >= target
        not_regressed = delta is None or delta >= -REGRESSION_THRESHOLD
        ok = meets_target and not_regressed
        pass_all = pass_all and ok

        delta_str = f"{delta:+.3f}" if delta is not None else "—"
        if delta is not None and delta < -REGRESSION_THRESHOLD:
            delta_str = f"[red]{delta_str}[/red]"

        table.add_row(
            metric.replace("_", " ").title(),
            f"{score:.3f}",
            f"≥ {target:.2f}",
            delta_str,
            "[green]✓[/green]" if ok else "[red]✗[/red]",
        )

    console.print(table)
    console.print(f"Latency p50: {current.latency_p50:.2f}s  ·  p95: {current.latency_p95:.2f}s")
    console.print(f"Total cost:  ${current.total_cost_usd:.4f}")
    if pass_all:
        console.print("[bold green]All metrics passed.[/bold green]")
    else:
        console.print("[bold red]Regression or target miss detected.[/bold red]")
    return pass_all


def load_latest_prior(current_timestamp: str) -> RunReport | None:
    REPORT_DIR.mkdir(exist_ok=True)
    runs = sorted(REPORT_DIR.glob("*.json"))
    runs = [r for r in runs if current_timestamp not in r.name]
    if not runs:
        return None
    data = json.loads(runs[-1].read_text())
    return RunReport(**data)


def save_run(report: RunReport) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{report.timestamp}.json"
    path.write_text(json.dumps(report.as_dict(), indent=2))
    return path
