"""Structured experiment reporting utilities."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tabular_blueprint.utils.jsonl import iter_events

LOWER_IS_BETTER_METRICS = {"rmse", "mae", "mse", "log_loss", "loss", "error"}


class LeaderboardEntry(BaseModel):
    model: str
    run_id: str
    cv_scores: dict[str, Any]
    primary_metric: str
    primary_score: float
    duration_seconds: float | str
    timestamp: str
    task: str
    dataset: str
    n_rows: int | str
    n_features: int | str
    artifact_path: str
    hardware: dict[str, Any]
    raw_event: dict[str, Any]


class ExperimentReport(BaseModel):
    leaderboard: list[LeaderboardEntry]
    latest_run: LeaderboardEntry | None
    registry: dict[str, Any]


def metric_higher_is_better(metric_name: str | None) -> bool:
    """Return whether larger values indicate better performance for a metric."""
    if metric_name is None:
        return True
    return metric_name.lower() not in LOWER_IS_BETTER_METRICS


def metric_sort_value(metric_name: str, score: float) -> float:
    """Normalize a score so leaderboard sorting can always be descending."""
    return score if metric_higher_is_better(metric_name) else -score


def metric_value_is_better(metric_name: str | None, candidate: float, incumbent: float) -> bool:
    """Compare two metric values using the metric's optimization direction."""
    if metric_higher_is_better(metric_name):
        return candidate > incumbent
    return candidate < incumbent


def resolve_primary_score(
    cv_scores: dict[str, Any], preferred_metric: str | None = None
) -> tuple[str, float]:
    """Resolve the metric/value pair used for ranking and promotion."""
    if preferred_metric and _is_numeric(cv_scores.get(preferred_metric)):
        return preferred_metric, float(cv_scores[preferred_metric])

    for metric_name in ("roc_auc", "r2"):
        if _is_numeric(cv_scores.get(metric_name)):
            return metric_name, float(cv_scores[metric_name])

    for metric_name, value in cv_scores.items():
        if _is_numeric(value):
            return metric_name, float(value)

    return preferred_metric or "score", 0.0


class ReportService:
    """Builds structured experiment summaries from logs and registry state."""

    def __init__(
        self,
        log_path: str | Path = "workspace/experiments.jsonl",
        registry_path: str | Path = "workspace/registry.json",
    ):
        self.log_path = Path(log_path)
        self.registry_path = Path(registry_path)

    def build_report(self, metric: str | None = None, limit: int | None = None) -> ExperimentReport:
        """Load events and registry and return a canonical report."""
        entries = [self._to_entry(event, metric) for event in self._load_completed_events()]
        leaderboard = sorted(
            entries,
            key=lambda entry: (
                metric_sort_value(entry.primary_metric, entry.primary_score),
                entry.timestamp,
            ),
            reverse=True,
        )

        if limit is not None:
            leaderboard = leaderboard[:limit]

        latest_run = entries[-1] if entries else None
        return ExperimentReport(
            leaderboard=leaderboard,
            latest_run=latest_run,
            registry=self._load_registry(),
        )

    def format_leaderboard_console(self, metric: str | None = None, limit: int = 10) -> str:
        """Format leaderboard as console table (CLI output)."""
        report = self.build_report(metric=metric, limit=limit)
        if not report.leaderboard:
            return "No experiments found."

        lines = [
            "\n# Leaderboard\n",
            "| Rank | Model | Run ID | Primary Metric | Score | Duration |",
            "|---|---|---|---|---|---|",
        ]

        for i, entry in enumerate(report.leaderboard, 1):
            lines.append(
                f"| {i} | {entry.model} | {entry.run_id} | {entry.primary_metric} "
                f"| {entry.primary_score:.4f} | {entry.duration_seconds}s |"
            )
        return "\n".join(lines)

    def format_leaderboard_markdown(
        self, metric: str | None = None, limit: int | None = None
    ) -> str:
        """Format leaderboard as markdown (file output)."""
        report = self.build_report(metric=metric, limit=limit)

        header = (
            "| Rank | Model | Run ID | Primary Metric | Score | All Scores | Duration | Timestamp |"
        )
        separator = "|---|---|---|---|---|---|---|---|"

        lines = ["# Experiment Leaderboard\n", header, separator]

        for i, entry in enumerate(report.leaderboard, 1):
            scores = ", ".join(f"{k}={v:.4f}" for k, v in entry.cv_scores.items())
            row = (
                f"| {i} | {entry.model} | {entry.run_id} | "
                f"{entry.primary_metric} | {entry.primary_score:.4f} | "
                f"{scores} | {entry.duration_seconds}s | {entry.timestamp} |"
            )
            lines.append(row)

        return "\n".join(lines)

    def _load_completed_events(self) -> list[dict[str, Any]]:
        return [
            event for event in iter_events(self.log_path) if event.get("event") == "model_completed"
        ]

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _to_entry(self, event: dict[str, Any], preferred_metric: str | None) -> LeaderboardEntry:
        primary_metric, primary_score = resolve_primary_score(
            event.get("cv_scores", {}), preferred_metric
        )
        return LeaderboardEntry(
            model=event.get("model", "?"),
            run_id=event.get("run_id", "unknown"),
            cv_scores=event.get("cv_scores", {}),
            primary_metric=primary_metric,
            primary_score=primary_score,
            duration_seconds=event.get("duration_seconds", "?"),
            timestamp=event.get("timestamp", "?"),
            task=event.get("task", "?"),
            dataset=event.get("dataset", "unknown"),
            n_rows=event.get("n_rows", "?"),
            n_features=event.get("n_features", "?"),
            artifact_path=event.get("artifact_path", ""),
            hardware=event.get("hardware", {}),
            raw_event=event,
        )


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
