"""Structured experiment reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from iter8ml.utils.io import iter_events

if TYPE_CHECKING:
    from iter8ml.workspace import Workspace

LOWER_IS_BETTER_METRICS = {"rmse", "mae", "mse", "log_loss", "loss", "error"}


class LeaderboardEntry(BaseModel):
    """A single model run entry in the experiment leaderboard."""

    model: str
    run_id: str
    cv_scores: dict[str, Any]
    primary_metric: str
    primary_score: float
    cv_std: dict[str, Any] = Field(default_factory=dict)
    calibration: str | None = None
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
    """Canonical experiment report with leaderboard and registry state."""

    leaderboard: list[LeaderboardEntry]
    latest_run: LeaderboardEntry | None
    registry: dict[str, Any]


def metric_higher_is_better(metric_name: str | None) -> bool:
    """Return whether larger values indicate better performance for a metric."""
    if metric_name is None:
        return True
    if metric_name == "score":  # resolve_primary_score sentinel: no direction
        return True
    from iter8ml.engine.evaluator import metric_lower_is_better

    return not metric_lower_is_better(metric_name)


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

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.log_path = workspace.experiments_path

    def build_report(
        self,
        metric: str | None = None,
        limit: int | None = None,
        task: str | None = None,
    ) -> ExperimentReport:
        """Load events and registry and return a canonical report.

        task optionally scopes the leaderboard (and latest_run) to one task;
        by default all tasks are included but task-isolated: classification
        and regression entries never interleave in one ranking.
        """
        loaded = [self._to_entry(event, metric) for event in self._load_completed_events()]
        entries = [entry for entry in loaded if task is None or entry.task == task]
        # Ranking (ascending key): unscored entries (resolve_primary_score
        # sentinel metric "score") group last — their 0.0 sentinel must never
        # outrank real results; tasks form contiguous blocks so roc_auc and
        # rmse/r2 never share a ranking; within a task, best-first
        # (metric_sort_value normalizes to descending-is-best, hence the
        # negation). Timestamps are pre-sorted newest-first; the stable sort
        # keeps that order for exact ties.
        entries.sort(key=lambda entry: entry.timestamp, reverse=True)
        entries.sort(
            key=lambda entry: (
                1 if entry.primary_metric == "score" else 0,
                entry.task,
                -metric_sort_value(entry.primary_metric, entry.primary_score),
            )
        )

        leaderboard = entries[:limit] if limit is not None else entries

        latest_run: LeaderboardEntry | None = None
        if entries:
            try:
                latest_run = max(entries, key=lambda entry: entry.timestamp)
            except (TypeError, ValueError):
                # Non-comparable timestamps: fall back to load order, where
                # the live log (newest events) is read before rotated backups.
                latest_run = loaded[0]
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
                f"| {i} | {entry.model}{_calibration_marker(entry)} | {entry.run_id} "
                f"| {entry.primary_metric} "
                f"| {_fmt_score(entry.primary_score, entry.cv_std.get(entry.primary_metric))} "
                f"| {entry.duration_seconds}s |"
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

        calibrated_present = False
        for i, entry in enumerate(report.leaderboard, 1):
            calibrated_present = calibrated_present or entry.calibration is not None
            scores = ", ".join(
                f"{k}={_fmt_score(v, entry.cv_std.get(k))}" for k, v in entry.cv_scores.items()
            )
            row = (
                f"| {i} | {entry.model}{_calibration_marker(entry)} | {entry.run_id} | "
                f"{entry.primary_metric} | "
                f"{_fmt_score(entry.primary_score, entry.cv_std.get(entry.primary_metric))} | "
                f"{scores} | {entry.duration_seconds}s | {entry.timestamp} |"
            )
            lines.append(row)

        if calibrated_present:
            lines.append("")
            lines.append(
                "*: scores are cross-validation metrics computed *before* probability "
                "calibration; the saved artifact is the calibrated model."
            )

        return "\n".join(lines)

    def _iter_log_files(self) -> list[Path]:
        """Live event log first, then rotated backups, newest backup first.

        Rotation (JSONLTracker) names backups ``<name>.jsonl.1`` … ``.N``
        where ``.1`` is the most recent backup; the count is whatever the
        tracker was configured with, so discover instead of hardcoding.
        """
        files: list[Path] = [self.log_path] if self.log_path.exists() else []
        parent = self.log_path.parent
        if not parent.exists():
            return files
        prefix = self.log_path.name + "."
        backups: list[tuple[int, Path]] = []
        for candidate in parent.iterdir():
            suffix = candidate.name[len(prefix) :]
            if candidate.name.startswith(prefix) and suffix.isdigit():
                backups.append((int(suffix), candidate))
        files.extend(path for _, path in sorted(backups, reverse=True))
        return files

    def _load_completed_events(self) -> list[dict[str, Any]]:
        """Completed-model events from the live log plus rotated backups.

        Files are read with torn-tail recovery; duplicates (same run, model,
        artifact — e.g. an event that survived a rotation boundary) collapse
        to the entry with the latest timestamp.
        """
        completed: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        for log_file in self._iter_log_files():
            for event in iter_events(log_file, on_error="skip_trailing"):
                if event.get("event") != "model_completed":
                    continue
                key = (event.get("run_id"), event.get("model"), event.get("artifact_path"))
                existing = completed.get(key)
                if existing is None or str(event.get("timestamp", "")) > str(
                    existing.get("timestamp", "")
                ):
                    completed[key] = event
        return list(completed.values())

    def _load_registry(self) -> dict[str, Any]:
        from iter8ml.services.registry import RegistryService

        return RegistryService(workspace=self.workspace).get_all()

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
            cv_std=event.get("cv_std", {}) or {},
            calibration=event.get("calibration"),
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


def _fmt_score(value: Any, std: Any = None) -> str:
    """Format a metric value (optionally with ±std); never raises on odd types."""
    if not _is_numeric(value):
        return str(value)
    if _is_numeric(std) and std:
        return f"{value:.4f} ±{std:.4f}"
    return f"{value:.4f}"


def _calibration_marker(entry: LeaderboardEntry) -> str:
    return "*" if entry.calibration else ""


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
