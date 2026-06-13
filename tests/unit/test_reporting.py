"""Tests for reporting metric directionality, score resolution, and formatting."""

import json

from iter8ml.services.reporting import (
    ReportService,
    metric_higher_is_better,
    metric_sort_value,
    metric_value_is_better,
    resolve_primary_score,
)
from iter8ml.workspace import Workspace


def test_metric_higher_is_better_roc_auc():
    assert metric_higher_is_better("roc_auc") is True


def test_metric_higher_is_better_rmse():
    assert metric_higher_is_better("rmse") is False


def test_metric_higher_is_better_none():
    assert metric_higher_is_better(None) is True


def test_metric_sort_value_higher():
    assert metric_sort_value("roc_auc", 0.85) == 0.85


def test_metric_sort_value_lower():
    assert metric_sort_value("rmse", 3.0) == -3.0


def test_metric_value_is_better_higher():
    assert metric_value_is_better("roc_auc", 0.9, 0.8) is True


def test_metric_value_is_better_lower():
    assert metric_value_is_better("rmse", 1.0, 2.0) is True


def test_resolve_primary_score_preferred():
    metric, score = resolve_primary_score(
        {"roc_auc": 0.91, "f1_macro": 0.75}, preferred_metric="f1_macro"
    )
    assert (metric, score) == ("f1_macro", 0.75)


def test_resolve_primary_score_fallback_roc_auc():
    metric, score = resolve_primary_score({"rmse": 1.5, "roc_auc": 0.88})
    assert (metric, score) == ("roc_auc", 0.88)


def test_resolve_primary_score_no_numeric():
    metric, score = resolve_primary_score({"label": "x"}, preferred_metric="f1_macro")
    assert (metric, score) == ("f1_macro", 0.0)


def test_build_report_empty(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.experiments_path.touch()
    report = ReportService(workspace=ws).build_report()
    assert report.leaderboard == []
    assert report.latest_run is None
    assert report.registry == {}


def _write_events(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_format_leaderboard_console(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            {
                "event": "model_completed",
                "run_id": "r1",
                "model": "ModelA",
                "task": "classification",
                "cv_scores": {"roc_auc": 0.88},
                "timestamp": "2026-01-01T00:00:00Z",
                "duration_seconds": 1.5,
            },
        ],
    )
    console = ReportService(workspace=ws).format_leaderboard_console()
    assert "| Rank | Model | Run ID | Primary Metric | Score | Duration |" in console
    assert "ModelA" in console
    assert "0.8800" in console


def test_format_leaderboard_markdown(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            {
                "event": "model_completed",
                "run_id": "r2",
                "model": "ModelB",
                "task": "classification",
                "cv_scores": {"roc_auc": 0.92},
                "timestamp": "2026-01-02T00:00:00Z",
                "duration_seconds": 2.1,
            },
        ],
    )
    md = ReportService(workspace=ws).format_leaderboard_markdown()
    assert "| All Scores | Duration | Timestamp |" in md
    assert "ModelB" in md
    assert "roc_auc=0.9200" in md
