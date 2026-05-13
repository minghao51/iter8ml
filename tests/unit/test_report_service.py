"""Tests for structured experiment reporting."""

import json

from iter8ml.services.reporting import ReportService
from iter8ml.workspace import Workspace


def test_build_report_empty(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.experiments_path.touch()

    report = ReportService(workspace=ws).build_report()

    assert report.leaderboard == []
    assert report.latest_run is None
    assert report.registry == {}


def test_build_report_classification_orders_by_primary_score(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    events = [
        {
            "event": "model_completed",
            "run_id": "run_low",
            "model": "ModelLow",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.81, "f1_macro": 0.7},
            "timestamp": "2026-04-04T00:00:00Z",
        },
        {
            "event": "model_completed",
            "run_id": "run_high",
            "model": "ModelHigh",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.92, "f1_macro": 0.6},
            "timestamp": "2026-04-05T00:00:00Z",
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    report = ReportService(workspace=ws).build_report()

    assert [entry.model for entry in report.leaderboard] == ["ModelHigh", "ModelLow"]
    assert report.latest_run.model == "ModelHigh"


def test_build_report_regression_uses_r2_by_default(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    events = [
        {
            "event": "model_completed",
            "run_id": "run_a",
            "model": "ModelA",
            "task": "regression",
            "cv_scores": {"rmse": 10.0, "r2": 0.4},
            "timestamp": "2026-04-04T00:00:00Z",
        },
        {
            "event": "model_completed",
            "run_id": "run_b",
            "model": "ModelB",
            "task": "regression",
            "cv_scores": {"rmse": 2.0, "r2": 0.9},
            "timestamp": "2026-04-05T00:00:00Z",
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    report = ReportService(workspace=ws).build_report()

    assert report.leaderboard[0].model == "ModelB"
    assert report.leaderboard[0].primary_metric == "r2"
    assert report.leaderboard[0].primary_score == 0.9


def test_build_report_respects_explicit_metric_override(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    events = [
        {
            "event": "model_completed",
            "run_id": "run_fast",
            "model": "Fast",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.8, "f1_macro": 0.95},
            "timestamp": "2026-04-04T00:00:00Z",
        },
        {
            "event": "model_completed",
            "run_id": "run_balanced",
            "model": "Balanced",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.9, "f1_macro": 0.85},
            "timestamp": "2026-04-05T00:00:00Z",
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    report = ReportService(workspace=ws).build_report(metric="f1_macro")

    assert [entry.model for entry in report.leaderboard] == ["Fast", "Balanced"]
    assert report.leaderboard[0].primary_metric == "f1_macro"


def test_build_report_sorts_lower_is_better_metric_ascending(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    events = [
        {
            "event": "model_completed",
            "run_id": "run_worse",
            "model": "Worse",
            "task": "regression",
            "cv_scores": {"rmse": 10.0},
            "timestamp": "2026-04-04T00:00:00Z",
        },
        {
            "event": "model_completed",
            "run_id": "run_better",
            "model": "Better",
            "task": "regression",
            "cv_scores": {"rmse": 2.0},
            "timestamp": "2026-04-05T00:00:00Z",
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    report = ReportService(workspace=ws).build_report(metric="rmse")

    assert [entry.model for entry in report.leaderboard] == ["Better", "Worse"]
    assert report.leaderboard[0].primary_metric == "rmse"
    assert report.leaderboard[0].primary_score == 2.0


def test_build_report_includes_registry_summary(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.experiments_path.write_text("")
    ws.registry_path.write_text(json.dumps({"champion": {"model": "CatBoost", "score": 0.95}}))

    report = ReportService(workspace=ws).build_report()

    assert report.registry["champion"]["model"] == "CatBoost"
