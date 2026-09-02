"""Tests for structured experiment reporting."""

import json
import logging

from iter8ml.services.reporting import (
    ReportService,
    metric_higher_is_better,
    metric_sort_value,
    metric_value_is_better,
    resolve_primary_score,
)
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


def test_build_report_includes_rotated_backups(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    backup = log_path.parent / (log_path.name + ".1")
    backup.write_text(
        json.dumps(
            {
                "event": "model_completed",
                "run_id": "run_rotated",
                "model": "ModelRotated",
                "task": "classification",
                "cv_scores": {"roc_auc": 0.70},
                "timestamp": "2026-04-01T00:00:00Z",
            }
        )
        + "\n"
    )
    log_path.write_text(
        json.dumps(
            {
                "event": "model_completed",
                "run_id": "run_live",
                "model": "ModelLive",
                "task": "classification",
                "cv_scores": {"roc_auc": 0.90},
                "timestamp": "2026-04-02T00:00:00Z",
            }
        )
        + "\n"
    )

    report = ReportService(workspace=ws).build_report()

    assert {entry.model for entry in report.leaderboard} == {"ModelLive", "ModelRotated"}


def test_build_report_dedupe_keeps_newest_timestamp(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    backup = log_path.parent / (log_path.name + ".1")
    stale = {
        "event": "model_completed",
        "run_id": "run_x",
        "model": "ModelX",
        "task": "classification",
        "cv_scores": {"roc_auc": 0.70},
        "artifact_path": "model.pkl",
        "timestamp": "2026-04-01T00:00:00Z",
    }
    backup.write_text(json.dumps(stale) + "\n")
    newest = dict(stale, cv_scores={"roc_auc": 0.95}, timestamp="2026-04-03T00:00:00Z")
    log_path.write_text(json.dumps(newest) + "\n")

    report = ReportService(workspace=ws).build_report()

    assert len(report.leaderboard) == 1
    assert report.leaderboard[0].primary_score == 0.95


def test_build_report_torn_trailing_line_does_not_brick_report(tmp_path, caplog):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    log_path.write_text(
        json.dumps(
            {
                "event": "model_completed",
                "run_id": "run_a",
                "model": "ModelA",
                "task": "classification",
                "cv_scores": {"roc_auc": 0.88},
                "timestamp": "2026-04-02T00:00:00Z",
            }
        )
        + '\n{"event": "trun'
    )

    with caplog.at_level(logging.WARNING):
        report = ReportService(workspace=ws).build_report()

    assert [entry.model for entry in report.leaderboard] == ["ModelA"]
    assert any("trailing" in record.getMessage() for record in caplog.records)


def _make_event(run_id: str, model: str, task: str, cv_scores: dict, timestamp: str) -> dict:
    return {
        "event": "model_completed",
        "run_id": run_id,
        "model": model,
        "task": task,
        "cv_scores": cv_scores,
        "timestamp": timestamp,
    }


def test_build_report_task_groups_never_interleave(tmp_path):
    ws = Workspace(root=tmp_path)
    events = [
        _make_event("r1", "RegSlow", "regression", {"rmse": 3.0}, "2026-04-03T00:00:00Z"),
        _make_event("c1", "ModelLow", "classification", {"roc_auc": 0.81}, "2026-04-04T00:00:00Z"),
        _make_event("r2", "RegGood", "regression", {"rmse": 1.0}, "2026-04-01T00:00:00Z"),
        _make_event("c2", "ModelHigh", "classification", {"roc_auc": 0.92}, "2026-04-02T00:00:00Z"),
    ]
    ws.experiments_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    report = ReportService(workspace=ws).build_report()

    ordered = [(entry.task, entry.model) for entry in report.leaderboard]
    assert ordered == [
        ("classification", "ModelHigh"),
        ("classification", "ModelLow"),
        ("regression", "RegGood"),
        ("regression", "RegSlow"),
    ]


def test_build_report_task_filter(tmp_path):
    ws = Workspace(root=tmp_path)
    events = [
        _make_event("c1", "ModelC", "classification", {"roc_auc": 0.92}, "2026-04-05T00:00:00Z"),
        _make_event("r1", "ModelR", "regression", {"rmse": 1.0}, "2026-04-01T00:00:00Z"),
    ]
    ws.experiments_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    report = ReportService(workspace=ws).build_report(task="regression")

    assert [entry.model for entry in report.leaderboard] == ["ModelR"]
    assert report.latest_run is not None
    assert report.latest_run.model == "ModelR"


def test_build_report_latest_run_by_timestamp_under_shuffled_insertion(tmp_path):
    ws = Workspace(root=tmp_path)
    backup = ws.experiments_path.parent / (ws.experiments_path.name + ".1")
    events_live = [  # newest line FIRST — file order ≠ time order
        _make_event(
            "c_new", "ModelNewest", "classification", {"roc_auc": 0.90}, "2026-04-05T00:00:00Z"
        ),
        _make_event(
            "c_mid", "ModelMid", "classification", {"roc_auc": 0.85}, "2026-04-04T00:00:00Z"
        ),
    ]
    ws.experiments_path.write_text("\n".join(json.dumps(e) for e in events_live) + "\n")
    backup.write_text(
        json.dumps(
            _make_event(
                "c_old", "ModelOld", "classification", {"roc_auc": 0.80}, "2026-04-01T00:00:00Z"
            )
        )
        + "\n"
    )

    report = ReportService(workspace=ws).build_report()

    assert report.latest_run is not None
    assert report.latest_run.model == "ModelNewest"


def test_build_report_unscored_entries_rank_last(tmp_path):
    ws = Workspace(root=tmp_path)
    events = [
        _make_event("r_empty", "NoScores", "classification", {}, "2026-04-06T00:00:00Z"),
        _make_event("c1", "RealHigh", "classification", {"roc_auc": 0.80}, "2026-04-02T00:00:00Z"),
        _make_event("r2", "RealLow", "regression", {"rmse": 5.0}, "2026-04-03T00:00:00Z"),
    ]
    ws.experiments_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    report = ReportService(workspace=ws).build_report()

    # The unscored sentinel (metric "score") must rank after every real result.
    assert [entry.model for entry in report.leaderboard] == ["RealHigh", "RealLow", "NoScores"]


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


def test_metric_helpers_directionality():
    assert metric_higher_is_better("roc_auc") is True
    assert metric_higher_is_better("rmse") is False
    assert metric_higher_is_better(None) is True
    assert metric_sort_value("rmse", 2.0) == -2.0
    assert metric_sort_value("roc_auc", 0.8) == 0.8
    assert metric_value_is_better("rmse", 1.0, 2.0) is True
    assert metric_value_is_better("roc_auc", 0.9, 0.8) is True


def test_resolve_primary_score_fallback_order_and_defaults():
    metric, score = resolve_primary_score(
        {"roc_auc": 0.91, "f1_macro": 0.8}, preferred_metric="f1_macro"
    )
    assert (metric, score) == ("f1_macro", 0.8)

    metric, score = resolve_primary_score({"r2": 0.42, "rmse": 1.5})
    assert (metric, score) == ("r2", 0.42)

    metric, score = resolve_primary_score({"label": "x", "count": 3})
    assert (metric, score) == ("count", 3.0)

    metric, score = resolve_primary_score({"label": "x"}, preferred_metric="f1_macro")
    assert (metric, score) == ("f1_macro", 0.0)


def test_build_report_limit_and_output_formatting(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
    events = [
        {
            "event": "model_completed",
            "run_id": "run_1",
            "model": "M1",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.6},
            "timestamp": "2026-04-03T00:00:00Z",
            "duration_seconds": 1.2,
        },
        {
            "event": "model_completed",
            "run_id": "run_2",
            "model": "M2",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.9},
            "timestamp": "2026-04-04T00:00:00Z",
            "duration_seconds": 2.3,
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    svc = ReportService(workspace=ws)

    report = svc.build_report(limit=1)
    assert len(report.leaderboard) == 1
    assert report.leaderboard[0].model == "M2"

    console = svc.format_leaderboard_console(limit=1)
    assert "| Rank | Model | Run ID |" in console
    assert "M2" in console

    markdown = svc.format_leaderboard_markdown(limit=1)
    assert "| All Scores |" in markdown
    assert "roc_auc=0.9000" in markdown
