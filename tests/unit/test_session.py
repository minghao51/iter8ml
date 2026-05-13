"""Tests for ExperimentSession — the primary programmatic API."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars as pl

from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.session import ExperimentSession


def test_init_default_workspace(tmp_path):
    session = ExperimentSession()
    assert session.workspace.root == Path("workspace")
    assert session.tracker is None


def test_init_with_custom_workspace(tmp_path):
    ws = MagicMock()
    ws.root = tmp_path
    session = ExperimentSession(workspace=ws)
    assert session.workspace is ws


def test_init_initializes_workspace(tmp_path):
    ws = MagicMock()
    ExperimentSession(workspace=ws)
    ws.init.assert_called_once()


def test_init_with_tracker(tmp_path):
    ws = MagicMock()
    tracker = MagicMock()
    session = ExperimentSession(workspace=ws, tracker=tracker)
    assert session.tracker is tracker


def test_run_delegates_to_trainer(tmp_path):
    ws = MagicMock()
    config = ExperimentConfig(
        name="test", task=TaskType.CLASSIFICATION, target_col="target", data_path="data.csv"
    )
    df = pl.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})
    expected = {"catboost": {"roc_auc": 0.85}}

    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.Trainer") as MockTrainer:
        mock_trainer = MagicMock()
        mock_trainer.run.return_value = expected
        MockTrainer.return_value = mock_trainer

        result = session.run(config, df)

        MockTrainer.assert_called_once_with(
            config=config,
            workspace=ws,
            tracker=None,
            resume_run_id=None,
            run_leakage_audit=True,
        )
        mock_trainer.run.assert_called_once_with(df)
        assert result == expected


def test_run_passes_resume_and_leakage(tmp_path):
    ws = MagicMock()
    config = ExperimentConfig(
        name="test", task=TaskType.CLASSIFICATION, target_col="target", data_path="data.csv"
    )
    df = pl.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})

    session = ExperimentSession(workspace=ws)
    with patch("iter8ml.session.Trainer") as MockTrainer:
        mock_trainer = MagicMock()
        mock_trainer.run.return_value = {}
        MockTrainer.return_value = mock_trainer

        session.run(config, df, resume_run_id="exp_123", run_leakage_audit=False)

        MockTrainer.assert_called_once_with(
            config=config,
            workspace=ws,
            tracker=None,
            resume_run_id="exp_123",
            run_leakage_audit=False,
        )


def test_leaderboard_returns_dataframe(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.ReportService") as MockReportService:
        mock_report = MagicMock()
        entry = SimpleNamespace(
            model="CatBoost",
            run_id="exp_1",
            primary_metric="roc_auc",
            primary_score=0.85,
            duration_seconds=5.2,
            timestamp="2026-01-01T00:00:00Z",
            task="classification",
        )
        mock_report.leaderboard = [entry]
        mock_service = MagicMock()
        mock_service.build_report.return_value = mock_report
        MockReportService.return_value = mock_service

        result = session.leaderboard()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert result["model"][0] == "CatBoost"
        assert result["primary_score"][0] == 0.85


def test_leaderboard_with_metric_and_limit(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.ReportService") as MockReportService:
        mock_report = MagicMock()
        mock_report.leaderboard = []
        mock_service = MagicMock()
        mock_service.build_report.return_value = mock_report
        MockReportService.return_value = mock_service

        result = session.leaderboard(metric="f1", limit=5)

        mock_service.build_report.assert_called_once_with(metric="f1", limit=5)
        assert isinstance(result, pl.DataFrame)


def test_leaderboard_empty(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.ReportService") as MockReportService:
        mock_report = MagicMock()
        mock_report.leaderboard = []
        mock_service = MagicMock()
        mock_service.build_report.return_value = mock_report
        MockReportService.return_value = mock_service

        result = session.leaderboard()
        assert len(result) == 0


def test_export_delegates(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.ExportService") as MockExportService:
        mock_service = MagicMock()
        mock_service.export.return_value = Path("/out/export.zip")
        MockExportService.return_value = mock_service

        result = session.export("test:classifier", output_dir="/out")

        MockExportService.assert_called_once_with(workspace=ws)
        mock_service.export.assert_called_once_with("test:classifier", output_dir="/out")
        assert result == Path("/out/export.zip")


def test_promote_delegates(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.RegistryService") as MockRegistryService:
        mock_service = MagicMock()
        mock_service.promote_run.return_value = {"status": "promoted"}
        MockRegistryService.return_value = mock_service

        result = session.promote("exp_1", "best_model")

        MockRegistryService.assert_called_once_with(workspace=ws)
        mock_service.promote_run.assert_called_once_with("exp_1", "best_model", ws.experiments_path)
        assert result == {"status": "promoted"}


def test_state_delegates(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.StateObserver") as MockStateObserver:
        mock_observer = MagicMock()
        mock_observer.generate.return_value = "# State content"
        MockStateObserver.return_value = mock_observer

        result = session.state(llm_enabled=True)

        MockStateObserver.assert_called_once_with(workspace=ws, llm_enabled=True)
        mock_observer.generate.assert_called_once()
        assert result == "# State content"


def test_state_default(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)

    with patch("iter8ml.session.StateObserver") as MockStateObserver:
        mock_observer = MagicMock()
        mock_observer.generate.return_value = "# No experiments"
        MockStateObserver.return_value = mock_observer

        result = session.state()

        MockStateObserver.assert_called_once_with(workspace=ws, llm_enabled=False)
        assert result == "# No experiments"


def test_drift_check_delegates(tmp_path):
    ws = MagicMock()
    session = ExperimentSession(workspace=ws)
    ref_df = pl.DataFrame({"x": [1.0, 2.0]})
    live_df = pl.DataFrame({"x": [1.5, 2.5]})

    with patch("iter8ml.engine.pipelines.executor.PipelineExecutor") as MockExecutor:
        mock_executor = MagicMock()
        mock_executor.run_drift.return_value = {"drift_detected": True}
        MockExecutor.return_value = mock_executor

        result = session.drift_check(ref_df, live_df, method="psi")

        assert result == {"drift_detected": True}
        mock_executor.run_drift.assert_called_once_with(ref_df, live_df, drift_method="psi")
