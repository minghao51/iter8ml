"""Tests for JSONLTracker log rotation, WandbTracker, and MLflowTracker."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iter8ml.engine.tracker import JSONLTracker, MLflowTracker, WandbTracker


def test_log_rotation_when_file_exceeds_limit():
    """Test that log rotation occurs when file size exceeds limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(
            log_path=str(log_path), max_file_size_mb=0.001, backup_count=3
        )  # 1KB limit

        # Write events until we exceed the limit
        for i in range(100):
            tracker.log_event({"event": "test", "iteration": i})

        # Check that rotation occurred
        assert log_path.exists()  # Current log
        backup_1 = log_path.with_suffix(".jsonl.1")
        assert backup_1.exists(), "First backup should exist after rotation"


def test_multiple_rotations_keep_specified_backups():
    """Test that only specified number of backups are kept."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path), max_file_size_mb=0.001, backup_count=2)

        # Write enough events to cause multiple rotations
        for i in range(300):
            tracker.log_event({"event": "test", "iteration": i})

        # Check that we have the correct number of backups
        assert log_path.exists()  # Current log
        assert log_path.with_suffix(".jsonl.1").exists()  # First backup
        assert log_path.with_suffix(".jsonl.2").exists()  # Second backup
        assert not log_path.with_suffix(".jsonl.3").exists()  # Third backup should not exist


def test_thread_safe_writes():
    """Test that concurrent writes are thread-safe."""
    import threading

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path), max_file_size_mb=10.0, backup_count=3)

        def write_events(thread_id: int):
            for i in range(50):
                tracker.log_event({"event": "test", "thread": thread_id, "iteration": i})

        # Launch multiple threads writing concurrently
        threads = [threading.Thread(target=write_events, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all events were written
        with open(log_path) as f:
            events = [json.loads(line) for line in f if line.strip()]
        assert len(events) == 250  # 5 threads * 50 events


def test_from_workspace(tmp_path):
    class FakeWorkspace:
        experiments_path = tmp_path / "exp.jsonl"
        experiments_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = JSONLTracker.from_workspace(FakeWorkspace())
    assert tracker.log_path == tmp_path / "exp.jsonl"


def test_from_workspace_passes_kwargs(tmp_path):
    class FakeWorkspace:
        experiments_path = tmp_path / "exp.jsonl"
        experiments_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = JSONLTracker.from_workspace(FakeWorkspace(), max_file_size_mb=0.5, backup_count=2)
    assert tracker.log_path == tmp_path / "exp.jsonl"
    assert tracker.max_file_size_bytes == int(0.5 * 1024 * 1024)
    assert tracker.backup_count == 2


def test_log_metrics_delegates_to_log_event():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path))
        tracker.current_run_id = "run_1"
        tracker.log_metrics({"roc_auc": 0.85, "accuracy": 0.9}, step=1)
        with open(log_path) as f:
            event = json.loads(f.readline())
        assert event["event"] == "metrics"
        assert event["metrics"]["roc_auc"] == 0.85
        assert event["step"] == 1


def test_log_params_delegates_to_log_event():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path))
        tracker.current_run_id = "run_1"
        tracker.log_params({"depth": 6, "lr": 0.1})
        with open(log_path) as f:
            event = json.loads(f.readline())
        assert event["event"] == "params"
        assert event["params"]["depth"] == 6


def test_log_artifact_delegates_to_log_event():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path))
        tracker.current_run_id = "run_1"
        tracker.log_artifact("/tmp/model.pkl")
        with open(log_path) as f:
            event = json.loads(f.readline())
        assert event["event"] == "artifact"
        assert event["path"] == "/tmp/model.pkl"


def test_finish_logs_completed_and_clears_run_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path))
        tracker.current_run_id = "run_1"
        tracker.finish()
        with open(log_path) as f:
            event = json.loads(f.readline())
        assert event["event"] == "run_completed"
        assert tracker.current_run_id is None


def test_log_event_adds_run_id_and_timestamp():
    """Test that tracker adds run_id and timestamp to events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        tracker = JSONLTracker(log_path=str(log_path))
        tracker.current_run_id = "test_run_123"

        tracker.log_event({"event": "custom_event", "data": "test"})

        with open(log_path) as f:
            event = json.loads(f.readline())

        assert event["run_id"] == "test_run_123"
        assert "timestamp" in event
        assert event["event"] == "custom_event"
        assert event["data"] == "test"


# --- WandbTracker tests ---


@patch.dict("sys.modules", {"wandb": MagicMock()})
def test_wandb_tracker_init():
    import wandb

    wandb.init = MagicMock()
    WandbTracker(project="test_project")
    wandb.init.assert_called_once_with(project="test_project")


def test_wandb_tracker_init_missing_dep():
    with patch.dict("sys.modules", {"wandb": None}):
        import sys

        orig = sys.modules.get("wandb")
        sys.modules["wandb"] = None
        try:
            with pytest.raises(ImportError, match="wandb is required"):
                WandbTracker()
        finally:
            if orig:
                sys.modules["wandb"] = orig
            else:
                del sys.modules["wandb"]


@patch.dict("sys.modules", {"wandb": MagicMock()})
def test_wandb_log_metrics():
    import wandb

    wandb.init = MagicMock()
    tracker = WandbTracker()
    tracker.wandb = wandb
    tracker.log_metrics({"acc": 0.9}, step=1)
    wandb.log.assert_called_once_with({"acc": 0.9}, step=1)


@patch.dict("sys.modules", {"wandb": MagicMock()})
def test_wandb_log_params():
    import wandb

    wandb.init = MagicMock()
    mock_run = MagicMock()
    wandb.init.return_value = mock_run
    tracker = WandbTracker()
    tracker.run = mock_run
    tracker.log_params({"depth": 6})
    mock_run.config.update.assert_called_once_with({"depth": 6})


@patch.dict("sys.modules", {"wandb": MagicMock()})
def test_wandb_log_artifact():
    import wandb

    wandb.init = MagicMock()
    wandb.Artifact = MagicMock()
    mock_artifact = MagicMock()
    wandb.Artifact.return_value = mock_artifact
    mock_run = MagicMock()
    wandb.init.return_value = mock_run
    tracker = WandbTracker()
    tracker.run = mock_run
    tracker.wandb = wandb
    tracker.log_artifact("/tmp/model.pkl")
    wandb.Artifact.assert_called_once()
    mock_artifact.add_file.assert_called_once_with("/tmp/model.pkl")


@patch.dict("sys.modules", {"wandb": MagicMock()})
def test_wandb_finish():
    import wandb

    wandb.init = MagicMock()
    mock_run = MagicMock()
    wandb.init.return_value = mock_run
    tracker = WandbTracker()
    tracker.run = mock_run
    tracker.finish()
    mock_run.finish.assert_called_once()


# --- MLflowTracker tests ---


@patch.dict("sys.modules", {"mlflow": MagicMock()})
def test_mlflow_tracker_init():
    import mlflow

    mlflow.set_tracking_uri = MagicMock()
    mlflow.set_experiment = MagicMock()
    mlflow.start_run = MagicMock()
    MLflowTracker(tracking_uri="http://localhost:5000", experiment_name="test_exp")
    mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    mlflow.set_experiment.assert_called_once_with("test_exp")
    mlflow.start_run.assert_called_once()


def test_mlflow_tracker_init_missing_dep():
    with patch.dict("sys.modules", {"mlflow": None}):
        import sys

        orig = sys.modules.get("mlflow")
        sys.modules["mlflow"] = None
        try:
            with pytest.raises(ImportError, match="mlflow is required"):
                MLflowTracker()
        finally:
            if orig:
                sys.modules["mlflow"] = orig
            else:
                del sys.modules["mlflow"]


@patch.dict("sys.modules", {"mlflow": MagicMock()})
def test_mlflow_log_metrics():
    import mlflow

    mlflow.log_metrics = MagicMock()
    tracker = MLflowTracker()
    tracker.mlflow = mlflow
    tracker.log_metrics({"rmse": 0.5}, step=1)
    mlflow.log_metrics.assert_called_once_with({"rmse": 0.5}, step=1)


@patch.dict("sys.modules", {"mlflow": MagicMock()})
def test_mlflow_log_params():
    import mlflow

    mlflow.log_params = MagicMock()
    tracker = MLflowTracker()
    tracker.mlflow = mlflow
    tracker.log_params({"alpha": 0.1})
    mlflow.log_params.assert_called_once_with({"alpha": 0.1})


@patch.dict("sys.modules", {"mlflow": MagicMock()})
def test_mlflow_log_artifact():
    import mlflow

    mlflow.log_artifact = MagicMock()
    tracker = MLflowTracker()
    tracker.mlflow = mlflow
    tracker.log_artifact("/tmp/model.pkl")
    mlflow.log_artifact.assert_called_once_with("/tmp/model.pkl")


@patch.dict("sys.modules", {"mlflow": MagicMock()})
def test_mlflow_finish():
    import mlflow

    mlflow.end_run = MagicMock()
    tracker = MLflowTracker()
    tracker.mlflow = mlflow
    tracker.finish()
    mlflow.end_run.assert_called_once()
