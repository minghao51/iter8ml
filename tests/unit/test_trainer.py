"""Tests for Trainer class."""

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.constants import TaskType
from tabular_blueprint.engine.trainer import Trainer


def test_trainer_uses_registry_service(tmp_path, monkeypatch):
    """Verify trainer uses RegistryService for updates."""

    # Track calls to RegistryService
    registry_calls = []

    original_registry = None

    class MockRegistryService:
        def __init__(self, registry_path):
            self.registry_path = registry_path

        def update_if_better(self, key, model_name, run_id, score, artifact_path, metric_name=None):
            registry_calls.append(
                {
                    "key": key,
                    "model_name": model_name,
                    "run_id": run_id,
                    "score": score,
                    "artifact_path": artifact_path,
                    "metric_name": metric_name,
                }
            )
            return True

    config = ExperimentConfig(
        name="test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        workspace_dir=tmp_path,
    )

    # Monkey patch on the model_trainer module where RegistryService is imported
    import tabular_blueprint.engine.model_trainer as mt

    original_registry = mt.RegistryService
    monkeypatch.setattr(mt, "RegistryService", MockRegistryService)

    trainer = Trainer(config)

    result = trainer._model_trainer._update_champion(
        model_name="catboost",
        run_id="test_run",
        score=0.95,
        artifact_path="/tmp/test.pkl",
        metric_name="roc_auc",
    )

    assert result is True

    assert len(registry_calls) == 1
    call = registry_calls[0]
    assert "test:classification" in call["key"]
    assert call["model_name"] == "catboost"
    assert call["run_id"] == "test_run"
    assert call["score"] == 0.95
    assert call["artifact_path"] == "/tmp/test.pkl"
    assert call["metric_name"] == "roc_auc"

    monkeypatch.setattr(mt, "RegistryService", original_registry)


def test_omp_threads_configurable(monkeypatch):
    """Test that OMP threads can be configured via HardwareProfile."""
    import os

    from tabular_blueprint.config import HardwareProfile

    # Test default
    thread_count = HardwareProfile.configure_omp_threads()
    assert os.environ.get("OMP_NUM_THREADS") == str(thread_count)

    # Test custom value
    custom_count = HardwareProfile.configure_omp_threads(threads=4)
    assert custom_count == 4
    assert os.environ.get("OMP_NUM_THREADS") == "4"


def test_hamilton_fallback_logs_warning_event(monkeypatch, tmp_path):
    import polars as pl

    from tabular_blueprint.engine.tracker import JSONLTracker
    from tabular_blueprint.engine.trainer import PipelineMode

    class FailingExecutor:
        available = True

        def run_training(self, **kwargs):
            raise RuntimeError("Hamilton failure")

    config = ExperimentConfig(
        name="test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
    )
    trainer = Trainer(config, tracker=JSONLTracker(log_path=str(tmp_path / "events.jsonl")))
    trainer.tracker.current_run_id = "exp_test"
    monkeypatch.setattr(
        "tabular_blueprint.engine.trainer.PipelineExecutor",
        lambda mode=PipelineMode.TRAINING: FailingExecutor(),
    )

    events = []
    original_log_event = trainer.tracker.log_event

    def _capture(event):
        events.append(event)
        original_log_event(event)

    monkeypatch.setattr(trainer.tracker, "log_event", _capture)
    result = trainer._try_hamilton_training(pl.DataFrame({"a": [1], "target": [0]}), "exp_test")

    assert result is None
    assert any(event.get("event") == "hamilton_fallback" for event in events)
