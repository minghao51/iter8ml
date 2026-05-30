"""Tests for Trainer class."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.engine.trainer import Trainer
from iter8ml.exceptions import TrainerStatePublishError
from iter8ml.workspace import Workspace


def test_trainer_init(tmp_path):
    config = ExperimentConfig(
        name="test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
    )
    trainer = Trainer(config, workspace=Workspace(root=tmp_path))
    assert trainer.config is config


def test_omp_threads_configurable(monkeypatch):
    import os

    from iter8ml.config import HardwareProfile

    thread_count = HardwareProfile.configure_omp_threads()
    assert os.environ.get("OMP_NUM_THREADS") == str(thread_count)

    custom_count = HardwareProfile.configure_omp_threads(threads=4)
    assert custom_count == 4
    assert os.environ.get("OMP_NUM_THREADS") == "4"


def test_trainer_resume_passes_completed_models_to_pipeline(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    run_id = "exp_old"
    log_path.write_text(
        '{"event":"model_completed","run_id":"exp_old","model":"catboost"}\n'
        '{"event":"model_completed","run_id":"exp_other","model":"lightgbm"}\n'
    )

    captured: dict[str, object] = {}

    def _fake_run_training(self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(results={})

    monkeypatch.setattr(
        "iter8ml.engine.pipelines.executor.PipelineExecutor.run_training",
        _fake_run_training,
    )

    config = ExperimentConfig(
        name="test_resume",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
    )
    trainer = Trainer(config=config, workspace=Workspace(root=tmp_path), resume_run_id=run_id)
    df = pl.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})
    trainer.run(df)

    assert captured["completed_models"] == {"catboost"}


def test_trainer_emits_ordered_events_with_run_ids(tmp_path):
    config = ExperimentConfig(
        name="event_contract",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
    )
    trainer = Trainer(config=config, workspace=Workspace(root=tmp_path))
    state = SimpleNamespace(results={"catboost": {"error": "boom"}})

    run_id = "exp_contract"
    trainer.tracker.current_run_id = run_id
    trainer.tracker.log_path = Path(tmp_path) / "experiments.jsonl"

    trainer.tracker.log_event(
        {
            "event": "experiment_started",
            "config": config.model_dump(mode="json"),
            "run_id": run_id,
        }
    )
    trainer._log_state_events(state, run_id)

    lines = [line for line in (tmp_path / "experiments.jsonl").read_text().splitlines() if line]
    assert '"event": "experiment_started"' in lines[0]
    assert '"event": "model_failed"' in lines[1]
    assert f'"run_id": "{run_id}"' in lines[0]
    assert f'"run_id": "{run_id}"' in lines[1]


def test_trainer_state_publish_failure_raises_typed_error(tmp_path):
    config = ExperimentConfig(
        name="state_failure",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
    )
    trainer = Trainer(config=config, workspace=Workspace(root=tmp_path))
    df = pl.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})

    class FailingStateAdapter:
        def publish(self) -> str:
            raise RuntimeError("observer down")

    trainer._state_adapter = FailingStateAdapter()

    def _fake_run_training(self, **kwargs):
        return SimpleNamespace(results={})

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "iter8ml.engine.pipelines.executor.PipelineExecutor.run_training",
            _fake_run_training,
        )
        with pytest.raises(TrainerStatePublishError) as exc_info:
            trainer.run(df)

    exc = exc_info.value
    assert exc.context["adapter"] == "FailingStateAdapter"
    assert exc.context["original_type"] == "RuntimeError"
    assert "observer down" in exc.context["original_message"]
    assert exc.context["run_id"].startswith("exp_")


def test_trainer_event_publish_failure_is_best_effort(tmp_path, caplog):
    config = ExperimentConfig(
        name="event_best_effort",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
    )
    trainer = Trainer(config=config, workspace=Workspace(root=tmp_path))
    df = pl.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})

    class FlakyEventAdapter:
        def publish(self, event):
            if event.get("event") == "model_completed":
                raise RuntimeError("sink unavailable")

    class NoopStateAdapter:
        def publish(self) -> str:
            return ""

    trainer._event_adapter = FlakyEventAdapter()
    trainer._state_adapter = NoopStateAdapter()

    def _fake_run_training(self, **kwargs):
        return SimpleNamespace(results={"catboost": {"model_name": "catboost"}})

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "iter8ml.engine.pipelines.executor.PipelineExecutor.run_training",
            _fake_run_training,
        )
        result = trainer.run(df)

    assert "catboost" in result
    assert "Trainer event publication failed" in caplog.text
    assert "adapter=FlakyEventAdapter" in caplog.text
