"""Tests for Trainer class."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.constants import TaskType
from tabular_blueprint.engine.trainer import Trainer


def test_trainer_init(tmp_path):
    config = ExperimentConfig(
        name="test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        workspace_dir=tmp_path,
    )
    trainer = Trainer(config)
    assert trainer.config is config
    assert trainer.run_leakage_audit is True


def test_omp_threads_configurable(monkeypatch):
    import os

    from tabular_blueprint.config import HardwareProfile

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
        "tabular_blueprint.pipelines.executor.PipelineExecutor.run_training",
        _fake_run_training,
    )

    config = ExperimentConfig(
        name="test_resume",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        workspace_dir=tmp_path,
    )
    trainer = Trainer(config=config, resume_run_id=run_id)
    df = pl.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})
    trainer.run(df)

    assert captured["completed_models"] == {"catboost"}


def test_trainer_emits_ordered_events_with_run_ids(tmp_path):
    config = ExperimentConfig(
        name="event_contract",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        workspace_dir=tmp_path,
    )
    trainer = Trainer(config=config)
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
