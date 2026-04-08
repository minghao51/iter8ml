"""Tests for Trainer class."""

from unittest.mock import Mock, patch

import polars as pl
import pytest

from configs.experiment import ExperimentConfig
from core.constants import TaskType
from core.engine.trainer import Trainer


def test_trainer_uses_registry_service(tmp_path, monkeypatch):
    """Verify trainer uses RegistryService for updates."""
    from unittest.mock import MagicMock

    # Track calls to RegistryService
    registry_calls = []

    original_registry = None

    class MockRegistryService:
        def __init__(self, registry_path):
            self.registry_path = registry_path

        def update_if_better(self, key, model_name, run_id, score, artifact_path):
            registry_calls.append({
                "key": key,
                "model_name": model_name,
                "run_id": run_id,
                "score": score,
                "artifact_path": artifact_path,
            })
            return True

    config = ExperimentConfig(
        name="test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        workspace_dir=tmp_path,
    )

    # Monkey patch before creating trainer
    import core.engine.trainer
    original_registry = core.engine.trainer.RegistryService
    monkeypatch.setattr(core.engine.trainer, "RegistryService", MockRegistryService)

    trainer = Trainer(config)

    # Test the method directly
    result = trainer._update_champion_if_better(
        key="test:classification",
        model_name="catboost",
        run_id="test_run",
        score=0.95,
        artifact_path="/tmp/test.pkl"
    )

    # Verify the method was called and returns True
    assert result is True

    # Verify RegistryService.update_if_better was called with correct args
    assert len(registry_calls) == 1
    call = registry_calls[0]
    assert call["key"] == "test:classification"
    assert call["model_name"] == "catboost"
    assert call["run_id"] == "test_run"
    assert call["score"] == 0.95
    assert call["artifact_path"] == "/tmp/test.pkl"

    # Restore original
    monkeypatch.setattr(core.engine.trainer, "RegistryService", original_registry)


def test_omp_threads_configurable(monkeypatch):
    """Test that OMP threads can be configured via HardwareProfile."""
    from configs.hardware import HardwareProfile
    import os

    # Test default
    thread_count = HardwareProfile.configure_omp_threads()
    assert os.environ.get("OMP_NUM_THREADS") == str(thread_count)

    # Test custom value
    custom_count = HardwareProfile.configure_omp_threads(threads=4)
    assert custom_count == 4
    assert os.environ.get("OMP_NUM_THREADS") == "4"
