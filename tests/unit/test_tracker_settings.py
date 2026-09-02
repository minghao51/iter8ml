"""Tests for tracker_settings pass-through and per-backend validation."""

import pytest

from iter8ml.config import ExperimentConfig, TrackerType
from iter8ml.engine.tracker import JSONLTracker
from iter8ml.engine.trainer import _build_tracker
from iter8ml.workspace import Workspace


def make_config(tracker: TrackerType, settings: dict | None) -> ExperimentConfig:
    return ExperimentConfig(
        name="t",
        task="classification",
        target_col="target",
        data_path="unused.csv",
        tracker=tracker,
        tracker_settings=settings,
    )


class TestJsonlSettings:
    def test_settings_pass_through(self, tmp_path):
        ws = Workspace(root=tmp_path)
        cfg = make_config(TrackerType.JSONL, {"max_file_size_mb": 1, "backup_count": 2})

        tracker = _build_tracker(cfg, ws)

        assert isinstance(tracker, JSONLTracker)
        assert tracker.max_file_size_bytes == 1024 * 1024
        assert tracker.backup_count == 2

    def test_default_settings_build_plain_tracker(self, tmp_path):
        ws = Workspace(root=tmp_path)
        cfg = make_config(TrackerType.JSONL, None)

        tracker = _build_tracker(cfg, ws)

        assert isinstance(tracker, JSONLTracker)
        assert tracker.backup_count == 5  # ctor default

    def test_unknown_key_fails_loud(self, tmp_path):
        ws = Workspace(root=tmp_path)
        # log_path is deliberately not allowlisted: it is workspace-managed.
        cfg = make_config(TrackerType.JSONL, {"log_path": "/tmp/evil.jsonl"})

        with pytest.raises(ValueError, match=r"backend 'jsonl'.*'log_path'"):
            _build_tracker(cfg, ws)


class TestWandbSettings:
    def test_settings_reach_ctor(self, tmp_path, monkeypatch):
        captured: dict = {}

        class FakeWandbTracker:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("iter8ml.engine.tracker.WandbTracker", FakeWandbTracker)
        cfg = make_config(TrackerType.WANDB, {"project": "my-proj", "mode": "offline"})

        tracker = _build_tracker(cfg, Workspace(root=tmp_path))

        assert isinstance(tracker, FakeWandbTracker)
        assert captured == {"project": "my-proj", "mode": "offline"}

    def test_unknown_key_fails_loud(self, tmp_path):
        cfg = make_config(TrackerType.WANDB, {"bogus": 1})

        with pytest.raises(ValueError, match="backend 'wandb'"):
            _build_tracker(cfg, Workspace(root=tmp_path))


class TestMlflowSettings:
    def test_settings_reach_ctor(self, tmp_path, monkeypatch):
        captured: dict = {}

        class FakeMlflowTracker:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("iter8ml.engine.tracker.MLflowTracker", FakeMlflowTracker)
        cfg = make_config(TrackerType.MLFLOW, {"experiment_name": "exp-x"})

        _build_tracker(cfg, Workspace(root=tmp_path))

        assert captured == {"experiment_name": "exp-x"}

    def test_unknown_key_fails_loud_without_extra(self, tmp_path):
        """Validation fires before any mlflow import — testable without the extra."""
        cfg = make_config(TrackerType.MLFLOW, {"nope": True})

        with pytest.raises(ValueError, match="backend 'mlflow'"):
            _build_tracker(cfg, Workspace(root=tmp_path))
