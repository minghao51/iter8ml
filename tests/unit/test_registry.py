"""Test registry file locking, concurrent access, and promotion logic."""

import json
import threading
from unittest.mock import patch

import pytest

from iter8ml.exceptions import RegistryError
from iter8ml.services.registry import RegistryService
from iter8ml.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(root=tmp_path)
    workspace.init()
    return workspace


@pytest.fixture
def svc(ws):
    return RegistryService(workspace=ws)


def test_update_if_better_better_score(svc):
    svc.update_if_better("key1", "model_a", "run1", 0.80, "/path/a")
    assert svc.update_if_better("key1", "model_b", "run2", 0.90, "/path/b")
    assert svc.load()["key1"]["score"] == 0.90
    assert svc.load()["key1"]["model"] == "model_b"


def test_update_if_better_worse_score(svc):
    svc.update_if_better("key1", "model_a", "run1", 0.90, "/path/a")
    assert not svc.update_if_better("key1", "model_b", "run2", 0.80, "/path/b")
    assert svc.load()["key1"]["model"] == "model_a"


def test_update_if_better_equal_score(svc):
    svc.update_if_better("key1", "model_a", "run1", 0.90, "/path/a", metric_name="roc_auc")
    assert not svc.update_if_better(
        "key1", "model_b", "run2", 0.90, "/path/b", metric_name="roc_auc"
    )
    entry = svc.load()["key1"]
    assert entry["model"] == "model_a"


def test_promote_run_single_best(ws, svc, tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    event = {
        "event": "model_completed",
        "run_id": "run1",
        "model": "CatBoost",
        "cv_scores": {"roc_auc": 0.93},
        "artifact_path": "/tmp/model",
    }
    log_path.write_text(json.dumps(event) + "\n")
    result = svc.promote_run("run1", "key1", log_path)
    assert result.status == "promoted"
    assert result.selected_score == 0.93
    assert svc.load()["key1"]["model"] == "CatBoost"


def test_promote_run_multiple_models(ws, svc, tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    events = [
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelA",
            "cv_scores": {"roc_auc": 0.88},
            "artifact_path": "/tmp/a",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelB",
            "cv_scores": {"roc_auc": 0.95},
            "artifact_path": "/tmp/b",
            "timestamp": "2026-01-01T00:00:01+00:00",
        },
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelC",
            "cv_scores": {"roc_auc": 0.95},
            "artifact_path": "/tmp/c",
            "timestamp": "2026-01-01T00:00:02+00:00",
        },
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    result = svc.promote_run("run1", "key1", log_path)
    assert result.status == "promoted"
    assert result.selected_model == "ModelC"
    assert svc.load()["key1"]["artifact_path"] == "/tmp/c"


def test_update_if_better_cross_metric_candidate_retained(svc):
    """A candidate scored on a different metric must not displace the champion."""
    svc.update_if_better("key1", "model_a", "run1", 0.83, "/path/a", metric_name="roc_auc")
    assert not svc.update_if_better(
        "key1", "model_b", "run2", 0.99, "/path/b", metric_name="accuracy"
    )
    entry = svc.load()["key1"]
    assert entry["model"] == "model_a"
    assert entry["score"] == 0.83
    assert entry["metric_name"] == "roc_auc"


def test_update_if_better_same_metric_better_candidate_promotes(svc):
    svc.update_if_better("key1", "model_a", "run1", 0.83, "/path/a", metric_name="roc_auc")
    assert svc.update_if_better("key1", "model_b", "run2", 0.86, "/path/b", metric_name="roc_auc")
    entry = svc.load()["key1"]
    assert entry["model"] == "model_b"
    assert entry["score"] == 0.86


def test_update_if_better_legacy_incumbent_without_metric_retained(ws, svc):
    """Legacy entries with metric_name=None are compatible with nothing."""
    legacy = {
        "key1": {
            "model": "legacy_model",
            "run_id": "run0",
            "score": 0.83,
            "metric_name": None,
            "artifact_path": "/path/legacy",
            "registered_at": "2025-01-01T00:00:00+00:00",
        }
    }
    ws.registry_path.write_text(json.dumps(legacy))
    assert not svc.update_if_better(
        "key1", "model_b", "run2", 0.99, "/path/b", metric_name="accuracy"
    )
    assert svc.load()["key1"]["model"] == "legacy_model"


def test_promote_run_metric_mismatch_retains_champion(ws, svc, tmp_path):
    svc.update_if_better("key1", "model_a", "run1", 0.83, "/path/a", metric_name="roc_auc")
    log_path = tmp_path / "experiments.jsonl"
    event = {
        "event": "model_completed",
        "run_id": "run2",
        "model": "ModelB",
        "cv_scores": {"accuracy": 0.99},
        "artifact_path": "/tmp/b",
    }
    log_path.write_text(json.dumps(event) + "\n")
    result = svc.promote_run("run2", "key1", log_path)
    assert result.status == "metric_mismatch"
    assert "champion retained" in result.message
    assert svc.load()["key1"]["model"] == "model_a"


def test_promote_run_ignores_cross_metric_run_events(ws, svc, tmp_path):
    """Run-event selection must compare only events scored on the same metric.

    The accuracy event has a higher value than the roc_auc event, but they are
    different metrics — the roc_auc anchor event must win.
    """
    log_path = tmp_path / "experiments.jsonl"
    events = [
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelA",
            "cv_scores": {"roc_auc": 0.83},
            "artifact_path": "/tmp/a",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelB",
            "cv_scores": {"accuracy": 0.99},
            "artifact_path": "/tmp/b",
            "timestamp": "2026-01-01T00:00:01+00:00",
        },
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    result = svc.promote_run("run1", "key1", log_path)
    assert result.status == "promoted"
    assert result.selected_model == "ModelA"
    assert result.selected_metric == "roc_auc"
    assert result.selected_score == 0.83


def test_promote_run_anchored_on_first_event_with_real_metric(ws, svc, tmp_path):
    """A leading score-less event must not poison the anchor metric.

    The first event has empty cv_scores (resolves to the degenerate "score"
    sentinel); the anchor must come from the first event with a real metric,
    otherwise every real event would be skipped as a mismatch.
    """
    log_path = tmp_path / "experiments.jsonl"
    events = [
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelEmpty",
            "cv_scores": {},
            "artifact_path": "/tmp/empty",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelA",
            "cv_scores": {"roc_auc": 0.80},
            "artifact_path": "/tmp/a",
            "timestamp": "2026-01-01T00:00:01+00:00",
        },
        {
            "event": "model_completed",
            "run_id": "run1",
            "model": "ModelB",
            "cv_scores": {"roc_auc": 0.90},
            "artifact_path": "/tmp/b",
            "timestamp": "2026-01-01T00:00:02+00:00",
        },
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    result = svc.promote_run("run1", "key1", log_path)
    assert result.status == "promoted"
    assert result.selected_model == "ModelB"
    assert result.selected_metric == "roc_auc"
    assert result.selected_score == 0.90


def test_concurrent_access(ws, svc):
    errors = []
    results = []
    barrier = threading.Barrier(4)

    def writer(key, model, score):
        try:
            barrier.wait(timeout=5)
            ok = svc.update_if_better(key, model, f"run_{model}", score, f"/path/{model}")
            results.append(ok)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=("key1", f"m{i}", 0.80 + i * 0.05)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Concurrent errors: {errors}"
    registry = svc.load()
    assert "key1" in registry
    data = json.loads(ws.registry_path.read_text())
    assert "key1" in data


def test_get_returns_none_when_empty(svc):
    assert svc.get("nonexistent") is None


def test_get_all_returns_all_entries(svc):
    svc.update_if_better("k1", "m1", "r1", 0.9, "/a")
    svc.update_if_better("k2", "m2", "r2", 0.8, "/b")
    all_entries = svc.get_all()
    assert len(all_entries) == 2
    assert all_entries["k1"]["score"] == 0.9
    assert all_entries["k2"]["score"] == 0.8


def test_atomic_write_on_crash(ws, svc):
    svc.update_if_better("key1", "model_a", "run1", 0.90, "/path/a")
    original = ws.registry_path.read_text()

    with (
        patch("os.replace", side_effect=OSError("disk full")),
        pytest.raises(RegistryError, match="disk full"),
    ):
        svc.update_if_better("key1", "model_b", "run2", 0.95, "/path/b")

    assert ws.registry_path.read_text() == original
    assert svc.load()["key1"]["model"] == "model_a"


def test_file_locking_prevents_corruption(ws):
    svc = RegistryService(workspace=ws)
    errors = []
    barrier = threading.Barrier(8)

    def writer(thread_id):
        try:
            barrier.wait(timeout=5)
            svc.update_if_better(
                "shared",
                f"model_{thread_id}",
                f"run_{thread_id}",
                0.5 + thread_id * 0.01,
                f"/path/{thread_id}",
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Errors during parallel writes: {errors}"
    data = json.loads(ws.registry_path.read_text())
    assert "shared" in data
    assert "score" in data["shared"]
