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
