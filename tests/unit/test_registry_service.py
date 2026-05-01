"""Test RegistryService."""

import json

import pytest

from tabular_blueprint.services.registry_service import RegistryService


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temporary registry file."""
    return tmp_path / "registry.json"


def test_load_empty_registry(temp_registry):
    """Test loading non-existent registry returns empty dict."""
    service = RegistryService(temp_registry)
    assert service.load() == {}


def test_load_existing_registry(temp_registry):
    """Test loading existing registry."""
    temp_registry.write_text(json.dumps({"key1": {"model": "catboost"}}))
    service = RegistryService(temp_registry)
    assert service.load() == {"key1": {"model": "catboost"}}


def test_update_if_better_new_key(temp_registry):
    """Test updating registry with new key."""
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run1", 0.95, "/path/to/model")
    assert result is True
    registry = service.load()
    assert registry["key1"]["score"] == 0.95


def test_update_if_better_higher_score(temp_registry):
    """Test updating registry with higher score."""
    temp_registry.write_text(json.dumps({"key1": {"score": 0.90}}))
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run2", 0.95, "/path/to/model")
    assert result is True
    assert service.load()["key1"]["score"] == 0.95


def test_update_if_better_lower_score(temp_registry):
    """Test that lower score doesn't update registry."""
    temp_registry.write_text(json.dumps({"key1": {"score": 0.95}}))
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run2", 0.90, "/path/to/model")
    assert result is False
    assert service.load()["key1"]["score"] == 0.95


def test_update_if_better_lower_is_better_metric(temp_registry):
    """Test that smaller rmse replaces a worse champion."""
    temp_registry.write_text(json.dumps({"key1": {"score": 10.0, "metric_name": "rmse"}}))
    service = RegistryService(temp_registry)
    result = service.update_if_better(
        "key1",
        "catboost",
        "run2",
        2.0,
        "/path/to/model",
        metric_name="rmse",
    )
    assert result is True
    assert service.load()["key1"]["score"] == 2.0


def test_promote_run_missing_run(temp_registry, tmp_path):
    service = RegistryService(temp_registry)
    result = service.promote_run("missing", "key1", tmp_path / "experiments.jsonl")
    assert result.status == "not_found"
    assert "not found" in result.message


def test_promote_run_regression_uses_r2(temp_registry, tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    event = {
        "event": "model_completed",
        "run_id": "run_reg",
        "model": "CatBoost",
        "task": "regression",
        "cv_scores": {"rmse": 1.2, "r2": 0.81},
        "artifact_path": "/tmp/model",
    }
    log_path.write_text(json.dumps(event) + "\n")

    service = RegistryService(temp_registry)
    result = service.promote_run("run_reg", "regression:test", log_path)

    assert result.status == "promoted"
    assert service.load()["regression:test"]["score"] == 0.81


def test_promote_run_rejects_when_existing_champion_is_better(temp_registry, tmp_path):
    temp_registry.write_text(json.dumps({"key1": {"score": 0.95}}))
    log_path = tmp_path / "experiments.jsonl"
    event = {
        "event": "model_completed",
        "run_id": "run_low",
        "model": "CatBoost",
        "task": "classification",
        "cv_scores": {"roc_auc": 0.90},
        "artifact_path": "/tmp/model",
    }
    log_path.write_text(json.dumps(event) + "\n")

    service = RegistryService(temp_registry)
    result = service.promote_run("run_low", "key1", log_path)

    assert result.status == "rejected"
    assert service.load()["key1"]["score"] == 0.95


def test_promote_run_lower_is_better_metric_replaces_existing(temp_registry, tmp_path):
    registry_data = {"regression:test": {"score": 10.0, "metric_name": "rmse"}}
    temp_registry.write_text(json.dumps(registry_data))
    log_path = tmp_path / "experiments.jsonl"
    event = {
        "event": "model_completed",
        "run_id": "run_better_rmse",
        "model": "CatBoost",
        "task": "regression",
        "cv_scores": {"rmse": 2.0},
        "artifact_path": "/tmp/model",
    }
    log_path.write_text(json.dumps(event) + "\n")

    service = RegistryService(temp_registry)
    result = service.promote_run("run_better_rmse", "regression:test", log_path)

    assert result.status == "promoted"
    assert service.load()["regression:test"]["score"] == 2.0
    assert service.load()["regression:test"]["metric_name"] == "rmse"


def test_promote_run_uses_best_score_then_latest_timestamp(temp_registry, tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    events = [
        {
            "event": "model_completed",
            "run_id": "run_multi",
            "model": "ModelA",
            "cv_scores": {"roc_auc": 0.90},
            "artifact_path": "/tmp/model_a",
            "timestamp": "2026-04-01T00:00:00+00:00",
        },
        {
            "event": "model_completed",
            "run_id": "run_multi",
            "model": "ModelB",
            "cv_scores": {"roc_auc": 0.92},
            "artifact_path": "/tmp/model_b",
            "timestamp": "2026-04-02T00:00:00+00:00",
        },
        {
            "event": "model_completed",
            "run_id": "run_multi",
            "model": "ModelC",
            "cv_scores": {"roc_auc": 0.92},
            "artifact_path": "/tmp/model_c",
            "timestamp": "2026-04-03T00:00:00+00:00",
        },
    ]
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    service = RegistryService(temp_registry)
    result = service.promote_run("run_multi", "key1", log_path)

    assert result.status == "promoted"
    assert result.selected_model == "ModelC"
    assert result.selected_metric == "roc_auc"
    assert result.selected_score == 0.92
    assert service.load()["key1"]["artifact_path"] == "/tmp/model_c"
