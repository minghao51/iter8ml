"""Tests for StateObserver."""

import json

from tabular_blueprint.engine.state_observer import StateObserver


def test_generate_no_events(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    log_path.touch()

    observer = StateObserver(
        log_path=str(log_path),
        registry_path=str(tmp_path / "registry.json"),
        output_path=str(tmp_path / "current_state.md"),
    )
    content = observer.generate()
    assert "No experiments run yet" in content


def test_generate_with_events(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    event = {
        "event": "model_completed",
        "model": "CatBoost",
        "task": "classification",
        "dataset": "test_data",
        "n_rows": 1000,
        "n_features": 10,
        "cv_scores": {"roc_auc": 0.85, "f1_macro": 0.75},
        "duration_seconds": 5.2,
        "hardware": {"device": "cpu", "vram_used_gb": 0.0},
        "timestamp": "2026-04-04T00:00:00Z",
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(event) + "\n")

    observer = StateObserver(
        log_path=str(log_path),
        registry_path=str(tmp_path / "registry.json"),
        output_path=str(tmp_path / "current_state.md"),
    )
    content = observer.generate()

    assert "CatBoost" in content
    assert "roc_auc" in content
    assert (tmp_path / "current_state.md").exists()
    assert (tmp_path / "leaderboard.md").exists()


def test_generate_uses_most_recent_completed_event_for_current_state(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    first_event = {
        "event": "model_completed",
        "model": "BestModel",
        "task": "classification",
        "dataset": "older_data",
        "n_rows": 1000,
        "n_features": 10,
        "cv_scores": {"roc_auc": 0.95, "f1_macro": 0.85},
        "duration_seconds": 5.2,
        "hardware": {"device": "cpu", "vram_used_gb": 0.0},
        "timestamp": "2026-04-04T00:00:00Z",
    }
    latest_event = {
        "event": "model_completed",
        "model": "LatestModel",
        "task": "regression",
        "dataset": "new_data",
        "n_rows": 500,
        "n_features": 8,
        "cv_scores": {"r2": 0.40},
        "duration_seconds": 3.1,
        "hardware": {"device": "cpu", "vram_used_gb": 0.0},
        "timestamp": "2026-04-05T00:00:00Z",
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(first_event) + "\n")
        f.write(json.dumps(latest_event) + "\n")

    observer = StateObserver(
        log_path=str(log_path),
        registry_path=str(tmp_path / "registry.json"),
        output_path=str(tmp_path / "current_state.md"),
    )
    content = observer.generate()

    assert "**Task:** Regression" in content
    assert "**Dataset:** new_data" in content
    assert "| 1 | BestModel | unknown | roc_auc | 0.9500 | 5.2s |" in content
