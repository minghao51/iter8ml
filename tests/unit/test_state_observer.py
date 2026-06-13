"""Tests for StateObserver."""

import json
import os
from unittest.mock import patch

from iter8ml.engine.state_observer import StateObserver
from iter8ml.workspace import Workspace


def test_generate_no_events(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.experiments_path.touch()

    observer = StateObserver(workspace=ws)
    content = observer.generate()
    assert "No experiments run yet" in content


def test_generate_with_events(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
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

    observer = StateObserver(workspace=ws)
    content = observer.generate()

    assert "CatBoost" in content
    assert "roc_auc" in content
    assert ws.state_path.exists()
    assert ws.leaderboard_path.exists()


def test_generate_uses_most_recent_completed_event_for_current_state(tmp_path):
    ws = Workspace(root=tmp_path)
    log_path = ws.experiments_path
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

    observer = StateObserver(workspace=ws)
    content = observer.generate()

    assert "**Task:** Regression" in content
    assert "**Dataset:** new_data" in content
    assert "| 1 | BestModel | unknown | roc_auc | 0.9500 | 5.2s |" in content


def _write_events(log_path, events):
    with open(log_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _base_model_completed(**overrides):
    event = {
        "event": "model_completed",
        "model": "CatBoost",
        "task": "classification",
        "dataset": "test_data",
        "n_rows": 1000,
        "n_features": 10,
        "cv_scores": {"roc_auc": 0.85},
        "duration_seconds": 5.2,
        "hardware": {"device": "cpu", "vram_used_gb": 0.0},
        "timestamp": "2026-04-04T00:00:00Z",
    }
    event.update(overrides)
    return event


def test_state_with_leakage_audit(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {"event": "leakage_audit", "n_flagged": 2, "baseline_score": 0.80},
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "## Leakage Audit" in content
    assert "2 flagged" in content
    assert "0.8" in content


def test_state_with_target_transform_applied(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {
                "event": "target_transform",
                "method": "log1p",
                "applied": True,
                "original_skewness": 5.2,
                "transformed_skewness": 0.3,
            },
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "## Target Transform" in content
    assert "log1p" in content
    assert "5.2000" in content
    assert "0.3000" in content


def test_state_with_target_transform_not_applied(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {"event": "target_transform", "method": "none", "applied": False},
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "## Target Transform" not in content


def test_state_with_afe(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {
                "event": "afe_completed",
                "n_candidates_tested": 50,
                "n_candidates_kept": 3,
                "new_feature_names": ["feat_a", "feat_b", "feat_c"],
            },
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "## Automated Feature Engineering" in content
    assert "50" in content
    assert "3" in content
    assert "feat_a, feat_b, feat_c" in content


def test_state_with_shap_explainability(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {
                "event": "shap_explainability",
                "model": "CatBoost",
                "n_features": 10,
                "task": "classification",
                "top_features": [
                    {"name": "feat_1", "importance": 0.5},
                    {"name": "feat_2", "importance": 0.3},
                ],
            },
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "## SHAP Explainability" in content
    assert "CatBoost" in content
    assert "feat_1: 0.5000" in content
    assert "feat_2: 0.3000" in content


def test_state_with_shap_plot_paths(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {
                "event": "shap_explainability",
                "model": "CatBoost",
                "n_features": 5,
                "task": "classification",
                "top_features": [{"name": "x", "importance": 0.5}],
                "plot_paths": ["/tmp/beeswarm.png", "/tmp/dep_0.png"],
            },
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "beeswarm.png" in content
    assert "dep_0.png" in content


def test_state_with_drift_psi(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {
                "event": "drift_check",
                "method": "psi",
                "drift_detected": True,
                "n_moderate": 2,
                "n_severe": 1,
            },
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "## Drift Detection" in content
    assert "DRIFT DETECTED" in content
    assert "moderate=2" in content
    assert "severe=1" in content


def test_state_with_drift_domain_classifier(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
            {
                "event": "drift_check",
                "method": "domain_classifier",
                "drift_detected": False,
                "auc_score": 0.55,
                "threshold": 0.7,
            },
        ],
    )
    content = StateObserver(workspace=ws).generate()
    assert "Domain Classifier" in content
    assert "No drift" in content
    assert "AUC=0.5500" in content


def test_state_with_registry_champions(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(
        ws.experiments_path,
        [
            _base_model_completed(),
        ],
    )
    ws.registry_path.parent.mkdir(parents=True, exist_ok=True)
    ws.registry_path.write_text('{"best_model": {"model": "CatBoost", "score": 0.85}}')
    content = StateObserver(workspace=ws).generate()
    assert "## Registered Champions" in content
    assert "best_model" in content
    assert "CatBoost" in content


def test_state_pipeline_dag_rendered(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(ws.experiments_path, [_base_model_completed()])
    content = StateObserver(workspace=ws).generate()
    assert "## Data Pipeline" in content
    assert "mermaid" in content
    assert "digraph" in content


def test_state_dag_fallback(tmp_path):
    ws = Workspace(root=tmp_path)
    _write_events(ws.experiments_path, [_base_model_completed()])
    with patch("iter8ml.engine.pipelines.visualize_pipeline") as mock_viz:
        mock_viz.side_effect = ImportError()
        content = StateObserver(workspace=ws).generate()
        assert "graph TD" in content
        assert "Fill Nulls" in content


def test_default_llm_model_no_env(tmp_path):
    if "ITER8ML_LLM_MODEL" in os.environ:
        del os.environ["ITER8ML_LLM_MODEL"]
    if "TABBLUEPRINT_LLM_MODEL" in os.environ:
        del os.environ["TABBLUEPRINT_LLM_MODEL"]
    ws = Workspace(root=tmp_path)
    observer = StateObserver(workspace=ws)
    from iter8ml.config import DEFAULT_LLM_MODEL

    assert observer._default_llm_model() == DEFAULT_LLM_MODEL


def test_default_llm_model_with_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ITER8ML_LLM_MODEL", "gpt-4o")
    ws = Workspace(root=tmp_path)
    observer = StateObserver(workspace=ws)
    assert observer._default_llm_model() == "gpt-4o"
