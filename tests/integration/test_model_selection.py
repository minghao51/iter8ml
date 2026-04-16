"""Integration tests for TabPFN and model selection routing."""

import json

import polars as pl
import pytest
from sklearn.datasets import make_classification

from configs.experiment import ExperimentConfig
from core.engine.tracker import JSONLTracker
from core.engine.trainer import Trainer
from core.models.selector import ModelSelector


@pytest.fixture
def small_classification_data():
    """Small dataset suitable for TabPFN (< 10k rows)."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture
def tiny_classification_data():
    """Very small dataset for TabPFN edge case tests."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.mark.skip(reason="TabPFN requires TABPFN_TOKEN for authentication")
def test_tabpfn_runs_on_small_data(small_classification_data, tmp_path):
    """TabPFN should run and produce results on small datasets."""
    config = ExperimentConfig(
        name="tabpfn_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["tabpfn"],
        cv_folds=3,
        metrics=["roc_auc", "accuracy"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(small_classification_data)

    assert "tabpfn" in results
    assert "roc_auc" in results["tabpfn"]["cv_scores"]
    assert 0.0 < results["tabpfn"]["cv_scores"]["roc_auc"] <= 1.0


@pytest.mark.skip(reason="TabPFN requires TABPFN_TOKEN for authentication")
def test_tabpfn_data_size_guardrail(tiny_classification_data, tmp_path):
    """TabPFN should handle small data correctly."""
    config = ExperimentConfig(
        name="tabpfn_tiny_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["tabpfn"],
        cv_folds=3,
        metrics=["roc_auc"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(tiny_classification_data)

    assert "tabpfn" in results
    assert results["tabpfn"]["cv_scores"]["roc_auc"] > 0.5


def test_model_selector_routes_large_data_to_gbdt(large_classification_data):
    """ModelSelector should route large data to GBDT models, not TabPFN."""
    selector = ModelSelector()
    models = selector.select(
        n_rows=len(large_classification_data),
        task="classification",
        vram_gb=0.0,
    )

    assert "tabpfn" not in models
    assert any(m in models for m in ["catboost", "lightgbm", "xgboost"])


def test_model_selector_includes_tabpfn_for_small_data(small_classification_data):
    """ModelSelector should include TabPFN for small datasets."""
    selector = ModelSelector()
    models = selector.select(
        n_rows=len(small_classification_data),
        task="classification",
        vram_gb=0.0,
    )

    assert "tabpfn" in models


def test_auto_model_selection_produces_results(small_classification_data, tmp_path):
    """Auto model selection should produce results from appropriate models."""
    config = ExperimentConfig(
        name="auto_selection_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["catboost", "lightgbm"],
        cv_folds=3,
        metrics=["roc_auc"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(small_classification_data)

    assert len(results) >= 1
    for _model_name, model_results in results.items():
        assert "cv_scores" in model_results
        assert "roc_auc" in model_results["cv_scores"]


def test_jsonl_event_logged_for_model_completion(small_classification_data, tmp_path):
    """Verify model_completed events are logged to JSONL."""
    config = ExperimentConfig(
        name="jsonl_event_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["catboost"],
        cv_folds=3,
        metrics=["roc_auc"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    trainer.run(small_classification_data)

    events = []
    with open(tmp_path / "experiments.jsonl") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    completed_events = [e for e in events if e.get("event") == "model_completed"]
    assert len(completed_events) >= 1

    event = completed_events[0]
    assert "model" in event
    assert "cv_scores" in event
    assert "run_id" in event
