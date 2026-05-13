"""Smoke test: end-to-end pipeline with minimal data and models."""

import polars as pl
import pytest
from sklearn.datasets import make_classification

from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy, TaskType, TrackerType
from iter8ml.engine.trainer import Trainer
from iter8ml.workspace import Workspace


@pytest.mark.smoke
def test_trainer_runs_end_to_end(tmp_workspace):
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])}).with_columns(
        target=pl.Series(y)
    )

    config = ExperimentConfig(
        name="smoke_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        models=["catboost"],
        cv_folds=2,
        cv_strategy=CVStrategy.KFOLD,
        tracker=TrackerType.JSONL,
        run_quality_audit=False,
        max_workers=1,
    )

    ws = Workspace(root=tmp_workspace)
    trainer = Trainer(config=config, workspace=ws, run_leakage_audit=False)
    results = trainer.run(df)

    assert isinstance(results, dict)
    assert "catboost" in results
    assert "error" not in results["catboost"]
    assert "cv_scores" in results["catboost"]
    assert "roc_auc" in results["catboost"]["cv_scores"]

    log_file = ws.experiments_path
    assert log_file.exists()
    events = log_file.read_text().strip().split("\n")
    assert any("experiment_started" in e for e in events)
    assert any("model_completed" in e for e in events)

    registry_file = ws.registry_path
    assert registry_file.exists()
    registry = registry_file.read_text()
    assert "catboost" in registry or "smoke_test" in registry
