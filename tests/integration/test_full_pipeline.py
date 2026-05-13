"""Integration test: full pipeline on synthetic data."""

import json
import tempfile
from pathlib import Path

import polars as pl
from sklearn.datasets import make_classification, make_regression

from iter8ml.config import ExperimentConfig
from iter8ml.engine.tracker import JSONLTracker
from iter8ml.engine.trainer import Trainer
from iter8ml.workspace import Workspace


def test_full_pipeline_catboost_classification():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    df = df.with_columns(target=pl.Series(y))

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Workspace(root=Path(tmpdir))
        config = ExperimentConfig(
            name="integration_test",
            task="classification",
            target_col="target",
            data_path="",
            cv_folds=3,
            metrics=["roc_auc", "f1_macro"],
        )

        tracker = JSONLTracker(str(ws.experiments_path))
        trainer = Trainer(config, workspace=ws, tracker=tracker)
        results = trainer.run(df)

        assert "catboost" in results
        assert "roc_auc" in results["catboost"]["cv_scores"]

        events = []
        with open(ws.experiments_path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        completed = [e for e in events if e.get("event") == "model_completed"]
        assert len(completed) >= 1

        leaderboard_path = ws.leaderboard_path
        assert leaderboard_path.exists()
        assert "# Experiment Leaderboard" in leaderboard_path.read_text()


def test_full_pipeline_catboost_regression():
    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    df = df.with_columns(target=pl.Series(y))

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Workspace(root=Path(tmpdir))
        config = ExperimentConfig(
            name="integration_regression_test",
            task="regression",
            target_col="target",
            data_path="",
            cv_folds=3,
            cv_strategy="kfold",
            metrics=["rmse", "r2"],
        )

        tracker = JSONLTracker(str(ws.experiments_path))
        trainer = Trainer(config, workspace=ws, tracker=tracker)
        results = trainer.run(df)

        assert "catboost" in results
        assert "rmse" in results["catboost"]["cv_scores"]
        assert "r2" in results["catboost"]["cv_scores"]

        events = []
        with open(ws.experiments_path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        completed = [e for e in events if e.get("event") == "model_completed"]
        assert len(completed) >= 1
        assert completed[-1]["task"] == "regression"
