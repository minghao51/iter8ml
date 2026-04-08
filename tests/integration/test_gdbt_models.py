"""Integration tests for LightGBM and XGBoost models."""

import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression

from configs.experiment import ExperimentConfig
from core.engine.tracker import JSONLTracker
from core.engine.trainer import Trainer


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


def test_lightgbm_classification(classification_data, tmp_path):
    config = ExperimentConfig(
        name="lightgbm_cls_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["lightgbm"],
        cv_folds=3,
        metrics=["roc_auc", "f1_macro"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(classification_data)

    assert "lightgbm" in results
    assert "roc_auc" in results["lightgbm"]["cv_scores"]
    assert results["lightgbm"]["cv_scores"]["roc_auc"] > 0.5


def test_lightgbm_regression(regression_data, tmp_path):
    config = ExperimentConfig(
        name="lightgbm_reg_test",
        task="regression",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["lightgbm"],
        cv_folds=3,
        cv_strategy="kfold",
        metrics=["rmse", "r2"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(regression_data)

    assert "lightgbm" in results
    assert "rmse" in results["lightgbm"]["cv_scores"]
    assert "r2" in results["lightgbm"]["cv_scores"]


def test_xgboost_classification(classification_data, tmp_path):
    config = ExperimentConfig(
        name="xgboost_cls_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["xgboost"],
        cv_folds=3,
        metrics=["roc_auc", "f1_macro"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(classification_data)

    assert "xgboost" in results
    assert "roc_auc" in results["xgboost"]["cv_scores"]
    assert results["xgboost"]["cv_scores"]["roc_auc"] > 0.5


def test_xgboost_regression(regression_data, tmp_path):
    config = ExperimentConfig(
        name="xgboost_reg_test",
        task="regression",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["xgboost"],
        cv_folds=3,
        cv_strategy="kfold",
        metrics=["rmse", "r2"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(regression_data)

    assert "xgboost" in results
    assert "rmse" in results["xgboost"]["cv_scores"]
    assert "r2" in results["xgboost"]["cv_scores"]


def test_multi_model_run(classification_data, tmp_path):
    config = ExperimentConfig(
        name="multi_model_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        models=["catboost", "lightgbm", "xgboost"],
        cv_folds=3,
        metrics=["roc_auc"],
    )
    tracker = JSONLTracker(str(tmp_path / "experiments.jsonl"))
    trainer = Trainer(config, tracker=tracker)
    results = trainer.run(classification_data)

    assert "catboost" in results
    assert "lightgbm" in results
    assert "xgboost" in results
    for model in ["catboost", "lightgbm", "xgboost"]:
        assert "roc_auc" in results[model]["cv_scores"]
        assert results[model]["cv_scores"]["roc_auc"] > 0.5
