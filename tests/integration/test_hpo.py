"""Integration tests for hyperparameter optimization."""

import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression

from iter8ml.config import ExperimentConfig
from iter8ml.engine.hpo import create_study, optimize_model
from iter8ml.engine.models.factory import get_model_class


@pytest.fixture
def hpo_classification_data():
    X, y = make_classification(n_samples=300, n_features=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture
def hpo_regression_data():
    X, y = make_regression(n_samples=300, n_features=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


def test_create_study_maximize():
    """Optuna study should be created with maximize direction."""
    import optuna

    study = create_study("test_model", direction="maximize", n_trials=10)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE


def test_create_study_minimize():
    """Optuna study should be created with minimize direction."""
    import optuna

    study = create_study("test_model", direction="minimize", n_trials=10)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


def test_create_study_with_pruner():
    """Optuna study should support pruner configuration."""
    study = create_study("test_model", pruner="median")
    assert study.pruner is not None


def test_hpo_returns_best_params(hpo_classification_data, tmp_path):
    """HPO should return optimized parameters."""
    config = ExperimentConfig(
        name="hpo_test",
        task="classification",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc"],
    )

    from iter8ml.engine.evaluator import Evaluator

    evaluator = Evaluator(config)

    X = hpo_classification_data.drop("target").to_numpy()
    y = hpo_classification_data["target"].to_numpy()

    model_cls = get_model_class("catboost")
    result = optimize_model(
        model_cls,
        X,
        y,
        evaluator,
        "catboost",
        n_trials=3,
        search_space=None,
        task="classification",
        log_path=str(tmp_path / "experiments.jsonl"),
    )

    assert "best_params" in result
    assert "best_value" in result
    assert "n_trials" in result
    assert result["n_trials"] > 0


def test_hpo_regression_returns_params(hpo_regression_data, tmp_path):
    """HPO should work for regression tasks."""
    config = ExperimentConfig(
        name="hpo_reg_test",
        task="regression",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["rmse"],
    )

    from iter8ml.engine.evaluator import Evaluator

    evaluator = Evaluator(config)

    X = hpo_regression_data.drop("target").to_numpy()
    y = hpo_regression_data["target"].to_numpy()

    model_cls = get_model_class("catboost")
    result = optimize_model(
        model_cls,
        X,
        y,
        evaluator,
        "catboost",
        n_trials=3,
        search_space=None,
        task="regression",
        log_path=str(tmp_path / "experiments.jsonl"),
    )

    assert "best_params" in result
    assert result["n_trials"] > 0


def test_hpo_with_warmstart(hpo_classification_data, tmp_path):
    """HPO warmstart should read historical runs from JSONL."""
    import json

    log_path = tmp_path / "experiments.jsonl"
    historical = [
        {
            "event": "model_completed",
            "run_id": "past_run",
            "model": "catboost",
            "cv_scores": {"roc_auc": 0.85, "f1_macro": 0.72},
            "params": {"depth": 6, "learning_rate": 0.1, "iterations": 100},
            "task": "classification",
        }
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in historical) + "\n")

    config = ExperimentConfig(
        name="hpo_warm_test",
        task="classification",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc"],
    )

    from iter8ml.engine.evaluator import Evaluator

    evaluator = Evaluator(config)
    X = hpo_classification_data.drop("target").to_numpy()
    y = hpo_classification_data["target"].to_numpy()

    model_cls = get_model_class("catboost")
    result = optimize_model(
        model_cls,
        X,
        y,
        evaluator,
        "catboost",
        n_trials=3,
        search_space=None,
        task="classification",
        log_path=str(log_path),
    )

    warmstart_summary = result.get("warmstart_summary")
    assert warmstart_summary is not None
    assert warmstart_summary["n_runs_scanned"] >= 1
    assert result["n_trials"] > 0


def test_hpo_lightgbm_returns_params(hpo_classification_data, tmp_path):
    """HPO should work for LightGBM."""
    config = ExperimentConfig(
        name="hpo_lgb_test",
        task="classification",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc"],
    )

    from iter8ml.engine.evaluator import Evaluator

    evaluator = Evaluator(config)
    X = hpo_classification_data.drop("target").to_numpy()
    y = hpo_classification_data["target"].to_numpy()

    model_cls = get_model_class("lightgbm")
    result = optimize_model(
        model_cls,
        X,
        y,
        evaluator,
        "lightgbm",
        n_trials=3,
        search_space=None,
        task="classification",
        log_path=str(tmp_path / "experiments.jsonl"),
    )

    assert "best_params" in result
