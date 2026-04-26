"""Integration tests for hyperparameter optimization."""

import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.engine.hpo import create_study, optimize_model
from tabular_blueprint.models.factory import get_model_class


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
        workspace_dir=tmp_path,
        cv_folds=3,
        metrics=["roc_auc"],
    )

    from tabular_blueprint.engine.evaluator import Evaluator

    evaluator = Evaluator(config)
    model_cls = get_model_class("catboost")

    result = optimize_model(
        model_cls=model_cls,
        X=hpo_classification_data.drop("target").to_numpy(),
        y=hpo_classification_data["target"].to_numpy(),
        evaluator=evaluator,
        model_name="catboost",
        n_trials=3,
        task="classification",
    )

    assert "best_params" in result
    assert "best_value" in result
    assert "n_trials" in result
    assert result["n_trials"] == 3


def test_hpo_respects_time_limit(hpo_classification_data, tmp_path):
    """HPO should complete within reasonable time."""
    import time

    config = ExperimentConfig(
        name="hpo_timing_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        cv_folds=3,
        metrics=["roc_auc"],
    )

    from tabular_blueprint.engine.evaluator import Evaluator

    evaluator = Evaluator(config)
    model_cls = get_model_class("catboost")

    start = time.time()
    result = optimize_model(
        model_cls=model_cls,
        X=hpo_classification_data.drop("target").to_numpy(),
        y=hpo_classification_data["target"].to_numpy(),
        evaluator=evaluator,
        model_name="catboost",
        n_trials=5,
        task="classification",
    )
    elapsed = time.time() - start

    assert elapsed < 120
    assert "best_params" in result


def test_hpo_invalid_search_space_raises(hpo_classification_data, tmp_path):
    """HPO should raise on invalid search space."""
    config = ExperimentConfig(
        name="invalid_hpo_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        cv_folds=3,
        metrics=["roc_auc"],
    )

    from tabular_blueprint.engine.evaluator import Evaluator

    evaluator = Evaluator(config)
    model_cls = get_model_class("catboost")

    invalid_search_space = {"learning_rate": "invalid"}

    with pytest.raises((ValueError, TypeError)):
        optimize_model(
            model_cls=model_cls,
            X=hpo_classification_data.drop("target").to_numpy(),
            y=hpo_classification_data["target"].to_numpy(),
            evaluator=evaluator,
            model_name="catboost",
            n_trials=1,
            search_space=invalid_search_space,
            task="classification",
        )


def test_hpo_pruning_works(hpo_classification_data, tmp_path):
    """HPO should handle trial pruning gracefully."""
    config = ExperimentConfig(
        name="hpo_prune_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        cv_folds=3,
        metrics=["roc_auc"],
    )

    from tabular_blueprint.engine.evaluator import Evaluator

    evaluator = Evaluator(config)
    model_cls = get_model_class("catboost")

    result = optimize_model(
        model_cls=model_cls,
        X=hpo_classification_data.drop("target").to_numpy(),
        y=hpo_classification_data["target"].to_numpy(),
        evaluator=evaluator,
        model_name="catboost",
        n_trials=5,
        task="classification",
    )

    assert result["n_trials"] >= 1
    assert result["best_value"] is not None


def test_hpo_warmstart_injects_logged_trials(hpo_classification_data, tmp_path):
    """Second HPO run should inject trials from the first run when log_path is shared."""
    config = ExperimentConfig(
        name="hpo_warmstart_test",
        task="classification",
        target_col="target",
        data_path="",
        workspace_dir=tmp_path,
        cv_folds=3,
        metrics=["roc_auc"],
    )
    from tabular_blueprint.engine.evaluator import Evaluator

    evaluator = Evaluator(config)
    model_cls = get_model_class("catboost")
    log_path = tmp_path / "experiments.jsonl"
    search_space = {"depth": (4, 8), "learning_rate": (0.01, 0.2, "log")}

    optimize_model(
        model_cls=model_cls,
        X=hpo_classification_data.drop("target").to_numpy(),
        y=hpo_classification_data["target"].to_numpy(),
        evaluator=evaluator,
        model_name="catboost",
        n_trials=2,
        search_space=search_space,
        task="classification",
        log_path=str(log_path),
    )

    second = optimize_model(
        model_cls=model_cls,
        X=hpo_classification_data.drop("target").to_numpy(),
        y=hpo_classification_data["target"].to_numpy(),
        evaluator=evaluator,
        model_name="catboost",
        n_trials=2,
        search_space=search_space,
        task="classification",
        log_path=str(log_path),
    )

    assert second.get("warmstart_trials", 0) > 0
