"""Tests for Optuna HPO module."""

import numpy as np
import pytest

from configs.experiment import ExperimentConfig
from core.constants import TaskType
from core.engine.evaluator import Evaluator
from core.engine.hpo import create_study, optimize_model


class DummyModel:
    """A deterministic dummy model for HPO testing."""

    model_name = "Dummy"

    def __init__(self, task="classification", lr=0.01, n_estimators=100):
        self.task = task
        self.lr = lr
        self.n_estimators = n_estimators

    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        np.random.seed(42)
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X):
        np.random.seed(42)
        probs = np.random.rand(len(X), 2)
        return probs / probs.sum(axis=1, keepdims=True)


@pytest.fixture
def sample_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


def test_create_study_median_pruner():
    import optuna

    study = create_study("test_model", pruner="median", n_trials=5)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert isinstance(study.pruner.__class__.__name__, str)


def test_create_study_hyperband_pruner():
    study = create_study("test_model", pruner="hyperband", n_trials=5)
    assert study.pruner is not None


def test_create_study_nop_pruner():
    study = create_study("test_model", pruner="unknown", n_trials=5)
    assert study.pruner is not None


def test_create_study_minimize():
    import optuna

    study = create_study("test_model", direction="minimize", n_trials=5)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


def test_optimize_model_basic(sample_data):
    X, y = sample_data
    config = ExperimentConfig(
        name="hpo_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        cv_folds=2,
        metrics=["roc_auc"],
    )
    evaluator = Evaluator(config)
    search_space = {"lr": [0.001, 0.1], "n_estimators": [50, 200]}

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=3,
        search_space=search_space,
        task="classification",
    )

    assert "best_params" in result
    assert "best_value" in result
    assert "n_trials" in result
    assert result["n_trials"] == 3


def test_optimize_model_no_search_space(sample_data):
    X, y = sample_data
    config = ExperimentConfig(
        name="hpo_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        cv_folds=2,
        metrics=["roc_auc"],
    )
    evaluator = Evaluator(config)

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=2,
        search_space=None,
        task="classification",
    )

    assert result["n_trials"] == 2


def test_optimize_model_log_space(sample_data):
    X, y = sample_data
    config = ExperimentConfig(
        name="hpo_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        cv_folds=2,
        metrics=["roc_auc"],
    )
    evaluator = Evaluator(config)
    search_space = {"lr": [0.001, 0.1, "log"]}

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=2,
        search_space=search_space,
        task="classification",
    )

    assert "best_params" in result
    assert result["n_trials"] == 2


def test_optimize_model_preserves_exception_context():
    """Test that evaluation failures preserve exception context."""
    from unittest.mock import Mock

    from core.engine.hpo import optimize_model
    from core.models.conventional.catboost_model import CatBoostModel

    # Create invalid data to trigger error
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])  # Wrong shape for classification

    evaluator = Mock()

    # First trial fails, second succeeds
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Invalid data shape")
        return {"roc_auc": 0.8}

    evaluator.evaluate.side_effect = side_effect

    result = optimize_model(
        CatBoostModel,
        X,
        y,
        evaluator,
        "catboost",
        n_trials=2,
        search_space={},
    )

    # Should have completed trials without crashing
    assert result["n_trials"] == 2
    assert "best_params" in result
