"""Tests for Optuna HPO module."""

import numpy as np
import pytest

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
    evaluator = Evaluator(task="classification", cv_folds=2, metrics=["roc_auc"])
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
    evaluator = Evaluator(task="classification", cv_folds=2, metrics=["roc_auc"])

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
    evaluator = Evaluator(task="classification", cv_folds=2, metrics=["roc_auc"])
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
