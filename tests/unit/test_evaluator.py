"""Tests for evaluator metric routing and probability handling."""

import numpy as np
import pytest

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.constants import TaskType
from tabular_blueprint.engine.evaluator import Evaluator


class ProbaDrivenModel:
    """Model where label predictions are weak but probabilities are informative."""

    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Intentionally bad thresholded predictions.
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        # Probability is strongly correlated with X[:, 0].
        proba_pos = np.clip(X[:, 0], 1e-6, 1 - 1e-6)
        return np.column_stack([1 - proba_pos, proba_pos])


class NoProbaModel:
    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return None


def test_roc_auc_uses_probabilities():
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=200)
    X = np.column_stack([0.05 + 0.9 * y + 0.02 * rng.randn(200), rng.randn(200)])
    X[:, 0] = np.clip(X[:, 0], 0.0, 1.0)

    config = ExperimentConfig(
        name="eval_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc", "accuracy"],
    )
    scores = Evaluator(config).evaluate(ProbaDrivenModel, X, y)
    assert scores["roc_auc"] > 0.9


def test_roc_auc_requires_predict_proba():
    X = np.random.RandomState(0).rand(40, 3)
    y = np.random.RandomState(1).randint(0, 2, size=40)

    config = ExperimentConfig(
        name="eval_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        cv_folds=2,
        metrics=["roc_auc"],
    )

    with pytest.raises(ValueError, match="Metric 'roc_auc' requires predict_proba"):
        Evaluator(config).evaluate(NoProbaModel, X, y)
