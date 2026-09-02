"""Property-based tests for Evaluator metrics and invariants."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy, TaskType
from iter8ml.engine.evaluator import Evaluator

pytestmark = pytest.mark.property


def _make_clf_data(n_samples, n_features):
    from sklearn.datasets import make_classification

    n_info = max(2, min(n_features - 1, n_features))
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_info,
        n_redundant=n_features - n_info,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=42,
    )


class _DummyModel:
    def __init__(self, *, task="classification", **kwargs):
        self.task = task
        self.model_name = "DummyModel"

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, 2))
        proba[:, 1] = 0.6
        proba[:, 0] = 0.4
        return proba


class TestPropertyEvaluator:
    """Property: evaluator metrics are finite and within expected ranges."""

    @settings(max_examples=20, deadline=None)
    @given(
        n_samples=st.integers(50, 200),
        n_features=st.integers(2, 6),
        cv_folds=st.integers(2, 5),
    )
    def test_classification_metrics_finite(self, n_samples, n_features, cv_folds):
        X, y = _make_clf_data(n_samples, n_features)
        config = ExperimentConfig(
            name="test",
            task=TaskType.CLASSIFICATION,
            target_col="target",
            data_path="data.csv",
            cv_folds=cv_folds,
            cv_strategy=CVStrategy.KFOLD,
            metrics=["roc_auc", "accuracy", "f1_macro"],
        )
        evaluator = Evaluator(config)
        results = evaluator.evaluate(_DummyModel, X, y)
        for metric, value in results.items():
            assert np.isfinite(value), f"{metric} is not finite: {value}"
            assert 0.0 <= value <= 1.0 or metric == "log_loss"

    @settings(max_examples=10)
    @given(
        n_samples=st.integers(50, 150),
        n_features=st.integers(2, 5),
    )
    def test_regression_metrics_finite(self, n_samples, n_features):
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
        config = ExperimentConfig(
            name="test",
            task=TaskType.REGRESSION,
            target_col="target",
            data_path="data.csv",
            cv_folds=3,
            cv_strategy=CVStrategy.KFOLD,
            metrics=["rmse", "mae", "r2"],
        )
        evaluator = Evaluator(config)
        results = evaluator.evaluate(_DummyModel, X, y)
        for metric, value in results.items():
            assert np.isfinite(value), f"{metric} is not finite: {value}"


class TestPropertyEvaluatorLift:
    """Property: compute_lift is zero for identical scores."""

    @settings(max_examples=50)
    @given(
        model_val=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    def test_lift_zero_for_identical(self, model_val):
        baseline_val = model_val
        assume(model_val != 0.0)  # 0 baseline → lift undefined (None), covered separately
        lift = Evaluator.compute_lift({"roc_auc": model_val}, {"roc_auc": baseline_val}, "roc_auc")
        assert lift == 0.0

    @settings(max_examples=50)
    @given(
        model_val=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    def test_lift_none_for_zero_baseline(self, model_val):
        """Baseline exactly 0 with the metric present: lift undefined — None."""
        lift = Evaluator.compute_lift({"roc_auc": model_val}, {"roc_auc": 0.0}, "roc_auc")
        assert lift is None
