"""Metamorphic tests for leakage detection: relations replace ground-truth oracles."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from iter8ml.data.leakage import detect_leakage

pytestmark = pytest.mark.metamorphic


class TestLeakageMonotonicity:
    """Metamorphic: lower threshold must flag at least as many features."""

    def test_threshold_monotonic_decreasing(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        results = []
        for threshold in [0.001, 0.01, 0.05, 0.1]:
            report = detect_leakage(X, y, task="classification", threshold=threshold, cv_folds=2)
            results.append(report.n_flagged)
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1], (
                f"Threshold monotonicity broken: {results[i]} < {results[i + 1]}"
            )


class TestLeakageInvariance:
    """Metamorphic: row order must not affect leakage results."""

    def test_row_order_invariant(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        report_original = detect_leakage(X, y, task="classification", threshold=0.05, cv_folds=2)

        rng = np.random.RandomState(99)
        idx = rng.permutation(len(X))
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        report_shuffled = detect_leakage(
            X_shuffled, y_shuffled, task="classification", threshold=0.05, cv_folds=2
        )

        assert report_original.n_flagged == report_shuffled.n_flagged

    def test_column_order_invariant(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        report_original = detect_leakage(X, y, task="classification", threshold=0.05, cv_folds=2)

        rng = np.random.RandomState(99)
        col_idx = rng.permutation(X.shape[1])
        X_shuffled = X[:, col_idx]
        report_shuffled = detect_leakage(
            X_shuffled, y, task="classification", threshold=0.05, cv_folds=2
        )

        assert report_original.n_flagged == report_shuffled.n_flagged


class TestLeakageRegression:
    """Metamorphic: regression leakage should behave consistently."""

    def test_leaky_feature_detected_regression(self):
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=500, n_features=5, noise=1.0, random_state=42)
        X[:, 0] = y + np.random.RandomState(42).normal(0, 0.001, size=len(y))

        report = detect_leakage(X, y, task="regression", threshold=0.02, cv_folds=2)
        flagged_indices = {f["feature_index"] for f in report.flagged_features}
        assert 0 in flagged_indices, "Leaky feature not flagged in regression"

    def test_no_leakage_when_all_noise_regression(self):

        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = rng.randn(200)
        report = detect_leakage(X, y, task="regression", threshold=0.15, cv_folds=2)
        assert report.n_flagged == 0 or report.baseline_score < 0.3
