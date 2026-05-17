"""Property-based and metamorphic tests for leakage detection."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.datasets import make_classification

from iter8ml.data.leakage import LeakageReport, _effective_parallel_jobs, detect_leakage

pytestmark = pytest.mark.property


class TestPropertyLeakage:
    """Property: score_drop non-negative, n_flagged bounded."""

    @settings(max_examples=50)
    @given(
        n_samples=st.integers(80, 200),
        n_features=st.integers(4, 8),
        threshold=st.floats(0.0, 0.5, allow_nan=False, allow_infinity=False),
    )
    def test_score_drop_non_negative(self, n_samples, n_features, threshold):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        report = detect_leakage(X, y, task="classification", threshold=threshold, cv_folds=2)
        assert isinstance(report, LeakageReport)
        assert 0.0 <= report.baseline_score <= 1.0
        for f in report.flagged_features:
            assert f["score_drop"] >= 0.0

    @settings(max_examples=50)
    @given(
        n_samples=st.integers(80, 150),
        n_features=st.integers(4, 6),
    )
    def test_n_flagged_bounded(self, n_samples, n_features):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        report = detect_leakage(X, y, threshold=0.0, cv_folds=2)
        assert report.n_flagged <= report.n_features_tested
        assert report.n_flagged >= 0

    @settings(max_examples=30)
    @given(
        n_samples=st.integers(80, 150),
        n_features=st.integers(4, 6),
    )
    def test_threshold_monotonic(self, n_samples, n_features):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        low = detect_leakage(X, y, threshold=0.01, cv_folds=2)
        high = detect_leakage(X, y, threshold=0.5, cv_folds=2)
        assert low.n_flagged >= high.n_flagged


class TestPropertyParallelJobs:
    """Property: _effective_parallel_jobs is bounded and monotonic."""

    @settings(max_examples=50)
    @given(
        requested_jobs=st.integers(0, 16),
        n_tasks=st.integers(1, 50),
        n_samples=st.integers(10, 10000),
        n_features=st.integers(1, 100),
    )
    def test_parallel_jobs_is_bounded(self, requested_jobs, n_tasks, n_samples, n_features):
        result = _effective_parallel_jobs(
            requested_jobs=requested_jobs,
            n_tasks=n_tasks,
            n_samples=n_samples,
            n_features=n_features,
        )
        assert result >= 1
        if requested_jobs >= 1:
            assert result <= requested_jobs or requested_jobs == 0
        assert result <= n_tasks

    @settings(max_examples=50)
    @given(
        requested_jobs=st.integers(1, 8),
        n_tasks=st.integers(2, 20),
        n_samples=st.integers(10, 500),
        n_features=st.integers(1, 50),
    )
    def test_parallel_jobs_monotonic_in_requested(
        self, requested_jobs, n_tasks, n_samples, n_features
    ):
        low = _effective_parallel_jobs(
            requested_jobs=requested_jobs,
            n_tasks=n_tasks,
            n_samples=n_samples,
            n_features=n_features,
        )
        high = _effective_parallel_jobs(
            requested_jobs=requested_jobs + 4,
            n_tasks=n_tasks,
            n_samples=n_samples,
            n_features=n_features,
        )
        assert high >= low


class TestMetamorphicLeakage:
    """Metamorphic: a leaky feature must consistently be flagged."""

    def test_leaky_feature_detected_across_seeds(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X[:, 0] = y + np.random.RandomState(42).normal(0, 0.01, size=len(y))

        reports = []
        for _seed in range(3):
            X_copy = X.copy()
            y_copy = y.copy()
            report = detect_leakage(
                X_copy, y_copy, task="classification", threshold=0.05, cv_folds=2
            )
            reports.append(report)

        for report in reports:
            flagged_indices = {f["feature_index"] for f in report.flagged_features}
            assert 0 in flagged_indices, "Leaky feature not flagged for seed"

    def test_no_leakage_when_all_noise(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = rng.randint(0, 2, 200)
        report = detect_leakage(X, y, task="classification", threshold=0.15, cv_folds=2)
        assert report.n_flagged == 0
