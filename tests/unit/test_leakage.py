"""Tests for leakage detection audit."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from iter8ml.data.leakage import LeakageReport, _effective_parallel_jobs, detect_leakage


@pytest.fixture
def clean_classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    return X, y


@pytest.fixture
def leaky_classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    X[:, 0] = y + np.random.normal(0, 0.01, size=len(y))
    return X, y


def test_no_leakage_detected(clean_classification_data):
    X, y = clean_classification_data
    report = detect_leakage(X, y, task="classification", threshold=0.15)
    assert isinstance(report, LeakageReport)
    assert report.n_features_tested == 10
    assert report.n_flagged == 0
    assert report.baseline_score > 0
    assert report.effective_n_jobs >= 1
    assert report.duration_seconds >= 0.0


def test_leaky_feature_flagged(leaky_classification_data):
    X, y = leaky_classification_data
    report = detect_leakage(X, y, task="classification", threshold=0.05)
    assert report.n_flagged > 0
    flagged_indices = [f["feature_index"] for f in report.flagged_features]
    assert 0 in flagged_indices


def test_regression_leakage():
    X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
    report = detect_leakage(X, y, task="regression")
    assert isinstance(report, LeakageReport)
    assert report.n_features_tested == 10


def test_threshold_controls_sensitivity(clean_classification_data):
    X, y = clean_classification_data
    low_threshold = detect_leakage(X, y, threshold=0.001)
    high_threshold = detect_leakage(X, y, threshold=0.5)
    assert low_threshold.n_flagged >= high_threshold.n_flagged


def test_default_parallel_jobs_is_one(clean_classification_data):
    X, _ = clean_classification_data
    jobs = _effective_parallel_jobs(
        requested_jobs=1,
        n_tasks=X.shape[1],
        n_samples=X.shape[0],
        n_features=X.shape[1],
    )
    assert jobs == 1


def test_parallel_jobs_is_clamped_to_at_least_one(clean_classification_data):
    X, _ = clean_classification_data
    jobs = _effective_parallel_jobs(
        requested_jobs=0,
        n_tasks=X.shape[1],
        n_samples=X.shape[0],
        n_features=X.shape[1],
    )
    assert jobs == 1


def test_effective_parallel_jobs_caps_for_large_matrices():
    jobs = _effective_parallel_jobs(
        requested_jobs=12,
        n_tasks=100,
        n_samples=2_000,
        n_features=2_000,
    )
    assert jobs == 2


def test_leakage_with_highly_predictive_feature():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=2, random_state=42)
    X[:, 0] = y.astype(float) * 100 + np.random.normal(0, 1, size=len(y))
    report = detect_leakage(X, y, task="classification", threshold=0.01)
    assert report.n_flagged >= 1
    assert any(f["feature_index"] == 0 for f in report.flagged_features)
