"""Integration-level guardrail checks for large tabular inputs."""

import time

import pytest
from sklearn.datasets import make_classification

from iter8ml.data.features import discover_interactions
from iter8ml.data.leakage import detect_leakage

pytestmark = [pytest.mark.integration]


def test_discover_interactions_respects_candidate_cap_on_large_input():
    X, y = make_classification(
        n_samples=1200,
        n_features=80,
        n_informative=20,
        random_state=42,
    )
    top_k = list(range(20))
    start = time.perf_counter()
    _, result = discover_interactions(
        X,
        y,
        top_k_indices=top_k,
        task="classification",
        n_jobs=1,
        max_candidate_pairs=25,
    )
    assert result.n_candidates_tested <= 25
    assert time.perf_counter() - start < 10.0


def test_detect_leakage_handles_large_input_with_single_worker():
    X, y = make_classification(
        n_samples=1000,
        n_features=60,
        n_informative=15,
        random_state=42,
    )
    start = time.perf_counter()
    report = detect_leakage(X, y, task="classification", cv_folds=2, n_jobs=1)
    assert report.n_features_tested == X.shape[1]
    assert report.n_flagged >= 0
    assert time.perf_counter() - start < 15.0
