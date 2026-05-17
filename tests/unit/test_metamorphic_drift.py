"""Metamorphic tests for drift detection: relations replace ground-truth oracles."""

import numpy as np
import polars as pl
import pytest

from iter8ml.analysis.drift import DriftDetector
from iter8ml.analysis.psi import PSIDriftDetector, compute_psi

pytestmark = pytest.mark.metamorphic


class TestDriftSelfRelation:
    """Metamorphic: a dataset compared to itself must not detect drift.

    Note: KS tests have a 5% false-positive rate at alpha=0.05.
    We run multiple seeds and check that the majority show no drift.
    """

    def test_self_drift_no_detection_numeric(self):
        results = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            df = pl.DataFrame({"a": rng.randn(200)})
            report = DriftDetector(df).detect(df)
            results.append(report.drift_detected)
        n_drifted = sum(results)
        assert n_drifted <= 2, f"Self-drift flagged in {n_drifted}/10 runs (false positive)"

    def test_self_drift_no_detection_categorical(self):
        df = pl.DataFrame({"cat": ["A", "B", "C"] * 50})
        report = DriftDetector(df).detect(df)
        assert report.drift_detected is False

    def test_self_drift_no_detection_mixed(self):
        results = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            df = pl.DataFrame({"a": rng.randn(100), "b": ["x", "y"] * 50})
            report = DriftDetector(df).detect(df)
            results.append(report.drift_detected)
        n_drifted = sum(results)
        assert n_drifted <= 2, f"Self-drift flagged in {n_drifted}/10 runs (false positive)"

    def test_self_psi_no_drift(self):
        rng = np.random.RandomState(42)
        df = pl.DataFrame({"a": rng.randn(200), "b": rng.randn(200)})
        report = PSIDriftDetector(df).detect(df)
        assert report.drift_detected is False


class TestDriftMonotonicity:
    """Metamorphic: larger distribution shifts produce at least as much drift."""

    def test_more_shift_more_drift(self):
        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(500)})

        small_shift = pl.DataFrame({"a": rng.randn(500) + 0.1})
        large_shift = pl.DataFrame({"a": rng.randn(500) + 1.0})

        small_report = DriftDetector(ref).detect(small_shift)
        large_report = DriftDetector(ref).detect(large_shift)

        assert large_report.n_drifted >= small_report.n_drifted

    def test_more_shift_higher_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.randn(500)

        small = rng.randn(500) + 0.2
        large = rng.randn(500) + 2.0

        assert compute_psi(ref, large) >= compute_psi(ref, small)

    def test_alpha_monotonic(self):
        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(200)})
        live = pl.DataFrame({"a": rng.randn(200) + 0.5})

        strict = DriftDetector(ref, alpha=0.01).detect(live)
        loose = DriftDetector(ref, alpha=0.10).detect(live)

        assert strict.drift_detected <= loose.drift_detected

    def test_psi_drift_subset_of_features(self):
        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(200), "b": rng.randn(200), "c": rng.randn(200)})
        live = pl.DataFrame(
            {"a": rng.randn(200) + 0.5, "b": rng.randn(200), "c": rng.randn(200) + 1.0}
        )
        report = PSIDriftDetector(ref).detect(live)
        psi_features = {f.feature for f in report.feature_psi if f.drift_level != "none"}
        assert psi_features.issubset({"a", "b", "c"})


class TestDriftPermutationInvariance:
    """Metamorphic: column order must not affect drift results."""

    def test_column_order_invariant(self):
        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(100), "b": rng.randn(100), "c": rng.randn(100)})
        live = pl.DataFrame(
            {"a": rng.randn(100) + 0.3, "b": rng.randn(100), "c": rng.randn(100) + 0.3}
        )

        original = DriftDetector(ref).detect(live)
        shuffled = DriftDetector(pl.DataFrame({k: ref[k] for k in reversed(ref.columns)})).detect(
            pl.DataFrame({k: live[k] for k in reversed(live.columns)})
        )

        assert original.n_drifted == shuffled.n_drifted
        assert original.drift_detected == shuffled.drift_detected

    def test_row_shuffle_invariant(self):
        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(100)})
        live = pl.DataFrame({"a": rng.randn(100) + 0.3})

        original = DriftDetector(ref).detect(live)

        ref_shuffled = pl.DataFrame(
            {"a": np.random.RandomState(99).choice(ref["a"].to_numpy(), len(ref), replace=False)}
        )
        shuffled_report = DriftDetector(ref_shuffled).detect(live)

        assert original.n_drifted == shuffled_report.n_drifted

    def test_psi_column_order_invariant(self):
        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(100), "b": rng.randn(100)})
        live = pl.DataFrame({"a": rng.randn(100) + 0.3, "b": rng.randn(100)})

        original = PSIDriftDetector(ref).detect(live)
        shuffled = PSIDriftDetector(
            pl.DataFrame({k: ref[k] for k in reversed(ref.columns)})
        ).detect(pl.DataFrame({k: live[k] for k in reversed(live.columns)}))

        assert original.n_features_tested == shuffled.n_features_tested
        assert original.drift_detected == shuffled.drift_detected


class TestPSIMetamorphic:
    """Metamorphic relations specific to PSI drift."""

    def test_non_numeric_columns_skipped(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        report = PSIDriftDetector(df).detect(df)
        assert report.n_features_tested == 1

    def test_self_psi_approximately_zero(self):
        rng = np.random.RandomState(42)
        data = rng.randn(1000)
        psi = compute_psi(data, data)
        assert psi < 1e-10

    def test_severe_drift_classification(self):
        psi = compute_psi(np.array([0.0] * 500), np.array([10.0] * 500))
        from iter8ml.analysis.psi import classify_drift

        assert classify_drift(psi) == "severe"
