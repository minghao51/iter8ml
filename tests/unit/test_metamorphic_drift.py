"""Metamorphic tests for drift detection: relations replace ground-truth oracles."""

import numpy as np
import polars as pl
import pytest

from iter8ml.analysis.drift import DriftDetector
from iter8ml.analysis.psi import PSIDriftDetector, compute_psi

pytestmark = pytest.mark.metamorphic


class TestDriftSelfRelation:
    """Metamorphic: a dataset compared to itself must not detect drift."""

    def test_self_drift_no_detection_numeric(self):
        df = pl.DataFrame({"a": np.random.RandomState(42).randn(200)})
        report = DriftDetector(df).detect(df)
        assert report.drift_detected is False

    def test_self_drift_no_detection_categorical(self):
        df = pl.DataFrame({"cat": ["A", "B", "C"] * 50})
        report = DriftDetector(df).detect(df)
        assert report.drift_detected is False

    def test_self_drift_no_detection_mixed(self):
        rng = np.random.RandomState(42)
        df = pl.DataFrame(
            {
                "a": rng.randn(100),
                "b": ["x", "y"] * 50,
            }
        )
        report = DriftDetector(df).detect(df)
        assert report.drift_detected is False

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


class TestDriftPermutationInvariance:
    """Metamorphic: column order must not affect drift results."""

    def test_column_order_invariant(self):
        rng = np.random.RandomState(42)
        ref = pl.DataFrame(
            {
                "a": rng.randn(100),
                "b": rng.randn(100),
                "c": rng.randn(100),
            }
        )
        live = pl.DataFrame(
            {
                "a": rng.randn(100) + 0.3,
                "b": rng.randn(100),
                "c": rng.randn(100) + 0.3,
            }
        )

        original = DriftDetector(ref).detect(live)
        shuffled = DriftDetector(pl.DataFrame({k: ref[k] for k in reversed(ref.columns)})).detect(
            pl.DataFrame({k: live[k] for k in reversed(live.columns)})
        )

        assert original.n_drifted == shuffled.n_drifted
        assert original.drift_detected == shuffled.drift_detected


class TestPSIMetamorphic:
    """Metamorphic relations specific to PSI drift."""

    def test_non_numeric_columns_skipped(self):
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": ["x", "y", "z"],
            }
        )
        report = PSIDriftDetector(df).detect(df)
        assert report.n_features_tested == 1

    def test_self_psi_approximately_zero(self):
        rng = np.random.RandomState(42)
        data = rng.randn(1000)
        psi = compute_psi(data, data)
        assert psi < 0.05

    def test_severe_drift_classification(self):
        psi = compute_psi(np.array([0.0] * 500), np.array([10.0] * 500))
        from iter8ml.analysis.psi import classify_drift

        assert classify_drift(psi) == "severe"
