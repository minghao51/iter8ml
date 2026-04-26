"""Tests for PSI drift detection module."""

import numpy as np
import polars as pl

from tabular_blueprint.monitoring.psi_drift import (
    PSIDriftDetector,
    classify_drift,
    compute_psi,
)


class TestComputePSI:
    def test_identical_distributions_low_psi(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.1

    def test_shifted_distributions_high_psi(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        live = np.random.normal(5, 1, 1000)
        psi = compute_psi(ref, live)
        assert psi > 0.25

    def test_empty_data_returns_zero(self):
        psi = compute_psi(np.array([]), np.array([1.0, 2.0]))
        assert psi == 0.0


class TestClassifyDrift:
    def test_none_level(self):
        assert classify_drift(0.05) == "none"

    def test_moderate_level(self):
        assert classify_drift(0.25) == "moderate"

    def test_severe_level(self):
        assert classify_drift(0.35) == "severe"


class TestPSIDriftDetector:
    def test_no_drift_identical_data(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        df = pl.DataFrame({"a": data, "b": data * 2})
        detector = PSIDriftDetector(df)
        report = detector.detect(df)
        assert report.drift_detected is False
        assert report.n_features_tested == 2

    def test_detects_drift_shifted_data(self):
        np.random.seed(42)
        ref = pl.DataFrame({"a": np.random.normal(0, 1, 500), "b": np.random.normal(0, 1, 500)})
        live = pl.DataFrame({"a": np.random.normal(3, 1, 500), "b": np.random.normal(0, 1, 500)})
        detector = PSIDriftDetector(ref)
        report = detector.detect(live)
        assert report.drift_detected is True
        assert report.n_severe >= 1

    def test_skips_non_numeric_columns(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        detector = PSIDriftDetector(df)
        report = detector.detect(df)
        assert report.n_features_tested == 1
