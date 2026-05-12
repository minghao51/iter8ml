"""Tests for domain classifier drift detection module."""

import numpy as np
import polars as pl

from iter8ml.analysis.domain_classifier import (
    DOMAIN_AUC_THRESHOLD,
    DomainClassifierDriftDetector,
)


class TestDomainClassifierDriftDetector:
    def test_no_drift_identical_data(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (200, 3))
        df = pl.DataFrame({f"f_{i}": data[:, i] for i in range(3)})
        detector = DomainClassifierDriftDetector(df)
        report = detector.detect(df)
        assert report.drift_detected is False
        assert report.auc_score < DOMAIN_AUC_THRESHOLD

    def test_detects_drift_shifted_data(self):
        np.random.seed(42)
        ref_data = np.random.normal(0, 1, (500, 3))
        live_data = np.random.normal(3, 1, (500, 3))
        ref = pl.DataFrame({f"f_{i}": ref_data[:, i] for i in range(3)})
        live = pl.DataFrame({f"f_{i}": live_data[:, i] for i in range(3)})
        detector = DomainClassifierDriftDetector(ref)
        report = detector.detect(live)
        assert report.drift_detected is True
        assert report.auc_score > DOMAIN_AUC_THRESHOLD

    def test_no_numeric_columns_returns_no_drift(self):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        detector = DomainClassifierDriftDetector(df)
        report = detector.detect(df)
        assert report.drift_detected is False
        assert report.auc_score == 0.5

    def test_threshold_configurable(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (200, 3))
        df = pl.DataFrame({f"f_{i}": data[:, i] for i in range(3)})
        detector = DomainClassifierDriftDetector(df, threshold=0.99)
        report = detector.detect(df)
        assert report.threshold == 0.99
