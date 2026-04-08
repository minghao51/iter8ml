"""Tests for DriftDetector."""

from decimal import Decimal

import polars as pl

from core.monitoring.drift import DriftDetector


def test_no_drift_same_distribution():
    ref_df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 20})
    new_df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 20})

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    assert report.n_columns_tested == 1
    assert report.n_drifted == 0
    assert report.drift_detected is False


def test_drift_detected_shifted_mean():
    ref_df = pl.DataFrame({"a": [0.0] * 100})
    new_df = pl.DataFrame({"a": [10.0] * 100})

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    assert report.drift_detected is True
    assert report.n_drifted == 1


def test_categorical_drift():
    ref_df = pl.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
    new_df = pl.DataFrame({"cat": ["A"] * 20 + ["B"] * 80})

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    assert report.drift_detected is True


def test_empty_new_df():
    ref_df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    new_df = pl.DataFrame({"a": []}).cast({"a": pl.Float64})

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    assert report.n_columns_tested == 1


def test_decimal_column_with_nulls():
    ref_df = pl.DataFrame({"a": [Decimal("1.1"), Decimal("2.2"), None]}).cast(
        {"a": pl.Decimal(10, 1)}
    )
    new_df = pl.DataFrame({"a": [Decimal("1.0"), Decimal("2.0"), None]}).cast(
        {"a": pl.Decimal(10, 1)}
    )

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    assert report.n_columns_tested == 1
    assert len(report.column_results) == 1
