"""Tests for cleanlab-based data quality audit."""

import importlib.util

import numpy as np
import polars as pl
import pytest

from iter8ml.data.quality import audit_data_quality, clean_noise

CLEANLAB_AVAILABLE = importlib.util.find_spec("cleanlab") is not None


def _make_classification_df(n_samples=200, n_features=5, noise_rate=0.0):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42,
    )
    cols = {f"feat_{i}": X[:, i] for i in range(n_features)}
    cols["target"] = y
    return pl.DataFrame(cols)


@pytest.mark.skipif(not CLEANLAB_AVAILABLE, reason="cleanlab not installed")
def test_audit_data_quality_enabled():
    df = _make_classification_df()
    report = audit_data_quality(df, "target", enabled=True)
    assert report["enabled"] is True
    assert report["n_rows"] == 200
    assert "n_issues" in report
    assert "noise_rate" in report
    assert "mean_quality_score" in report


def test_audit_data_quality_disabled():
    df = _make_classification_df()
    report = audit_data_quality(df, "target", enabled=False)
    assert report["enabled"] is False
    assert "skipped" in report["message"].lower()


@pytest.mark.skipif(not CLEANLAB_AVAILABLE, reason="cleanlab not installed")
def test_audit_data_quality_single_class():
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "target": [0, 0, 0],
        }
    )
    report = audit_data_quality(df, "target", enabled=True)
    assert report["enabled"] is False
    assert "2 classes" in report["message"]


@pytest.mark.skipif(not CLEANLAB_AVAILABLE, reason="cleanlab not installed")
def test_audit_data_quality_with_noisy_labels():
    df = _make_classification_df()
    # Introduce label noise by flipping some labels
    target = df["target"].to_numpy().copy()
    n_flip = 20
    flip_indices = np.random.RandomState(42).choice(len(target), size=n_flip, replace=False)
    target[flip_indices] = 1 - target[flip_indices]
    df = df.with_columns(target=pl.Series(target))

    report = audit_data_quality(df, "target", enabled=True)
    assert report["enabled"] is True
    assert report["n_issues"] > 0
    assert report["noise_rate"] > 0


def test_clean_noise_drops_flagged_rows():
    df = pl.DataFrame({"a": range(10), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    report = {
        "enabled": True,
        "quality_scores": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9],
    }
    cleaned, summary = clean_noise(df, report, "target", quality_threshold=0.5)
    assert len(cleaned) == 7
    assert summary["n_dropped"] == 3
    assert summary["n_before"] == 10
    assert summary["n_after"] == 7


def test_clean_noise_no_flagged_indices():
    df = pl.DataFrame({"a": range(5), "target": [0, 1, 0, 1, 0]})
    report = {"enabled": True, "flagged_indices": []}
    cleaned, summary = clean_noise(df, report, "target")
    assert len(cleaned) == 5
    assert summary["n_dropped"] == 0


def test_clean_noise_uses_quality_threshold_when_scores_present():
    df = pl.DataFrame({"a": range(5), "target": [0, 1, 0, 1, 0]})
    report = {
        "enabled": True,
        "quality_scores": [0.9, 0.4, 0.6, 0.1, 0.5],
        "flagged_indices": [0, 1, 2, 3, 4],
    }
    cleaned, summary = clean_noise(df, report, "target", quality_threshold=0.5)
    assert len(cleaned) == 3
    assert summary["n_dropped"] == 2


def test_clean_noise_disabled_report():
    df = pl.DataFrame({"a": range(5), "target": [0, 1, 0, 1, 0]})
    report = {"enabled": False, "message": "skipped"}
    cleaned, summary = clean_noise(df, report, "target")
    assert len(cleaned) == 5
    assert summary["n_dropped"] == 0
