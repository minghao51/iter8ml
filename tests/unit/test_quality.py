"""Tests for cleanlab-based data quality audit."""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from core.data.quality import audit_data_quality


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


def test_audit_data_quality_single_class():
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "target": [0, 0, 0],
    })
    report = audit_data_quality(df, "target", enabled=True)
    assert report["enabled"] is False
    assert "2 classes" in report["message"]


def test_audit_data_quality_output_path():
    df = _make_classification_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "quality_report.json"
        report = audit_data_quality(df, "target", output_path=str(output_path), enabled=True)
        assert output_path.exists()
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["enabled"] is True
        assert saved["n_rows"] == report["n_rows"]


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
