"""Property-based tests for data quality audit invariants.

Note: audit_data_quality runs sklearn cross-validation, making it
expensive. We keep max_examples low and set deadline=None.
"""

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.datasets import make_classification

from iter8ml.data.quality import audit_data_quality, clean_noise

pytestmark = pytest.mark.property


def _make_clf_data(n_samples, n_features):
    n_info = max(2, n_features - 1)
    n_info = min(n_info, n_features)
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_info,
        n_redundant=n_features - n_info,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=42,
    )


SLOW_SETTINGS = settings(deadline=None, max_examples=10)


class TestPropertyAuditQuality:
    """Property: audit_data_quality always returns dict with expected keys."""

    @given(
        n_samples=st.integers(50, 200),
        n_features=st.integers(2, 6),
    )
    @settings(deadline=None, max_examples=10)
    def test_report_structure_when_disabled(self, n_samples, n_features):
        X, y = _make_clf_data(n_samples, n_features)
        df = pl.DataFrame({**{f"f_{i}": X[:, i] for i in range(n_features)}, "target": y})
        report = audit_data_quality(df, target_col="target", enabled=False)
        assert isinstance(report, dict)
        assert report.get("enabled") is False
        assert "message" in report

    @given(
        n_samples=st.integers(50, 200),
        n_features=st.integers(2, 6),
    )
    @SLOW_SETTINGS
    def test_report_keys_when_enabled(self, n_samples, n_features):
        X, y = _make_clf_data(n_samples, n_features)
        df = pl.DataFrame({**{f"f_{i}": X[:, i] for i in range(n_features)}, "target": y})
        report = audit_data_quality(df, target_col="target")
        expected_keys = {
            "enabled",
            "n_rows",
            "n_issues",
            "noise_rate",
            "flagged_indices",
            "quality_scores",
            "mean_quality_score",
        }
        if report.get("enabled"):
            assert expected_keys.issubset(report.keys())
            assert 0 <= report["noise_rate"] <= 1.0
            assert isinstance(report["flagged_indices"], list)


class TestPropertyCleanNoise:
    """Property: clean_noise never increases row count, preserves schema."""

    @given(
        n_samples=st.integers(50, 150),
        n_features=st.integers(2, 4),
    )
    @SLOW_SETTINGS
    def test_clean_noise_row_count_non_increasing(self, n_samples, n_features):
        X, y = _make_clf_data(n_samples, n_features)
        df = pl.DataFrame({**{f"f_{i}": X[:, i] for i in range(n_features)}, "target": y})
        report = audit_data_quality(df, target_col="target")
        if not report.get("enabled"):
            return
        cleaned, _ = clean_noise(df, report, target_col="target")
        assert len(cleaned) <= len(df)
        assert cleaned.columns == df.columns

    @given(
        n_samples=st.integers(50, 150),
        n_features=st.integers(2, 4),
    )
    @SLOW_SETTINGS
    def test_clean_noise_summary_counts(self, n_samples, n_features):
        X, y = _make_clf_data(n_samples, n_features)
        df = pl.DataFrame({**{f"f_{i}": X[:, i] for i in range(n_features)}, "target": y})
        report = audit_data_quality(df, target_col="target")
        if not report.get("enabled"):
            return
        cleaned, summary = clean_noise(df, report, target_col="target")
        assert summary["n_before"] == len(df)
        assert summary["n_after"] == len(cleaned)
        assert summary["n_dropped"] == len(df) - len(cleaned)
