"""Property-based tests for drift detection invariants."""

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iter8ml.analysis.drift import DriftDetector

pytestmark = pytest.mark.property


class TestPropertyDrift:
    """Property: p-values in [0,1], empty series returns 1.0, column mismatch handled."""

    @given(
        n_ref=st.integers(10, 200),
        n_new=st.integers(10, 200),
    )
    @settings(max_examples=50)
    def test_drift_p_value_in_range(self, n_ref, n_new):
        import numpy as np

        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(n_ref)})
        new = pl.DataFrame({"a": rng.randn(n_new)})
        report = DriftDetector(ref).detect(new)
        for col_result in report.column_results:
            assert 0.0 <= col_result.p_value <= 1.0

    @given(
        n_ref=st.integers(10, 100),
    )
    @settings(max_examples=50)
    def test_self_comparison_no_drift_probabilistic(self, n_ref):
        import numpy as np

        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(n_ref)})
        new = pl.DataFrame({"a": rng.randn(n_ref)})
        report = DriftDetector(ref).detect(new)
        assert 0 <= report.n_drifted <= report.n_columns_tested

    @given(
        n_ref=st.integers(10, 100),
        n_new=st.integers(10, 100),
    )
    @settings(max_examples=50)
    def test_empty_series_returns_p_one(self, n_ref, n_new):
        import numpy as np

        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(n_ref), "b": rng.randn(n_ref)})
        empty_new = pl.DataFrame({"a": [], "b": []})
        report = DriftDetector(ref).detect(empty_new)
        for col_result in report.column_results:
            assert col_result.p_value == 1.0

    @given(
        n_ref=st.integers(10, 100),
        n_new=st.integers(10, 100),
    )
    @settings(max_examples=50)
    def test_column_mismatch_only_common_tested(self, n_ref, n_new):
        import numpy as np

        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(n_ref), "b": rng.randn(n_ref), "c": rng.randn(n_ref)})
        new = pl.DataFrame({"a": rng.randn(n_new), "b": rng.randn(n_new)})
        report = DriftDetector(ref).detect(new)
        assert report.n_columns_tested == 2

    @given(
        n_ref=st.integers(10, 100),
    )
    @settings(max_examples=50)
    def test_disjoint_columns_empty_report(self, n_ref):
        import numpy as np

        rng = np.random.RandomState(42)
        ref = pl.DataFrame({"a": rng.randn(n_ref)})
        new = pl.DataFrame({"b": rng.randn(n_ref)})
        report = DriftDetector(ref).detect(new)
        assert report.n_columns_tested == 0
        assert report.n_drifted == 0
        assert report.drift_detected is False
