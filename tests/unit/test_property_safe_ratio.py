"""Property-based tests for _safe_ratio: no NaN/inf invariant."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iter8ml.data.features import _safe_ratio

pytestmark = pytest.mark.property


class TestSafeRatioProperty:
    """Property: _safe_ratio must never produce NaN or infinity."""

    @given(
        a=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=50),
        b=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_no_nan_or_inf(self, a, b):
        min_len = min(len(a), len(b))
        a_arr = np.array(a[:min_len])
        b_arr = np.array(b[:min_len])
        result = _safe_ratio(a_arr, b_arr)
        assert result is not None
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    @given(
        a=st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=1,
            max_size=30,
        ),
        b=st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=1,
            max_size=30,
        ),
    )
    @settings(max_examples=50)
    def test_output_shape_matches_input(self, a, b):
        min_len = min(len(a), len(b))
        a_arr = np.array(a[:min_len])
        b_arr = np.array(b[:min_len])
        result = _safe_ratio(a_arr, b_arr)
        assert result is not None
        assert result.shape == a_arr.shape

    def test_zero_denominator_returns_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        result = _safe_ratio(a, b)
        assert result is not None
        assert np.all(result == 0.0)

    def test_very_small_denominator_clipped(self):
        a = np.array([5.0])
        b = np.array([1e-15])
        result = _safe_ratio(a, b)
        assert result is not None
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_ratio_property_for_nonzero_b(self):
        a = np.array([6.0, 12.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = _safe_ratio(a, b)
        expected = np.array([3.0, 3.0, 0.5])
        assert result is not None
        np.testing.assert_allclose(result, expected)
