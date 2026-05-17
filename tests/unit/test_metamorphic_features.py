"""Metamorphic tests for feature engineering invariants."""

import numpy as np
import pytest

from iter8ml.data.features import _safe_ratio, transform_target

pytestmark = pytest.mark.metamorphic


class TestSafeRatioMetamorphic:
    """Metamorphic: _safe_ratio has known invariants."""

    def test_homogeneity_positive(self):
        a = np.array([1.0, 2.0, 4.0, 8.0])
        b = np.array([2.0, 4.0, 2.0, 4.0])
        k = 3.0
        result1 = _safe_ratio(a, b)
        result2 = _safe_ratio(k * a, k * b)
        np.testing.assert_allclose(result1, result2)

    def test_safe_ratio_idempotent(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result1 = _safe_ratio(a, b)
        result2 = _safe_ratio(result1, np.ones_like(result1))
        np.testing.assert_allclose(result1, result2)

    def test_zero_numerator_zero_result(self):
        a = np.zeros(10)
        b = np.ones(10)
        result = _safe_ratio(a, b)
        assert np.all(result == 0.0)

    def test_all_zero_denominator(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)
        result = _safe_ratio(a, b)
        assert np.all(result == 0.0)

    def test_negative_values(self):
        a = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        b = np.array([2.0, 2.0, 1.0, 2.0, 2.0])
        result = _safe_ratio(a, b)
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(result, expected)


class TestTransformTargetMetamorphic:
    """Metamorphic: transform_target followed by inverse returns original."""

    def test_log1p_roundtrip(self):
        y = np.array([1.0, 2.0, 5.0, 10.0, 100.0])
        y_transformed, result, transformer = transform_target(y, method="log1p")
        assert result.applied is True
        assert result.method == "log1p"
        assert transformer is not None
        y_roundtrip = transformer.inverse_transform(y_transformed)
        np.testing.assert_allclose(y_roundtrip, y, rtol=1e-10)

    def test_none_method_returns_identity(self):
        y = np.array([1.0, -2.0, 5.0, 0.0, 100.0])
        y_out, result, transformer = transform_target(y, method="none")
        assert result.applied is False
        assert transformer is None
        np.testing.assert_array_equal(y_out, y)

    def test_low_skewness_returns_identity(self):
        y = np.array([10.0, 11.0, 10.5, 9.8, 10.2])
        y_out, result, transformer = transform_target(y, method="auto", skewness_threshold=5.0)
        assert result.applied is False
        assert transformer is None
        np.testing.assert_array_equal(y_out, y)

    def test_auto_selects_boxcox_for_positive_data(self):
        y = np.array([1.0, 2.0, 5.0, 10.0, 100.0, 200.0])
        _, _, transformer = transform_target(y, method="auto")
        assert transformer is not None

    def test_auto_selects_yeojohnson_for_mixed_sign(self):
        y = np.array([1.0, -2.0, 5.0, -10.0, 100.0])
        _, result, transformer = transform_target(y, method="auto")
        assert result.applied is True
        assert transformer is not None
