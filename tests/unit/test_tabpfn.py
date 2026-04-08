"""Tests for TabPFN model row-count guardrail."""

import numpy as np
import pytest

from core.models.tabular_foundation.tabpfn_model import DataSizeError, TabPFNModel


def test_tabpfn_accepts_small_data():
    model = TabPFNModel(task="classification")
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    try:
        model.fit(X, y)
    except Exception as e:
        is_license_error = (
            "TabPFNClassifier" in str(e)
            or "TabPFNLicenseError" in str(type(e).__name__)
            or "license" in str(e).lower()
        )
        if is_license_error:
            pytest.skip("TabPFN license not accepted (requires browser auth or TABPFN_TOKEN)")
        raise


def test_tabpfn_rejects_large_data():
    model = TabPFNModel(task="classification")
    X = np.random.randn(15000, 5)
    y = np.random.randint(0, 2, 15000)
    with pytest.raises(DataSizeError, match="max 10000 rows"):
        model.fit(X, y)


def test_tabpfn_max_rows_configurable():
    """Test that MAX_ROWS can be configured via constructor."""
    from unittest.mock import Mock

    # Create model with custom max rows
    model = TabPFNModel(task="classification", max_rows=5000)

    # Should fail at 5001 rows
    X = np.random.rand(5001, 2)
    y = np.random.randint(0, 2, 5001)

    with pytest.raises(DataSizeError, match="max 5000 rows"):
        model.fit(X, y)

    # Should succeed at 5000 rows
    X_small = np.random.rand(5000, 2)
    y_small = np.random.randint(0, 2, 5000)
    # Mock the model to avoid actual training
    model._build_model = Mock()
    model.fit(X_small, y_small)  # Should not raise
