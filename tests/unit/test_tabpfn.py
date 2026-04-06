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
