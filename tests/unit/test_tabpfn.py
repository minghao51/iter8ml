"""Tests for TabPFN model row-count and device resolution."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from iter8ml.engine.models.tabpfn_model import (
    DataSizeError,
    TabPFNModel,
)


def test_tabpfn_rejects_large_data():
    model = TabPFNModel(task="classification")
    X = np.random.randn(60_000, 5)
    y = np.random.randint(0, 2, 60_000)
    with pytest.raises(DataSizeError, match="max 50000 rows"):
        model.fit(X, y)


def test_tabpfn_resolve_device_cpu_fallback():
    model = TabPFNModel(task="classification")
    with patch.dict("sys.modules", {"torch": None}):
        device = model._resolve_device()
    assert device == "cpu"


def test_tabpfn_max_rows_configurable():
    model = TabPFNModel(task="classification", max_rows=5000)
    X = np.random.rand(5001, 2)
    y = np.random.randint(0, 2, 5001)

    with pytest.raises(DataSizeError, match="max 5000 rows"):
        model.fit(X, y)


def test_tabpfn_accepts_small_data_with_mocked_build():
    model = TabPFNModel(task="classification")
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    mock_inner = Mock()
    model._build_model = Mock(return_value=mock_inner)
    model.fit(X, y)
    mock_inner.fit.assert_called_once_with(X, y)
