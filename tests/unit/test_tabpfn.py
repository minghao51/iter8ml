"""Tests for TabPFN model row-count and GPU guardrails."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from tabular_blueprint.models.tabular_foundation.tabpfn_model import (
    DataSizeError,
    GPUUnavailableError,
    TabPFNModel,
)


def test_tabpfn_rejects_large_data():
    model = TabPFNModel(task="classification")
    X = np.random.randn(60_000, 5)
    y = np.random.randint(0, 2, 60_000)
    with pytest.raises(DataSizeError, match="max 50000 rows"):
        model.fit(X, y)


def test_tabpfn_gpu_check_raises_without_gpu():
    model = TabPFNModel(task="classification")
    with pytest.raises(GPUUnavailableError):
        model._check_gpu()


def test_tabpfn_max_rows_configurable():
    model = TabPFNModel(task="classification", max_rows=5000)
    X = np.random.rand(5001, 2)
    y = np.random.randint(0, 2, 5001)

    with pytest.raises(DataSizeError, match="max 5000 rows"):
        model.fit(X, y)


def test_tabpfn_accepts_small_data_with_mocked_gpu():
    model = TabPFNModel(task="classification")
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model._build_model = Mock()
    with patch.object(model, "_check_gpu"):
        model.fit(X, y)
    model.model.fit.assert_called_once()
