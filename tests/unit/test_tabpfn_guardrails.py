"""Tests for TabPFN guardrails."""

import numpy as np
import pytest

from tabular_blueprint.models.tabular_foundation.tabpfn_model import (
    DataSizeError,
    GPUUnavailableError,
    TabPFNModel,
)


class TestTabPFNGuardrails:
    def test_default_max_rows_is_50k(self):
        assert TabPFNModel.DEFAULT_MAX_ROWS == 50_000

    def test_data_size_error_on_oversized(self):
        model = TabPFNModel(task="classification", max_rows=100)
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, 200)
        with pytest.raises(DataSizeError, match="max 100 rows"):
            model.fit(X, y)

    def test_gpu_check_raises_on_no_gpu(self):
        model = TabPFNModel(task="classification")
        with pytest.raises(GPUUnavailableError):
            model._check_gpu()
