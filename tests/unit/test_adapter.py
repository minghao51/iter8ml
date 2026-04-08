"""Tests for DataAdapter round-trip conversions."""

import numpy as np
import polars as pl
import pytest

from core.data.adapter import DataAdapter


def test_numpy_conversion():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    adapter = DataAdapter(target_format="numpy")
    X, y = adapter.transform(df, "target")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_tensor_conversion():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    adapter = DataAdapter(target_format="tensor")
    X, y = adapter.transform(df, "target")
    import torch

    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_dataset_conversion():
    datasets = pytest.importorskip("datasets")

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    adapter = DataAdapter(target_format="dataset")
    dataset = adapter.transform(df, "target")

    assert isinstance(dataset, datasets.Dataset)
    assert dataset.num_rows == 3
    assert set(dataset.column_names) == {"a", "b", "label"}
