"""Tests for DataAdapter — Polars to NumPy conversion edge cases."""

import numpy as np
import polars as pl

from iter8ml.data.adapter import DataAdapter


def test_transform_valid_data():
    df = pl.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    adapter = DataAdapter()
    X, y = adapter.transform(df, "target")
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert list(X[:, 0]) == [1.0, 2.0, 3.0]


def test_transform_handles_nan():
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0], "target": [0, 1, 0]})
    adapter = DataAdapter()
    X, _y = adapter.transform(df, "target")
    assert X.shape == (3, 1)
    assert np.isnan(X).any()


def test_transform_handles_inf():
    df = pl.DataFrame({"a": [1.0, float("inf"), 3.0], "target": [0, 1, 0]})
    adapter = DataAdapter()
    X, _y = adapter.transform(df, "target")
    assert X.shape == (3, 1)
    assert np.isinf(X).any()


def test_transform_single_feature():
    df = pl.DataFrame({"x": [10.0, 20.0], "target": [1, 0]})
    adapter = DataAdapter()
    X, y = adapter.transform(df, "target")
    assert X.shape == (2, 1)
    assert y.shape == (2,)
