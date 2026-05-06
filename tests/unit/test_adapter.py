"""Tests for DataAdapter round-trip conversions."""

import numpy as np
import polars as pl

from tabular_blueprint.data.adapter import DataAdapter


def test_numpy_conversion():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    adapter = DataAdapter()
    X, y = adapter.transform(df, "target")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (3, 2)
    assert y.shape == (3,)
