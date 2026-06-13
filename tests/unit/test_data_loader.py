"""Tests for load_data() dispatch function."""

import polars as pl
import pytest

from iter8ml.data.loader import load_data
from iter8ml.exceptions import DataLoadError


def test_load_data_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    df.write_csv(csv_path)
    result = load_data(csv_path)
    assert len(result) == 3
    assert result.columns == ["a", "b"]


def test_load_data_parquet(tmp_path):
    pq_path = tmp_path / "test.parquet"
    df = pl.DataFrame({"x": [10, 20], "y": ["a", "b"]})
    df.write_parquet(pq_path)
    result = load_data(pq_path)
    assert len(result) == 2
    assert result.columns == ["x", "y"]


def test_load_data_file_not_found():
    with pytest.raises(DataLoadError):
        load_data("/nonexistent/path/data.csv")


def test_load_data_unsupported_format(tmp_path):
    json_path = tmp_path / "data.json"
    json_path.write_text("[]")
    with pytest.raises(DataLoadError, match="Unsupported file format"):
        load_data(json_path)


def test_load_data_csv_with_separator(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a;b\n1;2\n3;4\n")
    from iter8ml.data.loader import load_csv

    result = load_csv(csv_path, separator=";")
    assert len(result) == 2
