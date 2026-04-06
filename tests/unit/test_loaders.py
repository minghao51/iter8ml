"""Tests for data loaders and hash function."""

import tempfile

import polars as pl

from core.data.loaders import get_data_hash, load_csv, load_parquet


def test_get_data_hash_consistency():
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    hash1 = get_data_hash(df)
    hash2 = get_data_hash(df)
    assert hash1 == hash2


def test_get_data_hash_mutation():
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"a": [1, 2, 4]})
    assert get_data_hash(df1) != get_data_hash(df2)


def test_load_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n4,5,6\n")
        f.flush()
        df = load_csv(f.name)
        assert len(df) == 2
        assert df.columns == ["a", "b", "c"]


def test_load_parquet():
    df_original = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df_original.write_parquet(f.name)
        df_loaded = load_parquet(f.name)
        assert len(df_loaded) == 3
        assert df_loaded.columns == ["x", "y"]
