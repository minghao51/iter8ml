"""Tests for data loaders and hash function."""

import tempfile

import polars as pl
import pytest

from tabular_blueprint.data.loaders import get_data_hash, load_csv, load_parquet, load_sqlite


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


def test_load_sqlite_invalid_path():
    """Test loading from non-existent database path."""
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        load_sqlite("/nonexistent/db.sqlite", "SELECT 1")


def test_load_sqlite_invalid_query(tmp_path):
    """Test loading with invalid SQL query."""
    db_file = tmp_path / "test.db"
    # Create valid database
    import sqlite3

    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")

    with pytest.raises(ValueError, match="Only SELECT queries are supported"):
        load_sqlite(db_file, "INVALID SQL QUERY")


def test_load_sqlite_empty_query(tmp_path):
    """Test loading with empty query."""
    db_file = tmp_path / "test.db"
    import sqlite3

    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        load_sqlite(db_file, "")


def test_load_sqlite_whitespace_query(tmp_path):
    """Test loading with whitespace-only query."""
    db_file = tmp_path / "test.db"
    import sqlite3

    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        load_sqlite(db_file, "   \t\n")


def test_load_sqlite_multiple_statements(tmp_path):
    """Test loading with multiple statements (security check)."""
    db_file = tmp_path / "test.db"
    import sqlite3

    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")

    with pytest.raises(ValueError, match="Multiple statements are not supported"):
        load_sqlite(db_file, "SELECT * FROM test; DROP TABLE test")


def test_load_sqlite_empty_result(tmp_path):
    """Test loading query with no results."""
    import sqlite3

    db_file = tmp_path / "empty.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")

    df = load_sqlite(db_file, "SELECT * FROM test WHERE 1=0")
    assert len(df) == 0
