"""Security tests for load_sqlite SQL injection protection."""

import sqlite3

import polars as pl
import pytest

from iter8ml.data.loader import load_csv, load_parquet, load_sqlite


@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()
    return db_path


def test_sql_injection_drop_table(test_db):
    with pytest.raises(ValueError, match="Only SELECT queries are supported"):
        load_sqlite(test_db, "DROP TABLE t")


def test_sql_injection_delete(test_db):
    with pytest.raises(ValueError, match="Only SELECT queries are supported"):
        load_sqlite(test_db, "DELETE FROM t")


def test_sql_injection_insert(test_db):
    with pytest.raises(ValueError, match="Only SELECT queries are supported"):
        load_sqlite(test_db, "INSERT INTO t VALUES (3, 'c')")


def test_sql_injection_update(test_db):
    with pytest.raises(ValueError, match="Only SELECT queries are supported"):
        load_sqlite(test_db, "UPDATE t SET val = 'x' WHERE id = 1")


def test_sql_injection_multi_statement(test_db):
    with pytest.raises(ValueError, match="Multiple statements are not supported"):
        load_sqlite(test_db, "SELECT * FROM t; DROP TABLE t")


def test_valid_select_works(test_db):
    df = load_sqlite(test_db, "SELECT * FROM t")
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    assert set(df.columns) == {"id", "val"}


def test_empty_query_rejected(test_db):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        load_sqlite(test_db, "")


def test_nonexistent_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        load_sqlite(tmp_path / "no_such.db", "SELECT 1")


def test_load_csv_basic(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n3,4\n")
    df = load_csv(csv_path)
    assert len(df) == 2
    assert df.columns == ["x", "y"]


def test_load_parquet_basic(tmp_path):
    pq_path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [10, 20], "b": ["c", "d"]}).write_parquet(pq_path)
    df = load_parquet(pq_path)
    assert len(df) == 2
    assert df.columns == ["a", "b"]
