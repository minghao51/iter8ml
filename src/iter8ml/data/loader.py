"""Polars-based data ingestion from various sources."""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl


def load_csv(
    path: str | Path,
    *,
    separator: str = ",",
    infer_schema_length: int = 1000,
    low_memory: bool = False,
) -> pl.DataFrame:
    """Load a CSV file into a Polars DataFrame."""
    return pl.read_csv(
        str(path),
        separator=separator,
        infer_schema_length=infer_schema_length,
        low_memory=low_memory,
    )


def load_parquet(path: str | Path) -> pl.DataFrame:
    """Load a Parquet file into a Polars DataFrame."""
    return pl.read_parquet(str(path))


def load_data(path: str | Path) -> pl.DataFrame:
    """Load a data file (CSV or Parquet) into a Polars DataFrame."""
    path = Path(path)
    if path.suffix == ".parquet":
        return load_parquet(path)
    elif path.suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}. Supported: .csv, .parquet")


def load_sqlite(db_path: str | Path, query: str) -> pl.DataFrame:
    """Execute a SQL query against a SQLite database and return results as a Polars DataFrame.

    Args:
        db_path: Path to SQLite database file.
        query: SQL query to execute.

    Returns:
        Polars DataFrame with query results.

    Raises:
        FileNotFoundError: If database file doesn't exist.
        ValueError: If query is invalid or database is corrupted.
    """
    import sqlite3

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query_stripped = query.strip()
    query_upper = query_stripped.upper()

    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are supported for security reasons")

    if ";" in query_stripped:
        stripped_query = query_stripped.rstrip(";").strip()
        if ";" in stripped_query:
            raise ValueError("Multiple statements are not supported for security reasons")

    blocked_keywords = ("DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC", "EXECUTE")
    upper_no_select = query_upper.replace("SELECT", "").replace("FROM", "").replace("WHERE", "")
    upper_no_select = upper_no_select.replace("AND", "").replace("OR", "").replace("NOT", "")
    upper_no_select = upper_no_select.replace("JOIN", "").replace("LEFT", "").replace("RIGHT", "")
    upper_no_select = upper_no_select.replace("INNER", "").replace("OUTER", "").replace("ON", "")
    upper_no_select = upper_no_select.replace("GROUP", "").replace("ORDER", "").replace("BY", "")
    upper_no_select = upper_no_select.replace("HAVING", "").replace("LIMIT", "").replace("AS", "")
    upper_no_select = upper_no_select.replace("IN", "").replace("IS", "").replace("NULL", "")
    upper_no_select = (
        upper_no_select.replace("LIKE", "").replace("BETWEEN", "").replace("DISTINCT", "")
    )
    for keyword in blocked_keywords:
        if keyword in upper_no_select:
            raise ValueError(f"Destructive keyword '{keyword}' is not allowed in queries")

    if len(query_stripped) < 7:
        raise ValueError("Invalid SELECT query")

    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pl.read_database(query, conn)
    except sqlite3.Error as e:
        raise ValueError(f"Database error: {e}") from e

    return df


def get_data_hash(df: pl.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a Polars DataFrame."""
    row_hashes = df.hash_rows()
    combined_hash = int(np.bitwise_xor.reduce(row_hashes.to_numpy().astype(np.uint64)))
    return "sha256:" + hashlib.sha256(str(combined_hash).encode()).hexdigest()[:16]
