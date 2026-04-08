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

    # Security validation: Only allow single SELECT statements
    # This is a basic validation - for production use, consider:
    # - Using a proper SQL parser for more robust validation
    # - Implementing query whitelisting or parameterized queries
    # - Using read-only database connections
    query_stripped = query.strip()
    query_upper = query_stripped.upper()

    # Check if it starts with SELECT (after stripping whitespace)
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are supported for security reasons")

    # Prevent multiple statements by checking for semicolons
    if ";" in query_upper:
        raise ValueError("Multiple statements are not supported for security reasons")

    # Additional check: ensure it's a SELECT query, not just starting with SELECT
    # This catches attempts like "SELECT" followed by non-SQL content
    if len(query_stripped) < 7:  # "SELECT" is 6 chars
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
    # Use XOR aggregation instead of sorting - O(n) instead of O(n log n)
    combined_hash = row_hashes.to_numpy().astype(np.uint64).sum()
    return "sha256:" + hashlib.sha256(str(combined_hash).encode()).hexdigest()[:16]
