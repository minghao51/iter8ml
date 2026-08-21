"""Polars-based data ingestion from various sources."""

import re
from pathlib import Path

import polars as pl

from iter8ml.exceptions import DataLoadError, track_errors


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


@track_errors(DataLoadError)
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

    query_text = query.strip()
    query_upper = query_text.upper()

    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are supported for security reasons")

    if ";" in query_text:
        query_clean = query_text.rstrip(";").strip()
        if ";" in query_clean:
            raise ValueError("Multiple statements are not supported for security reasons")

    _BLOCKED_KEYWORDS = frozenset(
        {
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "ALTER",
            "CREATE",
            "EXEC",
            "EXECUTE",
        }
    )
    tokens = set(re.findall(r"[A-Z]+", query_upper))
    blocked = tokens & _BLOCKED_KEYWORDS
    if blocked:
        raise ValueError(f"Destructive keywords not allowed: {sorted(blocked)}")

    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pl.read_database(query, conn)
    except sqlite3.Error as e:
        raise ValueError(f"Database error: {e}") from e

    return df
