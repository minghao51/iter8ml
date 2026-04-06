"""Polars-based data ingestion from various sources."""

import hashlib
from pathlib import Path

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
    """Execute a SQL query against a SQLite database and return results as a Polars DataFrame."""
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        df = pl.read_database(query, conn)
    return df


def get_data_hash(df: pl.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a Polars DataFrame."""
    row_hashes = df.hash_rows()
    combined = str(sorted(row_hashes.to_list())).encode()
    return "sha256:" + hashlib.sha256(combined).hexdigest()[:16]
