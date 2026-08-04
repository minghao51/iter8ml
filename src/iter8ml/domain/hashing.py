"""Deterministic hashing helpers for plans, frames, and provenance."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import polars as pl


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value with stable ordering."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    """Return a full content digest using the canonical JSON representation."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dataframe_digest(df: pl.DataFrame) -> str:
    """Hash schema and row content without depending on Polars partition layout."""
    header = {
        "columns": df.columns,
        "schema": {name: str(dtype) for name, dtype in df.schema.items()},
        "row_count": df.height,
    }
    row_digests = sorted(
        hashlib.sha256(canonical_json(row).encode("utf-8")).digest()
        for row in df.iter_rows(named=True)
    )
    hasher = hashlib.sha256(canonical_json(header).encode("utf-8"))
    for row_digest in row_digests:
        hasher.update(row_digest)
    return "sha256:" + hasher.hexdigest()
