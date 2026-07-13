"""Source adapters for the supported local inputs."""

from __future__ import annotations

import polars as pl

from iter8ml.data.loader import load_data, load_sqlite
from iter8ml.domain.hashing import dataframe_digest
from iter8ml.domain.manifests import SourceSpec


def load_source(source: SourceSpec) -> pl.DataFrame:
    if source.source_type in {"csv", "parquet"}:
        return load_data(source.uri)
    if source.source_type == "sqlite":
        if not source.query:
            raise ValueError("SQLite source requires SourceSpec.query")
        return load_sqlite(source.uri, source.query)
    raise ValueError("memory sources must be supplied as a DataFrame to the execution service")


def inspect_source(source: SourceSpec) -> dict[str, object]:
    frame = load_source(source)
    return {
        "source_type": source.source_type,
        "uri": source.uri,
        "columns": frame.columns,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "row_count": frame.height,
        "fingerprint": dataframe_digest(frame),
    }
