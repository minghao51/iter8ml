"""Stateless helpers for detecting, extracting, and augmenting sparse categorical features."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def detect_high_cardinality_columns(
    df: pl.DataFrame,
    max_categories: int = 50,
    target_col: str = "",
) -> list[str]:
    """Return column names with unique count > *max_categories*.

    Checks string, categorical, and integer columns.  Skips the target
    column if provided.
    """
    cols: list[str] = []
    for c in df.columns:
        if c == target_col:
            continue
        dtype = df[c].dtype
        is_cat = dtype in (
            pl.Categorical,
            pl.String,
            pl.Utf8,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )
        if not is_cat:
            continue
        if df[c].n_unique() > max_categories:
            cols.append(c)
    return cols


def extract_cat_codes(
    df: pl.DataFrame,
    cat_columns: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, dict]]:
    """Extract contiguous integer codes for each categorical column.

    Returns:
        codes:       ``{col_name: np.ndarray of int64, shape (n_rows,)}``
        vocab_sizes: ``{col_name: n_unique_values}``
        mappings:    ``{col_name: {original_value: contiguous_code}}``
    """
    codes: dict[str, np.ndarray] = {}
    vocab_sizes: dict[str, int] = {}
    mappings: dict[str, dict] = {}

    for col in cat_columns:
        series = df[col]
        unique_vals = series.unique().sort()
        val_to_code: dict = {v: i for i, v in enumerate(unique_vals)}
        mapping = val_to_code

        def _map_val(v: Any, _m: dict = mapping) -> int:
            return _m.get(v, 0)

        code_series = series.map_elements(_map_val, return_dtype=pl.Int64)
        codes[col] = code_series.to_numpy().astype(np.int64)
        vocab_sizes[col] = len(unique_vals)
        mappings[col] = val_to_code

    return codes, vocab_sizes, mappings


def augment_with_embeddings(
    X: np.ndarray,
    embeddings: np.ndarray,
    feature_names: list[str],
    cat_columns: list[str],
    per_col_dim: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Remove categorical columns from *X* and concatenate embedding vectors.

    Embedding feature names are formatted as
    ``"{col}_emb_{i}"`` for each dimension *i* per column.
    If *per_col_dim* does not evenly divide the embedding width, generic
    names ``"emb_{i}"`` are used instead.
    """
    cat_set = set(cat_columns)
    keep_indices = [i for i, name in enumerate(feature_names) if name not in cat_set]
    kept_names = [feature_names[i] for i in keep_indices]
    X_kept = X[:, keep_indices] if keep_indices else np.empty((X.shape[0], 0), dtype=X.dtype)

    n_emb_cols = embeddings.shape[1]
    emb_names: list[str] = []

    if per_col_dim and per_col_dim > 0 and len(cat_columns) * per_col_dim == n_emb_cols:
        for col in sorted(cat_columns):
            emb_names.extend(f"{col}_emb_{d}" for d in range(per_col_dim))
    else:
        emb_names = [f"emb_{i}" for i in range(n_emb_cols)]

    if X_kept.shape[1] > 0:
        X_aug = np.hstack([X_kept, embeddings.astype(X.dtype)])
    else:
        X_aug = embeddings.astype(X.dtype)

    return X_aug, kept_names + emb_names
