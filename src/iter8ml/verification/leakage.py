"""Split and leakage checks independent of model implementation."""

from __future__ import annotations

import polars as pl


def validate_split_frame(split_frame: pl.DataFrame) -> dict[str, object]:
    required = {"row_id", "fold", "role"}
    missing = required - set(split_frame.columns)
    if missing:
        raise ValueError(f"split frame missing required columns: {sorted(missing)}")
    errors: list[str] = []
    for fold in split_frame["fold"].unique().to_list():
        fold_frame = split_frame.filter(pl.col("fold") == fold)
        train = set(fold_frame.filter(pl.col("role") == "train")["row_id"].to_list())
        validation = set(fold_frame.filter(pl.col("role") == "validation")["row_id"].to_list())
        overlap = train & validation
        if overlap:
            errors.append(f"fold {fold} has {len(overlap)} train/validation overlaps")
    return {"ok": not errors, "errors": errors, "folds": split_frame["fold"].n_unique()}
