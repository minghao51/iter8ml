"""Split validation checks independent of model implementation."""

from __future__ import annotations

from typing import Any

import polars as pl


def validate_split(split_frame: pl.DataFrame) -> dict[str, Any]:
    required = {"row_id", "fold", "role", "repeat"}
    missing = required - set(split_frame.columns)
    if missing:
        raise ValueError(f"split frame missing required columns: {sorted(missing)}")
    if split_frame.is_empty():
        return {"ok": False, "errors": ["split frame is empty"], "folds": 0}
    errors: list[str] = []
    invalid_roles = set(split_frame["role"].unique().to_list()) - {"train", "validation", "test"}
    if invalid_roles:
        errors.append(f"split frame has invalid roles: {sorted(invalid_roles)}")
    duplicate_count = split_frame.select(
        pl.struct(["row_id", "fold", "role", "repeat"]).is_duplicated().sum()
    ).item()
    if duplicate_count:
        errors.append(f"split frame has {duplicate_count} duplicate membership rows")
    for fold in split_frame["fold"].unique().to_list():
        fold_frame = split_frame.filter(pl.col("fold") == fold)
        train = set(fold_frame.filter(pl.col("role") == "train")["row_id"].to_list())
        validation = set(fold_frame.filter(pl.col("role") == "validation")["row_id"].to_list())
        if not train:
            errors.append(f"fold {fold} has no training rows")
        if not validation:
            errors.append(f"fold {fold} has no validation rows")
        overlap = train & validation
        if overlap:
            errors.append(f"fold {fold} has {len(overlap)} train/validation overlaps")
    return {"ok": not errors, "errors": errors, "folds": split_frame["fold"].n_unique()}
