"""Polars-native feature engineering and preprocessing."""

import polars as pl
from polars import selectors as cs


def cast_types(df: pl.DataFrame, type_map: dict[str, str]) -> pl.DataFrame:
    """Cast columns to specified types."""
    return df.with_columns([pl.col(col).cast(dtype) for col, dtype in type_map.items()])


def fill_nulls(
    df: pl.DataFrame,
    *,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pl.DataFrame:
    """Fill null values using specified strategies."""
    exprs = []

    if numeric_strategy == "median":
        exprs.extend(
            [pl.col(c).fill_null(pl.col(c).median()) for c in df.select(cs.numeric()).columns]
        )
    elif numeric_strategy == "mean":
        exprs.extend(
            [pl.col(c).fill_null(pl.col(c).mean()) for c in df.select(cs.numeric()).columns]
        )
    elif numeric_strategy == "zero":
        exprs.extend([pl.col(c).fill_null(0) for c in df.select(cs.numeric()).columns])

    if categorical_strategy == "mode":
        for c in df.select(cs.categorical()).columns:
            mode_val = df[c].mode().first()
            if mode_val is not None:
                exprs.append(pl.col(c).fill_null(mode_val))
    elif categorical_strategy == "unknown":
        exprs.extend([pl.col(c).fill_null("unknown") for c in df.select(cs.categorical()).columns])

    return df.with_columns(exprs) if exprs else df


def decompose_dates(df: pl.DataFrame, date_cols: list[str] | None = None) -> pl.DataFrame:
    """Decompose datetime columns into year, month, day, day_of_week components."""
    if date_cols is None:
        date_cols = [
            c for c, dtype in df.schema.items() if dtype == pl.Datetime or dtype == pl.Date
        ]

    exprs = []
    for col in date_cols:
        prefix = col.replace("_date", "").replace("_dt", "")
        exprs.extend(
            [
                pl.col(col).dt.year().alias(f"{prefix}_year"),
                pl.col(col).dt.month().alias(f"{prefix}_month"),
                pl.col(col).dt.day().alias(f"{prefix}_day"),
                pl.col(col).dt.weekday().alias(f"{prefix}_day_of_week"),
            ]
        )

    return df.with_columns(exprs) if exprs else df


def encode_categoricals(df: pl.DataFrame, strategy: str = "label") -> pl.DataFrame:
    """Encode categorical columns."""
    cat_cols = df.select(cs.categorical()).columns

    if strategy == "label":
        exprs = []
        for col in cat_cols:
            exprs.append(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
        return df.with_columns(exprs) if exprs else df

    return df


def pipeline(
    df: pl.DataFrame,
    *,
    do_fill_nulls: bool = True,
    do_decompose_dates: bool = True,
    do_encode_categoricals: bool = False,
    date_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Apply full preprocessing pipeline."""
    if do_fill_nulls:
        df = fill_nulls(df)
    if do_decompose_dates:
        df = decompose_dates(df, date_cols)
    if do_encode_categoricals:
        df = encode_categoricals(df)
    return df
