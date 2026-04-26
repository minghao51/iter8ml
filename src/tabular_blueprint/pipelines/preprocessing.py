"""Hamilton-powered data pipeline: DAG-based preprocessing."""

import polars as pl
from polars import selectors as cs


def raw_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df


def numeric_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return raw_dataframe.select(cs.numeric()).columns


def categorical_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return raw_dataframe.select(cs.categorical() | cs.string()).columns


def date_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return [
        c for c, dtype in raw_dataframe.schema.items() if dtype == pl.Datetime or dtype == pl.Date
    ]


def type_map() -> dict[str, str]:
    return {}


def null_filled_df(
    fill_nulls_numeric: pl.DataFrame,
    fill_nulls_categorical: pl.DataFrame,
) -> pl.DataFrame:
    # fill_nulls_categorical already contains numeric nulls if we chain them,
    # but the current nodes are both based on raw_dataframe.
    # Let's fix the chain: raw -> numeric_filled -> cat_filled -> null_filled_df
    return fill_nulls_categorical


def fill_nulls_numeric(
    raw_dataframe: pl.DataFrame,
    numeric_columns: list[str],
) -> pl.DataFrame:
    exprs = [pl.col(c).fill_null(pl.col(c).median()) for c in numeric_columns]
    return raw_dataframe.with_columns(exprs) if exprs else raw_dataframe


def fill_nulls_categorical(
    fill_nulls_numeric: pl.DataFrame,
    categorical_columns: list[str],
) -> pl.DataFrame:
    exprs = []
    for c in categorical_columns:
        mode_val = fill_nulls_numeric[c].mode().first()
        if mode_val is not None:
            exprs.append(pl.col(c).fill_null(mode_val))
    return fill_nulls_numeric.with_columns(exprs) if exprs else fill_nulls_numeric


def decomposed_dates_df(
    null_filled_df: pl.DataFrame,
    date_columns: list[str],
) -> pl.DataFrame:
    exprs = []
    for col in date_columns:
        prefix = col.replace("_date", "").replace("_dt", "")
        exprs.extend(
            [
                pl.col(col).dt.year().alias(f"{prefix}_year"),
                pl.col(col).dt.month().alias(f"{prefix}_month"),
                pl.col(col).dt.day().alias(f"{prefix}_day"),
                pl.col(col).dt.weekday().alias(f"{prefix}_day_of_week"),
            ]
        )
    return null_filled_df.with_columns(exprs) if exprs else null_filled_df


def encoded_df(
    decomposed_dates_df: pl.DataFrame,
    categorical_columns: list[str],
) -> pl.DataFrame:
    cat_cols = [c for c in categorical_columns if c in decomposed_dates_df.columns]
    if not cat_cols:
        return decomposed_dates_df
    exprs = [pl.col(col).cast(pl.Categorical).to_physical().alias(col) for col in cat_cols]
    return decomposed_dates_df.with_columns(exprs)


def processed_dataframe(encoded_df: pl.DataFrame) -> pl.DataFrame:
    return encoded_df
