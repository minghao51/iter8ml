from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from polars import selectors as cs

from iter8ml.data.adapter import DataAdapter
from iter8ml.data.leakage import LeakageReport, detect_leakage

try:
    from hamilton.function_modifiers import config as _hamilton_config

    _HAS_HAMILTON = True
except ImportError:
    _HAS_HAMILTON = False


@dataclass
class DataPrepResult:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    leakage_report: LeakageReport | None
    target_transformer: Any | None
    n_rows: int
    n_features: int
    target_transform_method: str
    target_original_skewness: float
    target_transformed_skewness: float
    target_transform_applied: bool
    noise_cleaned: bool
    n_noise_dropped: int


# ── preprocessing node layer ────────────────────────────────────────────


def raw_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df


def numeric_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return raw_dataframe.select(cs.numeric()).columns


def categorical_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return raw_dataframe.select(cs.categorical() | cs.string()).columns


def date_columns(raw_dataframe: pl.DataFrame) -> list[str]:
    return [c for c, dtype in raw_dataframe.schema.items() if dtype in (pl.Datetime, pl.Date)]


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


def null_filled_df(
    fill_nulls_numeric: pl.DataFrame,
    fill_nulls_categorical: pl.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pl.DataFrame:
    numeric_part = fill_nulls_numeric.select([pl.col(c) for c in numeric_columns])
    cat_part = fill_nulls_categorical.select([pl.col(c) for c in categorical_columns])
    other_cols = [
        c
        for c in fill_nulls_numeric.columns
        if c not in numeric_columns and c not in categorical_columns
    ]
    other_part = fill_nulls_numeric.select(other_cols) if other_cols else pl.DataFrame()
    parts = [p for p in [other_part, numeric_part, cat_part] if p.width > 0]
    return pl.concat(parts, how="horizontal") if parts else fill_nulls_numeric


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
    result = null_filled_df.with_columns(exprs) if exprs else null_filled_df
    return result.drop(date_columns) if date_columns else result


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


# ── data preparation node layer ──────────────────────────────────────────


def validate_target(
    processed_dataframe: pl.DataFrame,
    target_col: str,
) -> pl.DataFrame:
    if target_col not in processed_dataframe.columns:
        raise ValueError(
            f"target_col '{target_col}' not found in DataFrame. "
            f"Available columns: {processed_dataframe.columns}"
        )
    return processed_dataframe


if _HAS_HAMILTON:

    @_hamilton_config.when(run_quality_audit=True)
    def quality_cleaned_df__audit(
        validate_target: pl.DataFrame,
        target_col: str,
        auto_clean_noise: bool,
        noise_quality_threshold: float,
    ) -> tuple[pl.DataFrame, bool, int]:
        if not auto_clean_noise:
            return validate_target, False, 0

        from iter8ml.data.quality import audit_data_quality, clean_noise

        quality_report = audit_data_quality(validate_target, target_col, enabled=True)
        if not quality_report.get("enabled") or quality_report.get("n_issues", 0) == 0:
            return validate_target, False, 0

        cleaned_df, summary = clean_noise(
            validate_target, quality_report, target_col, quality_threshold=noise_quality_threshold
        )
        return cleaned_df, True, summary.get("n_dropped", 0)

    @_hamilton_config.when_not(run_quality_audit=True)
    def quality_cleaned_df__skip(
        validate_target: pl.DataFrame,
    ) -> tuple[pl.DataFrame, bool, int]:
        return validate_target, False, 0

    @_hamilton_config.when(run_leakage_audit=True)
    def leakage_report__enabled(
        adapter_result: tuple[np.ndarray, np.ndarray],
        task: str,
        leakage_n_jobs: int,
    ) -> LeakageReport | None:
        X, y = adapter_result
        return detect_leakage(X, y, task=task, n_jobs=leakage_n_jobs)

    @_hamilton_config.when_not(run_leakage_audit=True)
    def leakage_report__skip(
        adapter_result: tuple[np.ndarray, np.ndarray],
        task: str,
    ) -> LeakageReport | None:
        return None

    @_hamilton_config.when(target_transform="none")
    def target_transform_result__none(
        adapter_result: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, Any | None, str, float, float, bool]:
        _, y = adapter_result
        return y, None, "none", 0.0, 0.0, False

    @_hamilton_config.when_not(target_transform="none")
    def target_transform_result__transform(
        adapter_result: tuple[np.ndarray, np.ndarray],
        target_transform: str,
        target_skewness_threshold: float,
    ) -> tuple[np.ndarray, Any | None, str, float, float, bool]:
        from iter8ml.data.features import transform_target

        _, y_raw = adapter_result
        y, transform_result, transformer = transform_target(
            y_raw,
            method=target_transform,  # type: ignore[arg-type]
            skewness_threshold=target_skewness_threshold,
        )
        return (
            y,
            transformer,
            transform_result.method,
            transform_result.original_skewness,
            transform_result.transformed_skewness,
            transform_result.applied,
        )

else:

    def quality_cleaned_df(
        validate_target: pl.DataFrame,
        target_col: str,
        run_quality_audit: bool,
        auto_clean_noise: bool,
        noise_quality_threshold: float,
    ) -> tuple[pl.DataFrame, bool, int]:
        if not (run_quality_audit and auto_clean_noise):
            return validate_target, False, 0

        from iter8ml.data.quality import audit_data_quality, clean_noise

        quality_report = audit_data_quality(validate_target, target_col, enabled=True)
        if not quality_report.get("enabled") or quality_report.get("n_issues", 0) == 0:
            return validate_target, False, 0

        cleaned_df, summary = clean_noise(
            validate_target, quality_report, target_col, quality_threshold=noise_quality_threshold
        )
        return cleaned_df, True, summary.get("n_dropped", 0)

    def leakage_report(
        adapter_result: tuple[np.ndarray, np.ndarray],
        run_leakage_audit: bool,
        task: str,
        leakage_n_jobs: int,
    ) -> LeakageReport | None:
        if not run_leakage_audit:
            return None
        X, y = adapter_result
        return detect_leakage(X, y, task=task, n_jobs=leakage_n_jobs)

    def target_transform_result(
        adapter_result: tuple[np.ndarray, np.ndarray],
        target_transform: str,
        target_skewness_threshold: float,
    ) -> tuple[np.ndarray, Any | None, str, float, float, bool]:
        from iter8ml.data.features import transform_target

        _, y_raw = adapter_result
        y, transform_result, transformer = transform_target(
            y_raw,
            method=target_transform,  # type: ignore[arg-type]
            skewness_threshold=target_skewness_threshold,
        )
        return (
            y,
            transformer,
            transform_result.method,
            transform_result.original_skewness,
            transform_result.transformed_skewness,
            transform_result.applied,
        )


def adapter_result(
    quality_cleaned_df: tuple[pl.DataFrame, bool, int],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = quality_cleaned_df[0]
    X, y = DataAdapter().transform(df, target_col)
    return X, y


def feature_names(
    quality_cleaned_df: tuple[pl.DataFrame, bool, int],
    target_col: str,
) -> list[str]:
    return [c for c in quality_cleaned_df[0].columns if c != target_col]


def data_prep_result(
    adapter_result: tuple[np.ndarray, np.ndarray],
    target_transform_result: tuple[np.ndarray, Any | None, str, float, float, bool],
    feature_names: list[str],
    leakage_report: LeakageReport | None,
    quality_cleaned_df: tuple[pl.DataFrame, bool, int],
) -> DataPrepResult:
    X, _ = adapter_result
    y, transformer, method, orig_skew, trans_skew, applied = target_transform_result
    _, noise_cleaned, n_dropped = quality_cleaned_df
    df = quality_cleaned_df[0]
    return DataPrepResult(
        X=X,
        y=y,
        feature_names=feature_names,
        leakage_report=leakage_report,
        target_transformer=transformer,
        n_rows=len(df),
        n_features=len(feature_names),
        target_transform_method=method,
        target_original_skewness=orig_skew,
        target_transformed_skewness=trans_skew,
        target_transform_applied=applied,
        noise_cleaned=noise_cleaned,
        n_noise_dropped=n_dropped,
    )
