from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from polars import selectors as cs

from iter8ml.data.adapter import DataAdapter
from iter8ml.data.leakage import LeakageReport, detect_leakage
from iter8ml.domain.hashing import row_ids as frame_row_ids
from iter8ml.engine.pipelines.nodes._hamilton_compat import hamilton_config

_hamilton_config = hamilton_config()


@dataclass
class DataPrepResult:
    dataframe: pl.DataFrame
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
    row_ids: list[str]


# ── preprocessing node layer ────────────────────────────────────────────


def raw_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df


def filtered_dataframe(
    raw_dataframe: pl.DataFrame,
    target_col: str | None = None,
    ignore_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Drop user-excluded columns (IDs, leaky features) before feature engineering.

    Applied *after* ``row_ids`` is computed on the raw frame so row digests stay
    aligned with any medallion-layer split assignment. Unknown columns or a
    target listed in ``ignore_cols`` fail loudly here instead of mid-run.
    """
    cols = ignore_cols or []
    if not cols:
        return raw_dataframe
    unknown = [c for c in cols if c not in raw_dataframe.columns]
    if unknown:
        raise ValueError(
            f"ignore_cols not found in DataFrame: {unknown}. "
            f"Available columns: {raw_dataframe.columns}"
        )
    if target_col is not None and target_col in cols:
        raise ValueError(f"target_col '{target_col}' cannot also be listed in ignore_cols")
    return raw_dataframe.drop(cols)


def row_ids(raw_dataframe: pl.DataFrame) -> list[str]:
    """Stable per-row content digests, aligned to the engine's feature matrix.

    Computed from the raw frame (before ``ignore_cols`` filtering) so the
    training path can align its engineered rows to a split assigned on the same
    frame by the medallion layer.
    """
    return frame_row_ids(raw_dataframe)


def numeric_columns(filtered_dataframe: pl.DataFrame) -> list[str]:
    return filtered_dataframe.select(cs.numeric()).columns


def categorical_columns(filtered_dataframe: pl.DataFrame) -> list[str]:
    return filtered_dataframe.select(cs.categorical() | cs.string()).columns


def date_columns(filtered_dataframe: pl.DataFrame) -> list[str]:
    return [c for c, dtype in filtered_dataframe.schema.items() if dtype in (pl.Datetime, pl.Date)]


def fill_nulls_numeric(
    filtered_dataframe: pl.DataFrame,
    numeric_columns: list[str],
) -> pl.DataFrame:
    exprs = [pl.col(c).fill_null(pl.col(c).median()) for c in numeric_columns]
    return filtered_dataframe.with_columns(exprs) if exprs else filtered_dataframe


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


def target_oriented_df(
    decomposed_dates_df: pl.DataFrame,
    target_col: str | None = None,
    positive_class: str | float | bool | None = None,
) -> pl.DataFrame:
    """Orient a binary target so ``positive_class`` encodes to 1.

    Without ``positive_class`` a categorical/string target is encoded by
    Polars physical codes — appearance order, which is arbitrary w.r.t.
    semantics — so ``roc_auc`` orientation can amount to a coin flip for
    labels like "good"/"bad". With ``positive_class``, the positive class
    deterministically maps to 1 and the other class to 0, keeping
    ``predict_proba()[:, 1]`` aligned with the intended positive class.

    ``target_col`` defaults to None so modes executing the prep module with
    only ``df`` (no config inputs) keep working; orientation is a no-op
    unless both ``target_col`` and ``positive_class`` are supplied.
    """
    if positive_class is None:
        return decomposed_dates_df
    if target_col is None:
        raise ValueError("positive_class requires target_col to be provided")
    values = decomposed_dates_df[target_col].unique().to_list()
    observed = sorted(str(v) for v in values)
    if positive_class not in values:
        raise ValueError(
            f"positive_class {positive_class!r} not found in target column "
            f"'{target_col}'. Observed values: {observed}"
        )
    if len(values) != 2:
        raise ValueError(
            f"positive_class requires a binary target; '{target_col}' has "
            f"{len(values)} distinct values: {observed}"
        )
    return decomposed_dates_df.with_columns(
        pl.when(pl.col(target_col) == positive_class).then(1).otherwise(0).alias(target_col)
    )


def encoded_df(
    target_oriented_df: pl.DataFrame,
    categorical_columns: list[str],
) -> pl.DataFrame:
    cat_cols = [
        c
        for c in categorical_columns
        if c in target_oriented_df.columns
        # String/categorical columns only: an already-oriented integer target
        # (positive_class set) must not re-enter the categorical cast.
        and target_oriented_df.schema[c] in (pl.String, pl.Categorical)
    ]
    if not cat_cols:
        return target_oriented_df
    exprs = [pl.col(col).cast(pl.Categorical).to_physical().alias(col) for col in cat_cols]
    return target_oriented_df.with_columns(exprs)


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


if _hamilton_config is not None:

    @_hamilton_config.when(run_quality_audit=True)
    def quality_cleaned_df__audit(
        validate_target: pl.DataFrame,
        target_col: str,
        auto_clean_noise: bool,
        noise_quality_threshold: float,
        row_ids: list[str],
    ) -> tuple[pl.DataFrame, bool, int, list[str]]:
        if not auto_clean_noise:
            return validate_target, False, 0, row_ids

        from iter8ml.data.quality import audit_data_quality, clean_noise

        quality_report = audit_data_quality(validate_target, target_col, enabled=True)
        if not quality_report.get("enabled") or quality_report.get("n_issues", 0) == 0:
            return validate_target, False, 0, row_ids

        cleaned_df, summary = clean_noise(
            validate_target, quality_report, target_col, quality_threshold=noise_quality_threshold
        )
        kept = summary.get("kept_indices", list(range(len(row_ids))))
        kept_row_ids = [row_ids[i] for i in kept]
        return cleaned_df, True, summary.get("n_dropped", 0), kept_row_ids

    @_hamilton_config.when_not(run_quality_audit=True)
    def quality_cleaned_df__skip(
        validate_target: pl.DataFrame,
        row_ids: list[str],
    ) -> tuple[pl.DataFrame, bool, int, list[str]]:
        return validate_target, False, 0, row_ids

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
        row_ids: list[str],
    ) -> tuple[pl.DataFrame, bool, int, list[str]]:
        if not (run_quality_audit and auto_clean_noise):
            return validate_target, False, 0, row_ids

        from iter8ml.data.quality import audit_data_quality, clean_noise

        quality_report = audit_data_quality(validate_target, target_col, enabled=True)
        if not quality_report.get("enabled") or quality_report.get("n_issues", 0) == 0:
            return validate_target, False, 0, row_ids

        cleaned_df, summary = clean_noise(
            validate_target, quality_report, target_col, quality_threshold=noise_quality_threshold
        )
        kept = summary.get("kept_indices", list(range(len(row_ids))))
        kept_row_ids = [row_ids[i] for i in kept]
        return cleaned_df, True, summary.get("n_dropped", 0), kept_row_ids

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
    quality_cleaned_df: tuple[pl.DataFrame, bool, int, list[str]],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = quality_cleaned_df[0]
    X, y = DataAdapter().transform(df, target_col)
    return X, y


def feature_names(
    quality_cleaned_df: tuple[pl.DataFrame, bool, int, list[str]],
    target_col: str,
) -> list[str]:
    return [c for c in quality_cleaned_df[0].columns if c != target_col]


def data_prep_result(
    adapter_result: tuple[np.ndarray, np.ndarray],
    target_transform_result: tuple[np.ndarray, Any | None, str, float, float, bool],
    feature_names: list[str],
    leakage_report: LeakageReport | None,
    quality_cleaned_df: tuple[pl.DataFrame, bool, int, list[str]],
) -> DataPrepResult:
    X, _ = adapter_result
    y, transformer, method, orig_skew, trans_skew, applied = target_transform_result
    _, noise_cleaned, n_dropped, row_ids = quality_cleaned_df
    df = quality_cleaned_df[0]
    return DataPrepResult(
        dataframe=df,
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
        row_ids=row_ids,
    )
