from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from tabular_blueprint.data.adapter import DataAdapter
from tabular_blueprint.data.leakage import LeakageReport, detect_leakage


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


def quality_cleaned_df(
    validate_target: pl.DataFrame,
    target_col: str,
    run_quality_audit: bool,
    auto_clean_noise: bool,
    noise_quality_threshold: float,
) -> tuple[pl.DataFrame, bool, int]:
    if not (run_quality_audit and auto_clean_noise):
        return validate_target, False, 0

    from tabular_blueprint.data.quality import audit_data_quality, clean_noise

    quality_report = audit_data_quality(validate_target, target_col, enabled=True)
    if not quality_report.get("enabled") or quality_report.get("n_issues", 0) == 0:
        return validate_target, False, 0

    cleaned_df, summary = clean_noise(
        validate_target,
        quality_report,
        target_col,
        quality_threshold=noise_quality_threshold,
    )
    return cleaned_df, True, summary.get("n_dropped", 0)


def adapter_result(
    quality_cleaned_df: tuple[pl.DataFrame, bool, int],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = quality_cleaned_df[0]
    adapter = DataAdapter()
    X, y = adapter.transform(df, target_col)
    return X, y


def feature_names(
    quality_cleaned_df: tuple[pl.DataFrame, bool, int],
    target_col: str,
) -> list[str]:
    df = quality_cleaned_df[0]
    return [c for c in df.columns if c != target_col]


def leakage_report(
    adapter_result: tuple[np.ndarray, np.ndarray],
    run_leakage_audit: bool,
    task: str,
) -> LeakageReport | None:
    if not run_leakage_audit:
        return None
    X, y = adapter_result
    return detect_leakage(X, y, task=task)


def target_transform_result(
    adapter_result: tuple[np.ndarray, np.ndarray],
    target_transform: str,
    target_skewness_threshold: float,
) -> tuple[np.ndarray, Any | None, str, float, float, bool]:
    from tabular_blueprint.data.feature_engine import transform_target

    _, y_raw = adapter_result
    y, transform_result, transformer = transform_target(
        y_raw,
        method=target_transform,
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
