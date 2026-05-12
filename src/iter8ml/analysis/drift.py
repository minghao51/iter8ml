"""Drift detection using statistical tests."""

import polars as pl
from pydantic import BaseModel
from scipy import stats


class ColumnDriftResult(BaseModel):
    """Per-column statistical test result."""

    column: str
    p_value: float
    drift_detected: bool
    test_used: str


class DriftReport(BaseModel):
    """Aggregate drift detection report across all columns."""

    drift_detected: bool
    n_columns_tested: int
    n_drifted: int
    column_results: list[ColumnDriftResult]


class DriftDetector:
    """
    Compares a reference DataFrame (training) against a new DataFrame (production).

    Numeric columns: Kolmogorov-Smirnov test
    Categorical cols: Chi-squared test

    Returns a DriftReport with per-column p-values and a global drift flag.
    """

    def __init__(self, reference_df: pl.DataFrame, alpha: float = 0.05):
        self.reference_df = reference_df
        self.alpha = alpha

    def detect(self, new_df: pl.DataFrame) -> DriftReport:
        common_cols = set(self.reference_df.columns) & set(new_df.columns)
        column_results = []

        for col in common_cols:
            dtype = self.reference_df[col].dtype
            if dtype.is_numeric():
                p_value = self._ks_test(col, new_df)
                test_used = "ks_test"
            else:
                p_value = self._chi2_test(col, new_df)
                test_used = "chi2_test"

            column_results.append(
                ColumnDriftResult(
                    column=col,
                    p_value=round(p_value, 6),
                    drift_detected=p_value < self.alpha,
                    test_used=test_used,
                )
            )

        n_drifted = sum(1 for r in column_results if r.drift_detected)
        return DriftReport(
            drift_detected=n_drifted > 0,
            n_columns_tested=len(column_results),
            n_drifted=n_drifted,
            column_results=column_results,
        )

    def _ks_test(self, col: str, new_df: pl.DataFrame) -> float:
        ref_series = self.reference_df[col].drop_nulls()
        new_series = new_df[col].drop_nulls()

        # Handle float NaNs while keeping Decimal/object numeric support.
        if ref_series.dtype.is_float():
            ref_series = ref_series.filter(~ref_series.is_nan())
        if new_series.dtype.is_float():
            new_series = new_series.filter(~new_series.is_nan())

        if len(ref_series) == 0 or len(new_series) == 0:
            return 1.0

        _, p_value = stats.ks_2samp(ref_series.to_numpy(), new_series.to_numpy())
        return p_value  # type: ignore[no-any-return]

    def _chi2_test(self, col: str, new_df: pl.DataFrame) -> float:
        ref_counts = self.reference_df[col].drop_nulls().value_counts()
        new_counts = new_df[col].drop_nulls().value_counts()

        ref_dict = dict(zip(ref_counts[:, 0].to_list(), ref_counts[:, 1].to_list(), strict=False))
        new_dict = dict(zip(new_counts[:, 0].to_list(), new_counts[:, 1].to_list(), strict=False))

        all_categories = set(ref_dict.keys()) | set(new_dict.keys())
        ref_observed = [ref_dict.get(cat, 0) for cat in all_categories]
        new_observed = [new_dict.get(cat, 0) for cat in all_categories]

        if sum(ref_observed) == 0 or sum(new_observed) == 0:
            return 1.0

        _, p_value, _, _ = stats.chi2_contingency([ref_observed, new_observed])
        return p_value  # type: ignore[no-any-return]
