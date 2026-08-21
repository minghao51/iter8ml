"""Population Stability Index (PSI) drift detection."""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel

from iter8ml.analysis._protocol import DriftReportBase


class FeaturePSI(BaseModel):
    """PSI drift score for a single feature."""

    feature: str
    psi_value: float
    drift_level: str  # "none", "moderate", "severe"


class PSIDriftReport(DriftReportBase):
    """Aggregate PSI drift report across all numeric features."""

    method: str = "psi"
    n_features_tested: int
    n_moderate: int
    n_severe: int
    feature_psi: list[FeaturePSI]


MODERATE_THRESHOLD = 0.20
SEVERE_THRESHOLD = 0.30


def compute_psi(reference: np.ndarray, live: np.ndarray, n_bins: int = 10) -> float:
    ref_non_null = reference[~np.isnan(reference)]
    live_non_null = live[~np.isnan(live)]

    if len(ref_non_null) == 0 or len(live_non_null) == 0:
        return 0.0

    all_values = np.concatenate([ref_non_null, live_non_null])
    bin_edges = np.percentile(all_values, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(ref_non_null, bins=bin_edges)[0].astype(float)
    live_counts = np.histogram(live_non_null, bins=bin_edges)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum()
    live_pct = live_counts / live_counts.sum()

    ref_pct = np.clip(ref_pct, 1e-6, None)
    live_pct = np.clip(live_pct, 1e-6, None)

    psi = float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))
    return psi


def classify_drift(psi_value: float) -> str:
    if psi_value > SEVERE_THRESHOLD:
        return "severe"
    elif psi_value > MODERATE_THRESHOLD:
        return "moderate"
    return "none"


class PSIDriftDetector:
    def __init__(self, reference_df: pl.DataFrame, n_bins: int = 10):
        self.reference_df = reference_df
        self.n_bins = n_bins

    def detect(self, live_df: pl.DataFrame) -> PSIDriftReport:
        common_cols = sorted(set(self.reference_df.columns) & set(live_df.columns))
        feature_psi_results: list[FeaturePSI] = []

        for col in common_cols:
            dtype = self.reference_df[col].dtype
            if not dtype.is_numeric():
                continue

            ref_values = self.reference_df[col].drop_nulls().to_numpy()
            live_values = live_df[col].drop_nulls().to_numpy()

            psi_value = compute_psi(ref_values, live_values, self.n_bins)
            drift_level = classify_drift(psi_value)

            feature_psi_results.append(
                FeaturePSI(feature=col, psi_value=round(psi_value, 6), drift_level=drift_level)
            )

        n_moderate = sum(1 for f in feature_psi_results if f.drift_level == "moderate")
        n_severe = sum(1 for f in feature_psi_results if f.drift_level == "severe")

        return PSIDriftReport(
            drift_detected=(n_moderate + n_severe) > 0,
            n_features_tested=len(feature_psi_results),
            n_moderate=n_moderate,
            n_severe=n_severe,
            feature_psi=feature_psi_results,
        )
