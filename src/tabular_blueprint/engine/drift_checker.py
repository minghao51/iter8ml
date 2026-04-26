"""Drift detection: compares reference vs live data distributions."""

import polars as pl

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.engine.tracker import Tracker


class DriftChecker:
    def __init__(self, config: ExperimentConfig, tracker: Tracker):
        self.config = config
        self.tracker = tracker

    def check(self, df: pl.DataFrame, run_id: str) -> None:
        n_rows = len(df)
        split_idx = int(n_rows * 0.8)
        reference_df = df.slice(0, split_idx)
        live_df = df.slice(split_idx, n_rows - split_idx)

        feature_cols = [c for c in df.columns if c != self.config.target_col]
        ref_features = reference_df.select(feature_cols)
        live_features = live_df.select(feature_cols)

        drift_method = self.config.drift_detection

        if drift_method in ("psi", "both"):
            from tabular_blueprint.monitoring.psi_drift import PSIDriftDetector

            psi_detector = PSIDriftDetector(ref_features)
            psi_report = psi_detector.detect(live_features)
            self.tracker.log_event(
                {
                    "event": "drift_check",
                    "run_id": run_id,
                    "method": "psi",
                    "drift_detected": psi_report.drift_detected,
                    "n_features_tested": psi_report.n_features_tested,
                    "n_moderate": psi_report.n_moderate,
                    "n_severe": psi_report.n_severe,
                }
            )

        if drift_method in ("domain_classifier", "both"):
            from tabular_blueprint.monitoring.domain_classifier import (
                DomainClassifierDriftDetector,
            )

            domain_detector = DomainClassifierDriftDetector(ref_features)
            domain_report = domain_detector.detect(live_features)
            self.tracker.log_event(
                {
                    "event": "drift_check",
                    "run_id": run_id,
                    "method": "domain_classifier",
                    "drift_detected": domain_report.drift_detected,
                    "auc_score": domain_report.auc_score,
                    "threshold": domain_report.threshold,
                }
            )
