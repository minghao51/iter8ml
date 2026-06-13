from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from iter8ml.engine.pipelines.nodes._hamilton_compat import hamilton_config

_hamilton_config = hamilton_config()


@dataclass
class DriftNodeResult:
    drift_detected: bool
    psi_report: Any | None
    domain_report: Any | None


def reference_df_input(reference_df: pl.DataFrame) -> pl.DataFrame:
    return reference_df


def live_df_input(live_df: pl.DataFrame) -> pl.DataFrame:
    return live_df


def _numeric_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype.is_numeric()]


def reference_features(reference_df_input: pl.DataFrame) -> pl.DataFrame:
    cols = _numeric_feature_cols(reference_df_input)
    return reference_df_input.select(cols) if cols else reference_df_input


def live_features(live_df_input: pl.DataFrame) -> pl.DataFrame:
    cols = _numeric_feature_cols(live_df_input)
    return live_df_input.select(cols) if cols else live_df_input


if _hamilton_config is not None:

    @_hamilton_config.when(drift_method="psi")
    def psi_drift_report__psi(
        reference_features: pl.DataFrame,
        live_features: pl.DataFrame,
    ) -> Any:
        from iter8ml.analysis.psi import PSIDriftDetector

        detector = PSIDriftDetector(reference_features)
        return detector.detect(live_features)

    @_hamilton_config.when(drift_method="domain_classifier")
    def domain_drift_report__domain(
        reference_features: pl.DataFrame,
        live_features: pl.DataFrame,
    ) -> Any:
        from iter8ml.analysis.domain_classifier import (
            DomainClassifierDriftDetector,
        )

        detector = DomainClassifierDriftDetector(reference_features)
        return detector.detect(live_features)

    @_hamilton_config.when(drift_method="psi")
    def drift_report__psi(
        psi_drift_report: Any,
    ) -> DriftNodeResult:
        return DriftNodeResult(
            drift_detected=psi_drift_report.drift_detected,
            psi_report=psi_drift_report,
            domain_report=None,
        )

    @_hamilton_config.when(drift_method="domain_classifier")
    def drift_report__domain(
        domain_drift_report: Any,
    ) -> DriftNodeResult:
        return DriftNodeResult(
            drift_detected=domain_drift_report.drift_detected,
            psi_report=None,
            domain_report=domain_drift_report,
        )

    @_hamilton_config.when(drift_method="both")
    def psi_drift_report__both(
        reference_features: pl.DataFrame,
        live_features: pl.DataFrame,
    ) -> Any:
        from iter8ml.analysis.psi import PSIDriftDetector

        detector = PSIDriftDetector(reference_features)
        return detector.detect(live_features)

    @_hamilton_config.when(drift_method="both")
    def domain_drift_report__both(
        reference_features: pl.DataFrame,
        live_features: pl.DataFrame,
    ) -> Any:
        from iter8ml.analysis.domain_classifier import (
            DomainClassifierDriftDetector,
        )

        detector = DomainClassifierDriftDetector(reference_features)
        return detector.detect(live_features)

    @_hamilton_config.when(drift_method="both")
    def drift_report__both(
        psi_drift_report: Any,
        domain_drift_report: Any,
    ) -> DriftNodeResult:
        return DriftNodeResult(
            drift_detected=psi_drift_report.drift_detected or domain_drift_report.drift_detected,
            psi_report=psi_drift_report,
            domain_report=domain_drift_report,
        )

else:
    from iter8ml.engine.pipelines.nodes._hamilton_compat import hamilton_stub

    psi_drift_report__psi = hamilton_stub("drift detection")
    domain_drift_report__domain = hamilton_stub("drift detection")
    drift_report__psi = hamilton_stub("drift detection")
    drift_report__domain = hamilton_stub("drift detection")
    psi_drift_report__both = hamilton_stub("drift detection")
    domain_drift_report__both = hamilton_stub("drift detection")
    drift_report__both = hamilton_stub("drift detection")
