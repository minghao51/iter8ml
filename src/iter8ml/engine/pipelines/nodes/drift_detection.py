from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import polars as pl

from iter8ml.engine.pipelines.nodes._hamilton_compat import hamilton_config

_hamilton_config = hamilton_config()


@dataclass
class DriftNodeResult:
    drift_detected: bool
    psi_report: Any | None = None
    domain_report: Any | None = None
    ks_report: Any | None = None


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


# Single source of truth for the supported detectors. The detector body is
# identical regardless of which drift_method value activates it; only the
# Hamilton @when predicate and the result slot differ.
_DETECTOR_REGISTRY: dict[str, tuple[str, str, str]] = {
    "psi": ("iter8ml.analysis.psi", "PSIDriftDetector", "psi_drift_report"),
    "domain_classifier": (
        "iter8ml.analysis.domain_classifier",
        "DomainClassifierDriftDetector",
        "domain_drift_report",
    ),
    "ks": ("iter8ml.analysis.drift", "DriftDetector", "ks_drift_report"),
}

# Detector nodes active under a given drift_method value. "both" reuses the psi
# and domain_classifier detectors, each with its own @when(both) variant.
_DETECTOR_NODES: list[tuple[str, str, str]] = [
    ("psi", "psi", "psi"),
    ("domain_classifier", "domain_classifier", "domain"),
    ("ks", "ks", "ks"),
    ("both", "psi", "both"),
    ("both", "domain_classifier", "both"),
]


def _run_detector(key: str, reference_features: pl.DataFrame, live_features: pl.DataFrame) -> Any:
    mod_name, cls_name, _ = _DETECTOR_REGISTRY[key]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)(reference_features).detect(live_features)


if _hamilton_config is not None:
    # Generate one thin detector node per (config_value, detector) pair. Each
    # calls the shared _run_detector helper, removing the prior 6 near-duplicate
    # node functions.
    def _register_detector_node(cfg: str, key: str, suffix: str) -> None:
        basename = _DETECTOR_REGISTRY[key][2]

        @_hamilton_config.when(drift_method=cfg)
        def _node(reference_features: pl.DataFrame, live_features: pl.DataFrame) -> Any:
            return _run_detector(key, reference_features, live_features)

        _node.__name__ = f"{basename}__{suffix}"
        globals()[_node.__name__] = _node

    for _cfg, _key, _suffix in _DETECTOR_NODES:
        _register_detector_node(_cfg, _key, _suffix)

    @_hamilton_config.when(drift_method="psi")
    def drift_report__psi(psi_drift_report: Any) -> DriftNodeResult:
        return DriftNodeResult(
            drift_detected=psi_drift_report.drift_detected,
            psi_report=psi_drift_report,
        )

    @_hamilton_config.when(drift_method="domain_classifier")
    def drift_report__domain(domain_drift_report: Any) -> DriftNodeResult:
        return DriftNodeResult(
            drift_detected=domain_drift_report.drift_detected,
            domain_report=domain_drift_report,
        )

    @_hamilton_config.when(drift_method="ks")
    def drift_report__ks(ks_drift_report: Any) -> DriftNodeResult:
        return DriftNodeResult(
            drift_detected=ks_drift_report.drift_detected,
            ks_report=ks_drift_report,
        )

    @_hamilton_config.when(drift_method="both")
    def drift_report__both(
        psi_drift_report: Any,
        domain_drift_report: Any,
    ) -> DriftNodeResult:
        return DriftNodeResult(
            drift_detected=psi_drift_report.drift_detected
            or domain_drift_report.drift_detected,
            psi_report=psi_drift_report,
            domain_report=domain_drift_report,
        )

else:
    from iter8ml.engine.pipelines.nodes._hamilton_compat import hamilton_stub

    psi_drift_report__psi = hamilton_stub("drift detection")
    domain_drift_report__domain = hamilton_stub("drift detection")
    ks_drift_report__ks = hamilton_stub("drift detection")
    drift_report__psi = hamilton_stub("drift detection")
    drift_report__domain = hamilton_stub("drift detection")
    drift_report__ks = hamilton_stub("drift detection")
    psi_drift_report__both = hamilton_stub("drift detection")
    domain_drift_report__both = hamilton_stub("drift detection")
    drift_report__both = hamilton_stub("drift detection")
