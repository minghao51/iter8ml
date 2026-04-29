"""Data preparation: noise cleaning, leakage detection, target transformation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.data.adapter import DataAdapter
from tabular_blueprint.data.feature_engine import transform_target
from tabular_blueprint.data.leakage import LeakageReport, detect_leakage
from tabular_blueprint.engine.tracker import Tracker
from tabular_blueprint.pipelines.executor import PipelineExecutor


@dataclass
class DataPrepResult:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    leakage_report: LeakageReport | None
    target_transformer: object | None
    n_rows: int
    n_features: int


class DataPreparationService:
    def __init__(self, config: ExperimentConfig, tracker: Tracker):
        self.config = config
        self.tracker = tracker

    def prepare(
        self,
        df: pl.DataFrame,
        run_id: str,
        run_leakage_audit: bool = True,
    ) -> DataPrepResult:
        executor = PipelineExecutor(tracker=self.tracker)
        if executor.available:
            return self._prepare_via_hamilton(df, run_id, run_leakage_audit, executor)
        return self._prepare_imperative(df, run_id, run_leakage_audit)

    def _prepare_via_hamilton(
        self,
        df: pl.DataFrame,
        run_id: str,
        run_leakage_audit: bool,
        executor: PipelineExecutor,
    ) -> DataPrepResult:
        result = executor.run_data_prep(
            df=df,
            target_col=self.config.target_col,
            task=self.config.task.value,
            run_id=run_id,
            run_quality_audit=self.config.run_quality_audit,
            auto_clean_noise=self.config.auto_clean_noise,
            noise_quality_threshold=self.config.noise_quality_threshold,
            run_leakage_audit=run_leakage_audit,
            target_transform=self.config.target_transform,
            target_skewness_threshold=self.config.target_skewness_threshold,
        )
        if result is None:
            return self._prepare_imperative(df, run_id, run_leakage_audit)

        return DataPrepResult(
            X=result.X,
            y=result.y,
            feature_names=result.feature_names,
            leakage_report=result.leakage_report,
            target_transformer=result.target_transformer,
            n_rows=result.n_rows,
            n_features=result.n_features,
        )

    def _prepare_imperative(
        self,
        df: pl.DataFrame,
        run_id: str,
        run_leakage_audit: bool,
    ) -> DataPrepResult:
        if self.config.target_col not in df.columns:
            raise ValueError(
                f"target_col '{self.config.target_col}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        if self.config.auto_clean_noise and self.config.run_quality_audit:
            from tabular_blueprint.data.quality import audit_data_quality, clean_noise

            quality_report = audit_data_quality(df, self.config.target_col, enabled=True)
            if quality_report.get("enabled") and quality_report.get("n_issues", 0) > 0:
                df, clean_summary = clean_noise(
                    df,
                    quality_report,
                    self.config.target_col,
                    quality_threshold=self.config.noise_quality_threshold,
                )
                self.tracker.log_event(
                    {
                        "event": "noise_cleaned",
                        "run_id": run_id,
                        "n_before": clean_summary["n_before"],
                        "n_after": clean_summary["n_after"],
                        "n_dropped": clean_summary["n_dropped"],
                        "threshold": clean_summary["threshold"],
                    }
                )

        adapter = DataAdapter(target_format="numpy")
        X, y = adapter.transform(df, self.config.target_col)
        feature_names = [c for c in df.columns if c != self.config.target_col]

        leakage_report: LeakageReport | None = None
        if run_leakage_audit:
            leakage_report = detect_leakage(X, y, task=self.config.task.value)
            if leakage_report.n_flagged > 0:
                self.tracker.log_event(
                    {
                        "event": "leakage_audit",
                        "n_flagged": leakage_report.n_flagged,
                        "flagged_features": leakage_report.flagged_features,
                        "baseline_score": leakage_report.baseline_score,
                    }
                )

        target_transformer = None
        if self.config.target_transform != "none":
            y, transform_result, target_transformer = transform_target(
                y,
                method=self.config.target_transform,
                skewness_threshold=self.config.target_skewness_threshold,
            )
            self.tracker.log_event(
                {
                    "event": "target_transform",
                    "method": transform_result.method,
                    "original_skewness": transform_result.original_skewness,
                    "transformed_skewness": transform_result.transformed_skewness,
                    "applied": transform_result.applied,
                }
            )

        return DataPrepResult(
            X=X,
            y=y,
            feature_names=feature_names,
            leakage_report=leakage_report,
            target_transformer=target_transformer,
            n_rows=len(df),
            n_features=len(feature_names),
        )
