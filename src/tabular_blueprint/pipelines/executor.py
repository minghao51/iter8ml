from __future__ import annotations

from enum import StrEnum
from typing import Any

import polars as pl


class PipelineMode(StrEnum):
    TRAINING = "training"
    DRIFT = "drift"
    EXPORT = "export"
    HPO = "hpo"
    INFERENCE = "inference"


def _get_module(mode: PipelineMode) -> Any:
    from tabular_blueprint.pipelines.nodes import preprocessing

    return preprocessing


def _get_data_prep_module() -> Any:
    from tabular_blueprint.pipelines.nodes import data_preparation

    return data_preparation


def _get_training_modules() -> list[Any]:
    from tabular_blueprint.pipelines.nodes import (
        baselines,
        data_preparation,
        feature_engineering,
        model_selection,
        model_training,
        preprocessing,
        state_generation,
    )

    return [
        preprocessing,
        data_preparation,
        model_selection,
        baselines,
        feature_engineering,
        model_training,
        state_generation,
    ]


_MODE_MODULES: dict[PipelineMode, list[Any]] = {
    PipelineMode.TRAINING: [],
    PipelineMode.DRIFT: [],
    PipelineMode.EXPORT: [],
    PipelineMode.HPO: [],
    PipelineMode.INFERENCE: [],
}

_MODE_FINAL_VARS: dict[PipelineMode, list[str]] = {
    PipelineMode.TRAINING: ["processed_dataframe"],
    PipelineMode.DRIFT: ["drift_report"],
    PipelineMode.EXPORT: ["processed_dataframe"],
    PipelineMode.HPO: ["processed_dataframe"],
    PipelineMode.INFERENCE: ["processed_dataframe"],
}


def _try_import_hamilton() -> Any:
    try:
        from hamilton import driver

        return driver
    except ImportError:
        return None


class PipelineExecutor:
    def __init__(
        self,
        mode: PipelineMode = PipelineMode.TRAINING,
        config: dict[str, Any] | None = None,
        tracker: Any | None = None,
    ) -> None:
        self._mode = mode
        self._config = config or {}
        self._driver_mod = _try_import_hamilton()
        self._dr: Any = None
        self._tracker = tracker

        if self._driver_mod is not None:
            preprocessing = _get_module(mode)
            extra = _MODE_MODULES.get(mode, [])
            modules = [preprocessing, *extra]
            builder = self._driver_mod.Builder().with_modules(*modules)
            if self._config:
                builder = builder.with_config(self._config)
            self._dr = builder.build()

    @property
    def available(self) -> bool:
        return self._dr is not None

    def execute(
        self,
        inputs: dict[str, Any],
        final_vars: list[str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._dr is None:
            return {}

        targets = final_vars or _MODE_FINAL_VARS.get(self._mode, ["processed_dataframe"])
        return self._dr.execute(targets, inputs=inputs, overrides=overrides)

    def get_mermaid_graph(self) -> str:
        if self._dr is None:
            return "graph TD\n    A[Raw Data] --> B[Processed Data]"
        result = self._dr.display_all_functions()
        if isinstance(result, str):
            return result
        return getattr(result, "source", str(result))

    def run_preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._dr is None:
            return df
        result = self.execute(inputs={"df": df})
        return result.get("processed_dataframe", df)

    def run_data_prep(
        self,
        df: pl.DataFrame,
        target_col: str,
        task: str,
        run_id: str | None = None,
        run_quality_audit: bool = True,
        auto_clean_noise: bool = False,
        noise_quality_threshold: float = 0.5,
        run_leakage_audit: bool = True,
        target_transform: str = "none",
        target_skewness_threshold: float = 1.0,
    ) -> Any:
        if self._dr is None:
            return None
        from tabular_blueprint.pipelines.nodes import data_preparation as dp_mod

        preprocessing = _get_module(self._mode)
        modules = [preprocessing, dp_mod]
        builder = self._driver_mod.Builder().with_modules(*modules)
        if self._tracker is not None:
            from tabular_blueprint.pipelines.hooks.tracking_hook import TrackingHook

            hook = TrackingHook(self._tracker, run_id)
            builder = builder.with_adapters(hook)
        dr = builder.build()
        result = dr.execute(
            ["data_prep_result"],
            inputs={
                "df": df,
                "target_col": target_col,
                "task": task,
                "run_quality_audit": run_quality_audit,
                "auto_clean_noise": auto_clean_noise,
                "noise_quality_threshold": noise_quality_threshold,
                "run_leakage_audit": run_leakage_audit,
                "target_transform": target_transform,
                "target_skewness_threshold": target_skewness_threshold,
            },
        )
        return result.get("data_prep_result")

    def run_training(
        self,
        df: pl.DataFrame,
        target_col: str,
        task: str,
        config_models: Any,
        experiment_name: str,
        run_id: str,
        workspace_dir: str,
        vram_gb: float = 0.0,
        cv_folds: int = 5,
        cv_strategy: str = "stratified",
        metrics: list[str] | None = None,
        calibration: str = "none",
        afe_enabled: bool = False,
        afe_top_k: int = 10,
        afe_lift_threshold: float = 0.01,
        afe_pruning: bool = False,
        afe_prune_min_importance: float = 0.001,
        random_seed: int = 42,
        run_quality_audit: bool = True,
        auto_clean_noise: bool = False,
        noise_quality_threshold: float = 0.5,
        run_leakage_audit: bool = True,
        target_transform: str = "none",
        target_skewness_threshold: float = 1.0,
        embedding_enabled: bool = False,
        embedding_method: str = "entity",
        embedding_dim: int = 16,
        embedding_max_categories: int = 50,
        embedding_epochs: int = 10,
        embedding_lr: float = 1e-3,
        embedding_mlp_width: int = 128,
        embedding_mlp_depth: int = 2,
        embedding_ae_latent_dim: int = 32,
        embedding_ae_dropout: float = 0.2,
    ) -> Any:
        if self._driver_mod is None:
            return None

        modules = _get_training_modules()
        builder = self._driver_mod.Builder().with_modules(*modules)
        hamilton_config: dict[str, Any] = {"afe_enabled": afe_enabled}
        if embedding_enabled:
            hamilton_config["embedding_enabled"] = True
        builder = builder.with_config(hamilton_config)
        if self._tracker is not None:
            from tabular_blueprint.pipelines.hooks.tracking_hook import TrackingHook

            hook = TrackingHook(self._tracker, run_id)
            builder = builder.with_adapters(hook)
        dr = builder.build()

        inputs = {
            "df": df,
            "target_col": target_col,
            "task": task,
            "config_models": config_models,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "workspace_dir": workspace_dir,
            "vram_gb": vram_gb,
            "cv_folds": cv_folds,
            "cv_strategy": cv_strategy,
            "metrics": metrics or ["roc_auc", "f1_macro"],
            "calibration": calibration,
            "afe_top_k": afe_top_k,
            "afe_lift_threshold": afe_lift_threshold,
            "afe_pruning": afe_pruning,
            "afe_prune_min_importance": afe_prune_min_importance,
            "random_seed": random_seed,
            "run_quality_audit": run_quality_audit,
            "auto_clean_noise": auto_clean_noise,
            "noise_quality_threshold": noise_quality_threshold,
            "run_leakage_audit": run_leakage_audit,
            "target_transform": target_transform,
            "target_skewness_threshold": target_skewness_threshold,
            "embedding_method": embedding_method,
            "embedding_dim": embedding_dim,
            "embedding_max_categories": embedding_max_categories,
            "embedding_epochs": embedding_epochs,
            "embedding_lr": embedding_lr,
            "embedding_mlp_width": embedding_mlp_width,
            "embedding_mlp_depth": embedding_mlp_depth,
            "embedding_ae_latent_dim": embedding_ae_latent_dim,
            "embedding_ae_dropout": embedding_ae_dropout,
        }
        result = dr.execute(["training_state"], inputs=inputs)
        return result.get("training_state")

    def run_drift(
        self,
        reference_df: pl.DataFrame,
        live_df: pl.DataFrame,
        drift_method: str = "psi",
    ) -> Any:
        if self._driver_mod is None:
            return None

        from tabular_blueprint.pipelines.nodes import drift_detection, preprocessing

        modules = [preprocessing, drift_detection]
        builder = self._driver_mod.Builder().with_modules(*modules)
        builder = builder.with_config({"drift_method": drift_method})
        dr = builder.build()
        result = dr.execute(
            ["drift_report"],
            inputs={"reference_df": reference_df, "live_df": live_df},
        )
        return result.get("drift_report")
