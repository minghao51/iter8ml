from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from tabular_blueprint.config import ExperimentConfig

_DIRECT_FIELDS: tuple[str, ...] = (
    "target_col",
    "cv_folds",
    "metrics",
    "calibration",
    "afe_top_k",
    "afe_lift_threshold",
    "afe_pruning",
    "afe_prune_min_importance",
    "random_seed",
    "run_quality_audit",
    "auto_clean_noise",
    "noise_quality_threshold",
    "target_transform",
    "target_skewness_threshold",
    "embedding_dim",
    "embedding_max_categories",
    "embedding_epochs",
    "embedding_lr",
    "embedding_mlp_width",
    "embedding_mlp_depth",
    "embedding_ae_latent_dim",
    "embedding_ae_dropout",
    "model_overrides",
)


class PipelineMode(StrEnum):
    TRAINING = "training"
    DRIFT = "drift"
    EXPORT = "export"
    HPO = "hpo"
    INFERENCE = "inference"


def _get_module(mode: PipelineMode) -> Any:
    from tabular_blueprint.pipelines.nodes import preprocessing

    if mode == PipelineMode.DRIFT:
        from tabular_blueprint.pipelines.nodes import drift_detection

        return [preprocessing, drift_detection]
    return [preprocessing]


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


def _config_to_inputs(
    config: ExperimentConfig,
    df: pl.DataFrame,
    run_id: str,
    vram_gb: float,
    run_leakage_audit: bool,
    completed_models: set[str] | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "df": df,
        "run_id": run_id,
        "vram_gb": vram_gb,
        "run_leakage_audit": run_leakage_audit,
        "task": config.task.value,
        "config_models": config.models,
        "experiment_name": config.name,
        "cv_strategy": config.cv_strategy.value,
        "workspace_dir": str(config.workspace_dir),
        "embedding_method": config.embedding_method.value,
        "completed_models": sorted(completed_models or set()),
    }
    for field in _DIRECT_FIELDS:
        inputs[field] = getattr(config, field)
    return inputs


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
            modules = _get_module(mode)
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
        return self._dr.execute(targets, inputs=inputs, overrides=overrides)  # type: ignore[no-any-return]

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
        return result.get("processed_dataframe", df)  # type: ignore[no-any-return]

    def run_training(
        self,
        config: ExperimentConfig,
        df: pl.DataFrame,
        run_id: str,
        vram_gb: float = 0.0,
        run_leakage_audit: bool = True,
        completed_models: set[str] | None = None,
    ) -> Any:
        if self._driver_mod is None:
            return None

        modules = _get_training_modules()
        builder = self._driver_mod.Builder().with_modules(*modules)
        hamilton_config: dict[str, Any] = {"afe_enabled": config.afe_enabled}
        if config.embedding_enabled:
            hamilton_config["embedding_enabled"] = True
        builder = builder.with_config(hamilton_config)
        if self._tracker is not None:
            from tabular_blueprint.pipelines.hooks.tracking_hook import TrackingHook

            hook = TrackingHook(self._tracker, run_id)
            builder = builder.with_adapters(hook)
        dr = builder.build()

        inputs = _config_to_inputs(
            config,
            df,
            run_id,
            vram_gb,
            run_leakage_audit,
            completed_models=completed_models,
        )
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
