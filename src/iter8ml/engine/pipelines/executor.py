from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl

from iter8ml.config import PipelineSpec, StepName
from iter8ml.exceptions import HamiltonUnavailableError
from iter8ml.workspace import Workspace

if TYPE_CHECKING:
    from iter8ml.config import ExperimentConfig

_DIRECT_FIELDS: tuple[str, ...] = (
    "target_col",
    "cv_folds",
    "metrics",
    "afe_top_k",
    "afe_lift_threshold",
    "afe_pruning",
    "afe_prune_min_importance",
    "afe_n_jobs",
    "afe_max_candidate_pairs",
    "leakage_n_jobs",
    "random_seed",
    "embedding_dim",
    "embedding_max_categories",
    "embedding_epochs",
    "embedding_lr",
    "embedding_mlp_width",
    "embedding_mlp_depth",
    "embedding_ae_latent_dim",
    "embedding_ae_dropout",
    "model_overrides",
    "strict_thread_safety",
)


class PipelineMode(StrEnum):
    TRAINING = "training"
    DRIFT = "drift"
    EXPORT = "export"
    HPO = "hpo"
    INFERENCE = "inference"


def _get_module(mode: PipelineMode) -> Any:
    from iter8ml.engine.pipelines.nodes import prep

    if mode == PipelineMode.DRIFT:
        from iter8ml.engine.pipelines.nodes import drift_detection

        return [prep, drift_detection]
    return [prep]


def _get_training_modules(spec: PipelineSpec | None = None) -> list[Any]:
    from iter8ml.engine.pipelines.nodes import prep, train

    modules = [prep]
    if spec is None or spec.is_enabled(StepName.FEATURE_ENGINEERING):
        from iter8ml.engine.pipelines.nodes import features

        modules.append(features)
    modules.append(train)
    return modules


def _resolve_hamilton_config(config: ExperimentConfig) -> dict[str, Any]:
    spec = config.pipeline
    quality_params = spec.step_params(StepName.QUALITY_AUDIT)
    transform_params = spec.step_params(StepName.TARGET_TRANSFORM)
    cal_params = spec.step_params(StepName.CALIBRATION)
    feat_params = spec.step_params(StepName.FEATURE_ENGINEERING)
    cfg: dict[str, Any] = {
        "run_quality_audit": spec.is_enabled(StepName.QUALITY_AUDIT),
        "run_leakage_audit": spec.is_enabled(StepName.LEAKAGE_AUDIT),
        "target_transform": transform_params.get("method", "none"),
        "target_skewness_threshold": transform_params.get("skewness_threshold", 1.0),
        "calibration": cal_params.get("method", "none"),
        "feature_strategy": feat_params.get("strategy", "none"),
        "auto_clean_noise": quality_params.get("auto_clean_noise", False),
        "noise_quality_threshold": quality_params.get("noise_quality_threshold", 0.5),
    }
    return cfg


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
    completed_models: set[str] | None = None,
    workspace: Workspace | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "df": df,
        "run_id": run_id,
        "vram_gb": vram_gb,
        "task": config.task.value,
        "config_models": config.models,
        "experiment_name": config.name,
        "cv_strategy": config.cv_strategy.value,
        "workspace": workspace or Workspace(),
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

    def require_available(self) -> None:
        """Fail with an actionable configuration error when Hamilton is absent."""
        if self._dr is None:
            raise HamiltonUnavailableError(
                "Hamilton is required for DAG execution. "
                "Install the training extra with `uv sync --extra train`."
            )

    def execute(
        self,
        inputs: dict[str, Any],
        final_vars: list[str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.require_available()

        targets = final_vars or _MODE_FINAL_VARS.get(self._mode, ["processed_dataframe"])
        return self._dr.execute(targets, inputs=inputs, overrides=overrides)  # type: ignore[no-any-return]

    def describe_pipeline(self, spec: PipelineSpec) -> list[dict[str, Any]]:
        return [
            {
                "step": s.name.value,
                "enabled": s.enabled,
                "params": s.params,
            }
            for s in spec.steps
        ]

    def get_mermaid_graph(self, spec: PipelineSpec | None = None) -> str:
        if spec is not None:
            lines = ["graph TD"]
            for i, s in enumerate(spec.steps):
                label = s.name.value
                disabled = not s.enabled
                if disabled:
                    label = f"{label} ~disabled~"
                lines.append(f"    step_{i}[{label}]")
                if disabled:
                    lines.append(f"    class step_{i} disabled")
                if i > 0:
                    lines.append(f"    step_{i - 1} --> step_{i}")
            if any(not s.enabled for s in spec.steps):
                lines.append("    classDef disabled fill:#eee,stroke:#999,color:#999")
            return "\n".join(lines)
        if self._dr is None:
            return "graph TD\n    A[Raw Data] --> B[Processed Data]"
        result = self._dr.display_all_functions()
        if isinstance(result, str):
            return result
        return getattr(result, "source", str(result))

    def run_preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        self.require_available()
        result = self.execute(inputs={"df": df})
        return result.get("processed_dataframe", df)  # type: ignore[no-any-return]

    def run_training(
        self,
        config: ExperimentConfig,
        df: pl.DataFrame,
        run_id: str,
        vram_gb: float = 0.0,
        completed_models: set[str] | None = None,
        workspace: Workspace | None = None,
    ) -> Any:
        if self._driver_mod is None:
            self.require_available()

        modules = _get_training_modules(config.pipeline)
        hamilton_config = _resolve_hamilton_config(config)
        builder = self._driver_mod.Builder().with_modules(*modules).with_config(hamilton_config)
        if self._tracker is not None:
            from iter8ml.engine.pipelines.hooks.tracking_hook import TrackingHook

            hook = TrackingHook(self._tracker, run_id)
            builder = builder.with_adapters(hook)
        dr = builder.build()

        inputs = _config_to_inputs(
            config,
            df,
            run_id,
            vram_gb,
            completed_models=completed_models,
            workspace=workspace,
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
            self.require_available()

        from iter8ml.engine.pipelines.nodes import drift_detection, prep

        modules = [prep, drift_detection]
        builder = self._driver_mod.Builder().with_modules(*modules)
        builder = builder.with_config({"drift_method": drift_method})
        dr = builder.build()
        result = dr.execute(
            ["drift_report"],
            inputs={"reference_df": reference_df, "live_df": live_df},
        )
        return result.get("drift_report")
