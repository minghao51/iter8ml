"""Application configuration: experiment settings and hardware profiles."""

import importlib.util
import json
import os
import platform
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import psutil
import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from iter8ml.constants import CVStrategy, EmbeddingMethod, TaskType, TrackerType

_omp_configured: bool = False

# Conservative OpenMP thread cap for libgomp-based backends (lightgbm, xgboost).
# On Intel hybrid-core (P+E) CPUs under Linux/WSL2, libgomp deadlocks when
# spawning threads across all cores — verified to hang at threads >= 10 on a
# 14-core Core Ultra 225H, while <= 8 is stable (see docs/plan/
# portfolio-roadmap-20260805.md §1.6b). Applied on Linux only; Windows/macOS use
# a different OpenMP runtime and are unaffected. 8 is also a sane default for
# GBDT tabular work (sublinear scaling past ~8 threads). Large servers: override
# via the OMP_NUM_THREADS env var.
_OMP_THREAD_CAP: int = 8

DEFAULT_LLM_MODEL: str = "claude-sonnet-4-20250514"


def _raise_if_unknown_model_names(names: set[str], label: str) -> None:
    from iter8ml.engine.models.factory import available_model_names

    known = set(available_model_names())
    unknown = names - known
    if unknown:
        raise ValueError(f"Unknown {label}: {sorted(unknown)}. Known: {sorted(known)}")


class EmbeddingConfig(BaseModel):
    """Embedding configuration for high-cardinality categorical features."""

    method: EmbeddingMethod = EmbeddingMethod.ENTITY
    dim: int = 16
    max_categories: int = 50
    epochs: int = 10
    lr: float = 1e-3
    mlp_width: int = 128
    mlp_depth: int = 2
    ae_latent_dim: int = 32
    ae_dropout: float = 0.2


class AFEConfig(BaseModel):
    """Automated feature engineering configuration."""

    top_k: int = 10
    lift_threshold: float = 0.01
    pruning: bool = False
    prune_min_importance: float = 0.001
    n_jobs: int = 1
    max_candidate_pairs: int = 200


class QualityConfig(BaseModel):
    """Data quality audit worker settings.

    Audit enablement and noise-cleaning behavior live in pipeline step params
    (``quality_audit`` step); see ``docs/pipeline-architecture.md``.
    """

    leakage_n_jobs: int = 1


class StepName(StrEnum):
    DATA_PREP = "data_prep"
    QUALITY_AUDIT = "quality_audit"
    LEAKAGE_AUDIT = "leakage_audit"
    TARGET_TRANSFORM = "target_transform"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


class PipelineStep(BaseModel):
    name: StepName
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineSpec(BaseModel):
    steps: list[PipelineStep] = Field(default_factory=lambda: PipelineSpec._default_steps())

    @staticmethod
    def _default_steps() -> list[PipelineStep]:
        return [
            PipelineStep(name=StepName.DATA_PREP),
            PipelineStep(name=StepName.QUALITY_AUDIT),
            PipelineStep(name=StepName.LEAKAGE_AUDIT),
            PipelineStep(name=StepName.TARGET_TRANSFORM),
            PipelineStep(name=StepName.FEATURE_ENGINEERING),
            PipelineStep(name=StepName.MODEL_TRAINING),
            PipelineStep(name=StepName.CALIBRATION),
            PipelineStep(name=StepName.EVALUATION),
        ]

    def is_enabled(self, name: StepName) -> bool:
        return any(s.name == name and s.enabled for s in self.steps)

    def step_params(self, name: StepName) -> dict[str, Any]:
        for s in self.steps:
            if s.name == name:
                return s.params
        return {}


# --- Legacy flat-key compatibility layer (single supported shim) -------------
# These mappings are the ONLY supported way to accept the older flat YAML/JSON
# config schema. ``_FLAT_DELEGATES`` maps flat keys onto nested sub-config
# attributes (and powers __getattr__/__setattr__ delegate access),
# ``_LEGACY_PIPELINE_KEYS`` covers deprecated step-level keys, and
# ``ExperimentConfig.accept_legacy_flat_keys`` performs the nesting at parse time.
# Do not add new flat keys elsewhere — extend this layer only.
# ---------------------------------------------------------------------------
# Config keys removed in favor of pipeline step params (PipelineSpec). A user
# config containing any of these (flat or nested under ``hpo``/``quality``)
# raises a clear error instead of being silently ignored — see
# ``ExperimentConfig.accept_legacy_flat_keys`` and docs/pipeline-architecture.md.
_REMOVED_CONFIG_KEYS: dict[str, str] = {
    "run_hpo": "HPO is configured via pipeline step params",
    "hpo_n_trials": "HPO is configured via pipeline step params",
    "run_audit": "quality auditing is configured via pipeline step params",
    "auto_clean_noise": "noise cleaning is configured via the quality_audit step",
    "noise_quality_threshold": "noise cleaning is configured via the quality_audit step",
    "shap_enabled": "SHAP explainability is not part of the training pipeline; "
    "use model_overrides + external analysis instead",
    "drift_detection": "run drift analysis explicitly with `iter8 drift "
    "--reference <ref> --new <new> --method <ks|psi|domain|both>`",
}

# Same keys in their old nested positions (``hpo.run`` was a ``HPOConfig`` field;
# the ``quality`` equivalents were ``QualityConfig`` fields).
_REMOVED_NESTED_CONFIG_KEYS: dict[str, str] = {
    "hpo.run": "HPO is configured via pipeline step params",
    "hpo.n_trials": "HPO is configured via pipeline step params",
    "quality.run_audit": "quality auditing is configured via pipeline step params",
    "quality.auto_clean_noise": "noise cleaning is configured via the quality_audit step",
    "quality.noise_quality_threshold": "noise cleaning is configured via the quality_audit step",
}

_FLAT_DELEGATES: dict[str, tuple[str, str]] = {
    "embedding_method": ("embedding", "method"),
    "embedding_dim": ("embedding", "dim"),
    "embedding_max_categories": ("embedding", "max_categories"),
    "embedding_epochs": ("embedding", "epochs"),
    "embedding_lr": ("embedding", "lr"),
    "embedding_mlp_width": ("embedding", "mlp_width"),
    "embedding_mlp_depth": ("embedding", "mlp_depth"),
    "embedding_ae_latent_dim": ("embedding", "ae_latent_dim"),
    "embedding_ae_dropout": ("embedding", "ae_dropout"),
    "afe_top_k": ("afe", "top_k"),
    "afe_lift_threshold": ("afe", "lift_threshold"),
    "afe_pruning": ("afe", "pruning"),
    "afe_prune_min_importance": ("afe", "prune_min_importance"),
    "afe_n_jobs": ("afe", "n_jobs"),
    "afe_max_candidate_pairs": ("afe", "max_candidate_pairs"),
    "leakage_n_jobs": ("quality", "leakage_n_jobs"),
}

_LEGACY_PIPELINE_KEYS: tuple[str, ...] = (
    "run_quality_audit",
    "run_leakage_audit",
    "target_transform",
    "target_skewness_threshold",
    "feature_strategy",
    "calibration",
)


class ExperimentConfig(BaseModel):
    """Experiment configuration.

    Sections: Core, Data Quality, Feature Engineering, Embedding,
    Tracking & Output, Advanced, LLM, Model Overrides.

    Unknown top-level keys are rejected (``extra="forbid"``) so config typos
    fail at parse time instead of silently running with defaults. Legacy flat
    keys are rewritten first by :meth:`accept_legacy_flat_keys`.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Core ---
    name: str
    task: TaskType
    target_col: str
    data_path: str
    cv_folds: int = 5
    cv_strategy: CVStrategy = CVStrategy.STRATIFIED
    models: list[str] | Literal["auto"] = "auto"
    random_seed: int = 42
    positive_class: str | float | bool | None = Field(
        default=None,
        description=(
            "Classification only: which target value is the positive class. "
            "Encoded to 1 before training so roc_auc orientation is explicit "
            "instead of depending on value ordering. Must be present in the "
            "target's values; binary targets only."
        ),
    )
    metrics: list[str] = Field(default_factory=lambda: ["roc_auc", "f1_macro"])
    primary_metric: str | None = Field(
        default=None,
        description=(
            "Metric used for leaderboard ranking, best-model promotion, and lift. "
            "Defaults to metrics[0]; must be a member of metrics."
        ),
    )
    ignore_cols: list[str] = Field(
        default_factory=list,
        description="Columns to drop before feature engineering (e.g. IDs, leaky columns).",
    )

    # --- Nested configs ---
    quality: QualityConfig = Field(default_factory=QualityConfig)
    afe: AFEConfig = Field(default_factory=AFEConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)

    # --- Tracking & Output ---
    tracker: TrackerType = TrackerType.JSONL
    tracker_settings: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Constructor kwargs for the selected tracker backend, validated "
            "against a per-backend allowlist (jsonl: max_file_size_mb, "
            "backup_count; wandb: project/entity/mode/tags/name/notes/group/"
            "job_type/dir; mlflow: tracking_uri/experiment_name). Unknown "
            "keys fail loudly."
        ),
    )

    # --- Advanced ---
    max_workers: int = Field(
        default=1,
        description=(
            "Number of models to train concurrently. "
            "Warning: values > 1 may cause thread contention with GBDT models "
            "(CatBoost, LightGBM, XGBoost) that manage their own thread pools."
        ),
    )
    strict_thread_safety: bool = Field(
        default=True,
        description=(
            "When true, disables cross-model parallelism if any selected model is "
            "internally multi-threaded (CatBoost/LightGBM/XGBoost)."
        ),
    )
    data_sample: float = Field(
        default=1.0, description="Fraction of data to use (0.0, 1.0]. 1.0 = full dataset"
    )

    # --- LLM ---
    llm_enabled: bool = False
    llm_model: str = Field(
        default_factory=lambda: os.getenv("ITER8ML_LLM_MODEL", DEFAULT_LLM_MODEL),
        description="LLM model for commentary. Override via ITER8ML_LLM_MODEL env var.",
    )

    # --- Model Overrides ---
    model_overrides: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-model hyperparameter overrides, e.g. {'catboost': {'depth': 8}}",
    )

    def __getattr__(self, name: str) -> Any:
        if name in _FLAT_DELEGATES:
            cfg_attr, field_attr = _FLAT_DELEGATES[name]
            return getattr(getattr(self, cfg_attr), field_attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _FLAT_DELEGATES:
            cfg_attr, field_attr = _FLAT_DELEGATES[name]
            sub = object.__getattribute__(self, cfg_attr)
            setattr(sub, field_attr, value)
            return
        super().__setattr__(name, value)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_flat_keys(cls, data: Any) -> Any:
        """Accept flat field keys from YAML/JSON and nest them into sub-configs.

        This is the single supported compatibility layer for legacy flat-key
        configs (see ``_FLAT_DELEGATES`` / ``_LEGACY_PIPELINE_KEYS``). It is
        intentionally the only place that maps the older flat schema onto the
        nested ``ExperimentConfig`` model.
        """
        if not isinstance(data, dict):
            return data

        # Deprecation guard: removed keys must fail loudly, not silently no-op.
        for key in _REMOVED_CONFIG_KEYS:
            if key in data:
                raise ValueError(
                    f"'{key}' was removed; {_REMOVED_CONFIG_KEYS[key]} "
                    "(see docs/pipeline-architecture.md)"
                )
        for full_key, reason in _REMOVED_NESTED_CONFIG_KEYS.items():
            section, key = full_key.split(".", 1)
            sub = data.get(section)
            if isinstance(sub, dict) and key in sub:
                raise ValueError(
                    f"'{full_key}' was removed; {reason} (see docs/pipeline-architecture.md)"
                )

        # Backward compatibility for legacy step-level flat keys.
        legacy: dict[str, Any] = {}
        for key in _LEGACY_PIPELINE_KEYS:
            if key in data:
                legacy[key] = data.pop(key)

        if legacy:
            user_provided_pipeline = "pipeline" in data
            pipeline = data.setdefault("pipeline", {})
            steps = pipeline.setdefault("steps", [])
            if not isinstance(steps, list):
                raise ValueError("pipeline.steps must be a list")
            if not user_provided_pipeline and not steps:
                # Legacy keys must MODIFY the default 8-step pipeline, not
                # replace it with a fragment: a steps list containing only the
                # rewritten key would drop every other step (e.g. no
                # FEATURE_ENGINEERING step => no ``training_features`` input
                # => the training DAG cannot execute).
                steps.extend({"name": step.value} for step in StepName)

            def _upsert(
                step_name: str,
                enabled: bool | None = None,
                params: dict[str, Any] | None = None,
            ) -> None:
                for step in steps:
                    if isinstance(step, dict) and step.get("name") == step_name:
                        if enabled is not None:
                            step["enabled"] = enabled
                        if params:
                            step_params = step.setdefault("params", {})
                            if not isinstance(step_params, dict):
                                raise ValueError(
                                    f"pipeline step '{step_name}' params must be a dict"
                                )
                            step_params.update(params)
                        return
                new_step: dict[str, Any] = {"name": step_name}
                if enabled is not None:
                    new_step["enabled"] = enabled
                if params:
                    new_step["params"] = dict(params)
                steps.append(new_step)

            if "run_quality_audit" in legacy:
                _upsert("quality_audit", enabled=bool(legacy["run_quality_audit"]))

            if "run_leakage_audit" in legacy:
                _upsert("leakage_audit", enabled=bool(legacy["run_leakage_audit"]))

            target_params: dict[str, Any] = {}
            if "target_transform" in legacy:
                target_params["method"] = legacy["target_transform"]
            if "target_skewness_threshold" in legacy:
                target_params["skewness_threshold"] = legacy["target_skewness_threshold"]
            if target_params:
                _upsert("target_transform", params=target_params)

            if "feature_strategy" in legacy:
                _upsert("feature_engineering", params={"strategy": legacy["feature_strategy"]})

            if "calibration" in legacy:
                _upsert("calibration", params={"method": legacy["calibration"]})

        nested: dict[str, dict[str, Any]] = {}
        for flat_key, (cfg_key, field_key) in _FLAT_DELEGATES.items():
            if (val := data.pop(flat_key, None)) is not None and cfg_key not in data:
                nested.setdefault(cfg_key, {})[field_key] = val

        for cfg_key, fields in nested.items():
            data.setdefault(cfg_key, {}).update(fields)

        return data

    @field_validator("data_sample")
    @classmethod
    def validate_data_sample(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("data_sample must be in (0.0, 1.0]")
        return v

    @field_validator("models")
    @classmethod
    def validate_model_names(cls, v: list[str] | Literal["auto"]) -> list[str] | Literal["auto"]:
        if v != "auto":
            _raise_if_unknown_model_names(set(v), "model names")
        return v

    @field_validator("model_overrides")
    @classmethod
    def validate_model_overrides(
        cls, v: dict[str, dict[str, Any]] | None
    ) -> dict[str, dict[str, Any]] | None:
        if v is not None:
            _raise_if_unknown_model_names(set(v), "model_overrides keys")
        return v

    @model_validator(mode="after")
    def apply_task_defaults(self) -> "ExperimentConfig":
        if "metrics" not in self.model_fields_set:
            self.metrics = (
                ["roc_auc", "f1_macro"] if self.task == TaskType.CLASSIFICATION else ["rmse", "r2"]
            )
        if "cv_strategy" not in self.model_fields_set:
            self.cv_strategy = (
                CVStrategy.STRATIFIED if self.task == TaskType.CLASSIFICATION else CVStrategy.KFOLD
            )
        return self

    @model_validator(mode="after")
    def validate_task_consistency(self) -> "ExperimentConfig":
        """Fail at parse time on task-incompatible metrics, CV strategy, or primary metric.

        Without this, a wrong-task metric surfaces only as a per-model ``KeyError``
        mid-run (previously swallowed into a green-but-empty result), and a
        ``stratified`` split on a continuous regression target fails only when
        folds are being drawn.
        """
        from iter8ml.engine.evaluator import available_metric_names

        task = self.task.value
        valid = set(available_metric_names(task))
        unknown = [m for m in self.metrics if m not in valid]
        if unknown:
            raise ValueError(
                f"Unknown metrics for task '{task}': {unknown}. Valid metrics: {sorted(valid)}"
            )
        if self.task == TaskType.REGRESSION and self.cv_strategy == CVStrategy.STRATIFIED:
            raise ValueError(
                "cv_strategy='stratified' requires class labels and is invalid for "
                "regression tasks; use 'kfold' or 'timeseries'"
            )
        if self.task == TaskType.REGRESSION and self.positive_class is not None:
            raise ValueError(
                f"positive_class={self.positive_class!r} is only valid for classification "
                "tasks; remove it from this regression config"
            )
        if self.primary_metric is None:
            self.primary_metric = self.metrics[0] if self.metrics else None
        elif self.primary_metric not in self.metrics:
            raise ValueError(
                f"primary_metric '{self.primary_metric}' must be one of metrics {self.metrics}"
            )
        return self

    @classmethod
    def from_file(
        cls, path: str | Path, *, allow_unsafe_python: bool = False
    ) -> "ExperimentConfig":
        """Load config from a .yaml, .toml, .json, or .py file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls.model_validate(data)

        if suffix == ".toml":
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef, import-not-found]
            with open(path, "rb") as f:
                data = tomllib.load(f)
            return cls.model_validate(data)

        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            return cls.model_validate(data)

        if suffix == ".py":
            if not allow_unsafe_python:
                raise ValueError(
                    "Python config files are disabled by default for safety. "
                    "Use --allow-unsafe-config to enable loading .py config files."
                )
            spec = importlib.util.spec_from_file_location("experiment_config", path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load config module: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            experiment_config = getattr(module, "config", None)
            if experiment_config is None:
                raise ValueError(f"Config module must define `config`: {path}")
            return experiment_config  # type: ignore[no-any-return]

        raise ValueError(
            f"Unsupported config format: {suffix}. Use .yaml, .yml, .toml, .json, or .py"
        )

    @field_serializer("task", "cv_strategy", "tracker", when_used="json")
    def serialize_enum(self, value: Any) -> str:
        if isinstance(value, TaskType | CVStrategy | TrackerType):
            return value.value  # type: ignore[no-any-return]
        return str(value)


class HardwareProfile(BaseModel):
    """Detected hardware capabilities (GPU, RAM, CPU cores)."""

    vram_gb: float
    system_ram_gb: float
    cpu_cores: int
    has_gpu: bool
    gpu_name: str | None

    @classmethod
    def detect(cls) -> "HardwareProfile":
        try:
            import torch

            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                has_gpu = True
                gpu_name = torch.cuda.get_device_name(0)
            else:
                vram = 0.0
                has_gpu = False
                gpu_name = None
        except ImportError:
            vram = 0.0
            has_gpu = False
            gpu_name = None

        return cls(
            vram_gb=round(vram, 1),
            system_ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
            cpu_cores=psutil.cpu_count(logical=False),
            has_gpu=has_gpu,
            gpu_name=gpu_name,
        )

    @classmethod
    def _get_default_threads(cls) -> int:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return 1
        # The libgomp hybrid-core deadlock is Linux-specific; other platforms use
        # a different OpenMP runtime and don't need the cap.
        if platform.system() == "Linux":
            return min(os.cpu_count() or 1, _OMP_THREAD_CAP)
        return os.cpu_count() or 1

    @classmethod
    def configure_omp_threads(cls, threads: int | None = None) -> int:
        global _omp_configured
        # Passive wait policy prevents libgomp active-spin live-locks on
        # Intel hybrid CPUs (P+E cores, no SMT): spinning worker threads can
        # starve on heterogeneous cores during GBDT training barriers.
        # Must be set before the OpenMP runtime initializes (i.e. before any
        # GBDT library loads libgomp) — same constraint as the thread cap.
        os.environ.setdefault("OMP_WAIT_POLICY", "passive")
        if threads is None:
            if _omp_configured:
                return int(os.environ.get("OMP_NUM_THREADS", str(cls._get_default_threads())))
            thread_count = cls._get_default_threads()
            os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
            _omp_configured = True
            return thread_count
        os.environ["OMP_NUM_THREADS"] = str(threads)
        return threads
