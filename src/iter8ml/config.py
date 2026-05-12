"""Application configuration: experiment settings and hardware profiles."""

import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any, Literal

import psutil
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from iter8ml.constants import CVStrategy, EmbeddingMethod, FeatureStrategy, TaskType, TrackerType

_omp_configured: bool = False

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


class HPOConfig(BaseModel):
    """Hyperparameter optimization configuration."""

    run: bool = False
    n_trials: int = 50


class QualityConfig(BaseModel):
    """Data quality audit configuration."""

    run_audit: bool = True
    auto_clean_noise: bool = False
    noise_quality_threshold: float = 0.5


# Map flat YAML keys → nested model attribute names
# Also powers __getattr__/__setattr__ delegate access on ExperimentConfig.
_FLAT_DELEGATES: dict[str, tuple[str, str, str | None]] = {
    # flat_key: (nested_config, nested_field, nested_config_cls_key_for_default_factory)
    "embedding_method": ("embedding", "method", None),
    "embedding_dim": ("embedding", "dim", None),
    "embedding_max_categories": ("embedding", "max_categories", None),
    "embedding_epochs": ("embedding", "epochs", None),
    "embedding_lr": ("embedding", "lr", None),
    "embedding_mlp_width": ("embedding", "mlp_width", None),
    "embedding_mlp_depth": ("embedding", "mlp_depth", None),
    "embedding_ae_latent_dim": ("embedding", "ae_latent_dim", None),
    "embedding_ae_dropout": ("embedding", "ae_dropout", None),
    "afe_top_k": ("afe", "top_k", None),
    "afe_lift_threshold": ("afe", "lift_threshold", None),
    "afe_pruning": ("afe", "pruning", None),
    "afe_prune_min_importance": ("afe", "prune_min_importance", None),
    "run_hpo": ("hpo", "run", None),
    "hpo_n_trials": ("hpo", "n_trials", None),
    "run_quality_audit": ("quality", "run_audit", None),
    "auto_clean_noise": ("quality", "auto_clean_noise", None),
    "noise_quality_threshold": ("quality", "noise_quality_threshold", None),
}


class ExperimentConfig(BaseModel):
    """Experiment configuration.

    Sections: Core, HPO, Data Quality, Feature Engineering, Embedding,
    Tracking & Output, Advanced, LLM, Model Overrides.
    """

    # --- Core ---
    name: str
    task: TaskType
    target_col: str
    data_path: str
    cv_folds: int = 5
    cv_strategy: CVStrategy = CVStrategy.STRATIFIED
    models: list[str] | Literal["auto"] = "auto"
    random_seed: int = 42
    metrics: list[str] = Field(default_factory=lambda: ["roc_auc", "f1_macro"])

    # --- Nested configs ---
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    afe: AFEConfig = Field(default_factory=AFEConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    target_transform: Literal["none", "auto", "log1p", "yeo-johnson", "box-cox"] = "none"
    target_skewness_threshold: float = 1.0

    # --- Tracking & Output ---
    tracker: TrackerType = TrackerType.JSONL
    workspace_dir: Path = Field(default_factory=lambda: Path("workspace"))
    feature_strategy: FeatureStrategy = FeatureStrategy.NONE

    # --- Advanced ---
    max_workers: int = Field(default=1, description="Number of models to train concurrently")
    data_sample: float = Field(
        default=1.0, description="Fraction of data to use (0.0, 1.0]. 1.0 = full dataset"
    )
    calibration: Literal["none", "platt", "isotonic"] = "none"
    drift_detection: Literal["none", "psi", "domain_classifier", "both"] = "psi"
    shap_enabled: bool = False

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
            cfg_attr, field_attr, _ = _FLAT_DELEGATES[name]
            return getattr(getattr(self, cfg_attr), field_attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _FLAT_DELEGATES:
            cfg_attr, field_attr, _ = _FLAT_DELEGATES[name]
            sub = object.__getattribute__(self, cfg_attr)
            setattr(sub, field_attr, value)
            return
        super().__setattr__(name, value)

    @model_validator(mode="before")
    @classmethod
    def nest_flat_config_fields(cls, data: Any) -> Any:
        """Accept flat field keys from YAML/JSON and nest them into sub-configs."""
        if not isinstance(data, dict):
            return data

        nested: dict[str, dict[str, Any]] = {}
        for flat_key, (cfg_key, field_key, _) in _FLAT_DELEGATES.items():
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
        if self.run_hpo and self.hpo_n_trials <= 0:
            raise ValueError("hpo_n_trials must be > 0 when run_hpo is True")
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

    @field_serializer("workspace_dir", when_used="json")
    def serialize_path(self, value: Path) -> str:
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
        return os.cpu_count() or 1

    @classmethod
    def configure_omp_threads(cls, threads: int | None = None) -> int:
        global _omp_configured
        if threads is None:
            if _omp_configured:
                return int(os.environ.get("OMP_NUM_THREADS", str(cls._get_default_threads())))
            thread_count = cls._get_default_threads()
            os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
            _omp_configured = True
            return thread_count
        os.environ["OMP_NUM_THREADS"] = str(threads)
        return threads
