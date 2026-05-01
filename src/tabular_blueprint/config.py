"""Application configuration: experiment settings and hardware profiles."""

import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any, Literal

import psutil
import yaml  # type: ignore[import]
from pydantic import BaseModel, Field, field_serializer, model_validator

from tabular_blueprint.constants import CVStrategy, EmbeddingMethod, TaskType, TrackerType

_omp_configured: bool = False


class ExperimentConfig(BaseModel):
    name: str
    task: TaskType
    target_col: str
    data_path: str
    cv_folds: int = 5
    cv_strategy: CVStrategy = CVStrategy.STRATIFIED
    run_hpo: bool = False
    hpo_n_trials: int = 50
    models: list[str] | Literal["auto"] = "auto"
    random_seed: int = 42
    metrics: list[str] = Field(default_factory=lambda: ["roc_auc", "f1_macro"])
    tracker: TrackerType = TrackerType.JSONL
    run_quality_audit: bool = True
    auto_clean_noise: bool = False
    noise_quality_threshold: float = 0.5
    workspace_dir: Path = Field(default_factory=lambda: Path("workspace"))
    max_workers: int = Field(default=1, description="Number of models to train concurrently")
    data_sample: float = Field(
        default=1.0, description="Fraction of data to use (0.0-1.0). 1.0 = full dataset"
    )
    afe_enabled: bool = False
    afe_top_k: int = 10
    afe_lift_threshold: float = 0.01
    afe_pruning: bool = False
    afe_prune_min_importance: float = 0.001
    target_transform: Literal["none", "auto", "log1p", "yeo-johnson", "box-cox"] = "none"
    target_skewness_threshold: float = 1.0
    calibration: Literal["none", "platt", "isotonic"] = "none"
    drift_detection: Literal["none", "psi", "domain_classifier", "both"] = "psi"
    shap_enabled: bool = False
    llm_enabled: bool = False
    llm_model: str = "claude-sonnet-4-20250514"
    embedding_enabled: bool = False
    embedding_method: EmbeddingMethod = EmbeddingMethod.ENTITY
    embedding_dim: int = 16
    embedding_max_categories: int = 50
    embedding_epochs: int = 10
    embedding_lr: float = 1e-3
    embedding_mlp_width: int = 128
    embedding_mlp_depth: int = 2
    embedding_ae_latent_dim: int = 32
    embedding_ae_dropout: float = 0.2

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
                import tomli as tomllib  # type: ignore[no-redef]
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
            return experiment_config

        raise ValueError(
            f"Unsupported config format: {suffix}. Use .yaml, .yml, .toml, .json, or .py"
        )

    @field_serializer("task", "cv_strategy", "tracker", "embedding_method", when_used="json")
    def serialize_enum(self, value: Any) -> str:
        if isinstance(value, TaskType | CVStrategy | TrackerType | EmbeddingMethod):
            return value.value
        return str(value)

    @field_serializer("workspace_dir", when_used="json")
    def serialize_path(self, value: Path) -> str:
        return str(value)


class HardwareProfile(BaseModel):
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
