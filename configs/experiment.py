from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, model_validator

from core.constants import CVStrategy, TaskType, TrackerType


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
    workspace_dir: Path = Field(default_factory=lambda: Path("workspace"))
    max_workers: int = Field(default=1, description="Number of models to train concurrently")

    @model_validator(mode="after")
    def apply_task_defaults(self):
        """Align default metrics and CV strategy with the selected task."""
        if "metrics" not in self.model_fields_set:
            self.metrics = (
                ["roc_auc", "f1_macro"] if self.task == TaskType.CLASSIFICATION else ["rmse", "r2"]
            )
        if "cv_strategy" not in self.model_fields_set:
            self.cv_strategy = (
                CVStrategy.STRATIFIED if self.task == TaskType.CLASSIFICATION else CVStrategy.KFOLD
            )
        return self

    @field_serializer("task", "cv_strategy", "tracker", when_used="json")
    def serialize_enum(self, value: Any) -> str:
        """Serialize enum values to strings for JSON compatibility."""
        if isinstance(value, TaskType | CVStrategy | TrackerType):
            return value.value
        return value

    @field_serializer("workspace_dir", when_used="json")
    def serialize_path(self, value: Path) -> str:
        """Serialize Path to string for JSON compatibility."""
        return str(value)
