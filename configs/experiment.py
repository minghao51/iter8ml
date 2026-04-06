from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    name: str
    task: Literal["classification", "regression"]
    target_col: str
    data_path: str
    cv_folds: int = 5
    cv_strategy: Literal["kfold", "stratified", "timeseries"] = "stratified"
    run_hpo: bool = False
    hpo_n_trials: int = 50
    models: list[str] | Literal["auto"] = "auto"
    random_seed: int = 42
    metrics: list[str] = Field(default_factory=lambda: ["roc_auc", "f1_macro"])
    tracker: Literal["jsonl", "wandb", "mlflow"] = "jsonl"
    run_quality_audit: bool = True

    @model_validator(mode="after")
    def apply_task_defaults(self):
        """Align default metrics and CV strategy with the selected task."""
        if "metrics" not in self.model_fields_set:
            self.metrics = (
                ["roc_auc", "f1_macro"] if self.task == "classification" else ["rmse", "r2"]
            )
        if "cv_strategy" not in self.model_fields_set:
            self.cv_strategy = "stratified" if self.task == "classification" else "kfold"
        return self
