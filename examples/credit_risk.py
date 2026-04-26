"""Example: binary classification on a credit risk dataset.

This config demonstrates a typical tabular classification experiment
with HPO enabled on the champion model after an initial baseline run.

Usage:
    uv run tabblueprint run --config configs/examples/credit_risk.py
    uv run tabblueprint hpo --data data/credit.parquet --target default \
        --model catboost --trials 100
"""

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.constants import CVStrategy, TaskType, TrackerType

config = ExperimentConfig(
    name="credit_risk_v2",
    task=TaskType.CLASSIFICATION,
    target_col="default",
    data_path="data/credit_risk_v2.parquet",
    cv_folds=5,
    cv_strategy=CVStrategy.STRATIFIED,
    run_hpo=False,
    hpo_n_trials=100,
    models="auto",
    metrics=["roc_auc", "f1_macro", "log_loss"],
    tracker=TrackerType.JSONL,
    run_quality_audit=True,
)
