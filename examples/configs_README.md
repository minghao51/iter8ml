# Example Configuration

This file provides a complete, runnable example for the iter8ml experiment configuration using `ExperimentConfig` (Pydantic). Use this as a template for your own projects.

## Basic Classification Example

```python
from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy, TaskType, TrackerType

config = ExperimentConfig(
    name="credit_risk_v1",
    task=TaskType.CLASSIFICATION,
    target_col="default",
    data_path="data/credit_risk.csv",
    models="auto",                     # Auto-selects models based on data size/hardware
    run_hpo=False,                     # Disable HPO for quick baseline
    hpo_n_trials=50,                   # Used only when run_hpo=True
    cv_folds=5,
    cv_strategy=CVStrategy.STRATIFIED,
    metrics=["roc_auc", "f1_macro"],
    random_seed=42,
    max_workers=4,                     # Controls concurrent model training
    run_quality_audit=True,            # Cleanlab audit for label noise (default True)
    tracker=TrackerType.JSONL,        # "jsonl" | "wandb" | "mlflow"
)
```

## Basic Regression Example

```python
from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy, TaskType, TrackerType

config = ExperimentConfig(
    name="house_prices_regression",
    task=TaskType.REGRESSION,
    target_col="price",
    data_path="data/housing.csv",
    models=["catboost", "lightgbm"],   # Explicit model list
    run_hpo=True,
    hpo_n_trials=100,
    cv_folds=3,
    metrics=["rmse", "mae", "r2"],
    random_seed=123,
    tracker=TrackerType.JSONL,
)
```

## With HPO and Multiple Models

```python
from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType, TrackerType

config = ExperimentConfig(
    name="tabular_benchmark",
    task=TaskType.CLASSIFICATION,
    target_col="target",
    data_path="data/medium_dataset.csv",   # ~50k rows
    models="auto",                         # ModelSelector decides: CatBoost/LightGBM/XGBoost
    run_hpo=True,
    hpo_n_trials=100,
    cv_folds=5,
    metrics=["roc_auc"],
    random_seed=42,
    max_workers=8,                         # Enable concurrent training (ensure GPU/CPU capacity)
)
```

## Notes on Key Fields

- `models="auto"`: Uses `ModelSelector` to choose based on `n_rows`, `vram_gb`, `has_text_cols`
- `run_hpo=True`: Triggers Optuna study with per-model default search spaces
- `tracker=TrackerType.JSONL`: Always writes `workspace/experiments.jsonl` (W&B/MLflow are additive mirrors)
- `run_quality_audit=True`: Runs Cleanlab label-noise audit (skip only for very large datasets >500k)
- `max_workers`: Limits concurrent model training; set to 1 for single GPU with low VRAM

## Advanced Configuration

### Target Transform

For regression tasks with skewed targets, enable automatic transform:

```python
config = ExperimentConfig(
    name="skewed_regression",
    task=TaskType.REGRESSION,
    target_col="price",
    data_path="data/housing.csv",
    target_transform="auto",           # Automatically selects best transform
    target_skewness_threshold=0.5,     # Threshold for triggering transform
)
```

### Calibration

Post-training probability calibration for classification:

```python
config = ExperimentConfig(
    name="calibrated_classification",
    task=TaskType.CLASSIFICATION,
    target_col="target",
    data_path="data/classification.csv",
    calibration="isotonic",           # "none" | "platt" | "isotonic"
)
```

### SHAP Analysis

Enable SHAP value computation after training:

```python
config = ExperimentConfig(
    name="explainable_classification",
    task=TaskType.CLASSIFICATION,
    target_col="target",
    data_path="data/classification.csv",
    shap_enabled=True,
)
```

## Running the Example

```bash
# Using an example config
uv run iter8 run --config examples/credit_risk.py

# View leaderboard after run
uv run iter8 leaderboard --top 5 --metric roc_auc
```
