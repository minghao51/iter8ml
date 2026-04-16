# Example Configuration

This file provides a complete, runnable example for the tabular-blueprint experiment configuration using `ExperimentConfig` (Pydantic). Use this as a template for your own projects.

## Basic Classification Example

```python
from configs.experiment import ExperimentConfig

config = ExperimentConfig(
    name="credit_risk_v1",
    task="classification",
    target_col="default",
    data_path="data/credit_risk.csv",
    models="auto",                     # Auto-selects models based on data size/hardware
    run_hpo=False,                     # Disable HPO for quick baseline
    hpo_n_trials=50,                   # Used only when run_hpo=True
    cv_folds=5,
    cv_strategy="stratified",
    metrics=["roc_auc", "f1_macro"],
    random_seed=42,
    max_workers=4,                     # Controls concurrent model training
    run_quality_audit=True,            # Cleanlab audit for label noise (default True)
    tracker="jsonl",                   # "jsonl" | "wandb" | "mlflow"
)
```

## Basic Regression Example

```python
from configs.experiment import ExperimentConfig

config = ExperimentConfig(
    name="house_prices_regression",
    task="regression",
    target_col="price",
    data_path="data/housing.csv",
    models=["catboost", "lightgbm"],   # Explicit model list
    run_hpo=True,
    hpo_n_trials=100,
    cv_folds=3,
    metrics=["rmse", "mae", "r2"],
    random_seed=123,
    tracker="jsonl",
)
```

## With HPO and Multiple Models

```python
from configs.experiment import ExperimentConfig

config = ExperimentConfig(
    name="tabular_benchmark",
    task="classification",
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
- `run_hpo=True`: Triggers Optuna study with per-model default search spaces from `configs/model_configs.py`
- `tracker="jsonl"`: Always writes `workspace/experiments.jsonl` (W&B/MLflow are additive mirrors)
- `run_quality_audit=True`: Runs Cleanlab label-noise audit (skip only for very large datasets >500k)
- `max_workers`: Limits concurrent model training; set to 1 for single GPU with low VRAM

## Overriding Per-Model Defaults (Advanced)

Individual model hyperparameter defaults can be adjusted in `configs/model_configs.py`:

```python
# Example: stricter early stopping for CatBoost
from configs.model_configs import get_model_defaults

defaults = get_model_defaults("catboost")
defaults["early_stopping_rounds"] = 20
defaults["depth"] = 6
```

## Running the Example

```bash
# Using the example config
uv run tabblueprint run --config configs/example.py

# View leaderboard after run
uv run tabblueprint leaderboard --top 5 --metric roc_auc
```