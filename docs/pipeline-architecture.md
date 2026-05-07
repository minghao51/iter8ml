# Pipeline Architecture (Hamilton DAG)

Reference for the Hamilton-based DAG pipeline orchestration: how nodes are composed into pipelines, config variants, tracking hooks, and how to extend the graph.

---

## Overview

The pipeline layer uses [Hamilton](https://github.com/DAGWorks-Inc/hamilton) to compose processing stages into a directed acyclic graph (DAG). Each Python function is a **node** — its parameters declare dependencies (other nodes), and Hamilton resolves execution order automatically.

**Key benefits:**
- **Declarative wiring:** Function signatures define the DAG — no manual orchestration
- **Config variants:** `@config.when(...)` activates different node implementations based on config
- **Observability:** Every node execution can be tracked via lifecycle hooks
- **Single path:** Training execution is DAG-only through Hamilton

---

## Architecture

```
                        ┌─────────────┐
                        │   Trainer    │
                        └──────┬───────┘
                               │
                    ┌──────────┴──────────┐
                    │  Hamilton DAG run   │
                    └──────────┬──────────┘
                               │
                          ┌────┴────┐
                          │ Pipeline │
                          │ Executor │
                          └──────────┘
```

### Entry Point: `Trainer.run()`

**Source:** `src/tabular_blueprint/engine/trainer.py:50`

1. Calls `PipelineExecutor.run_training()` — builds and executes the full DAG

---

## Pipeline Executor

**Source:** `src/tabular_blueprint/pipelines/executor.py:122`

**Class:** `PipelineExecutor`

### Pipeline Modes

| Mode | Enum | Purpose |
|------|------|---------|
| `TRAINING` | `PipelineMode.TRAINING` | Full experiment: preprocessing → data prep → model selection → baselines → feature engineering → training → state |
| `DRIFT` | `PipelineMode.DRIFT` | Drift detection between two datasets |
| `EXPORT` | `PipelineMode.EXPORT` | Export champion model |
| `HPO` | `PipelineMode.HPO` | Hyperparameter optimization |
| `INFERENCE` | `PipelineMode.INFERENCE` | Batch prediction |

### Key Methods

| Method | Final Variables | Description |
|--------|----------------|-------------|
| `run_preprocessing(df)` | `["processed_dataframe"]` | Preprocessing-only pipeline |
| `run_data_prep(df, target_col, ...)` | `["data_prep_result"]` | Preprocessing + quality audit + leakage + target transform |
| `run_training(df, target_col, ...)` | `["training_state"]` | Full training pipeline (all 7 modules) |
| `run_drift(reference_df, live_df, method)` | `["drift_report"]` | Drift detection pipeline |
| `get_mermaid_graph()` | — | Returns Mermaid diagram of the DAG |
| `execute(inputs, final_vars, overrides)` | Custom | Generic execution with custom targets |

---

## Training Pipeline DAG (7 Modules)

**Source:** `pipelines/executor.py:56`

The full training pipeline composes 7 node modules in order:

```python
modules = [
    preprocessing,        # 1. Column detection + imputation + encoding
    data_preparation,     # 2. Quality audit + leakage + target transform
    model_selection,      # 3. Hardware-aware model routing
    baselines,            # 4. Naive + Linear baseline CV
    feature_engineering,  # 5. AFE (config-variant: enabled/disabled)
    model_training,       # 6. Per-model train + calibrate + save
    state_generation,     # 7. Leaderboard + registry + champion update
]
```

### Full DAG Flow

```
df (input)
 │
 ├─→ raw_dataframe
 │     ├─→ numeric_columns
 │     ├─→ categorical_columns
 │     └─→ date_columns
 │           │
 ├─→ fill_nulls_numeric ─→ fill_nulls_categorical
 │           │
 ├─→ null_filled_df
 │           │
 ├─→ decomposed_dates_df
 │           │
 ├─→ encoded_df
 │           │
 ├─→ processed_dataframe ──→ validate_target
 │                             │
 │              ┌──────────────┤
 │              │              │
 │     quality_cleaned_df   (run_quality_audit,
 │              │            auto_clean_noise,
 │              │            noise_quality_threshold)
 │              │
 │     ┌────────┴────────┐
 │     │                 │
 │  adapter_result    feature_names
 │     │
 │  leakage_report ──→ (run_leakage_audit, task)
 │     │
 │  target_transform_result
 │     │
 │  data_prep_result
 │     │
 │  models_to_run ──→ (config_models, vram_gb, task)
 │     │
 │  baseline_models → baseline_scores
 │     │                 │
 │  training_features ──┤  ← @config.when(afe_enabled=True/False)
 │     │                 │
 │  training_results ←───┘  (per-model loop)
 │     │
 │  training_state ──→ (leaderboard, registry, champion)
```

---

## Module Details

### 1. Preprocessing

**Source:** `pipelines/nodes/preprocessing.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `raw_dataframe` | `df` (input) | `pl.DataFrame` |
| `numeric_columns` | `raw_dataframe` | `list[str]` |
| `categorical_columns` | `raw_dataframe` | `list[str]` |
| `date_columns` | `raw_dataframe` | `list[str]` |
| `fill_nulls_numeric` | `raw_dataframe`, `numeric_columns` | `pl.DataFrame` |
| `fill_nulls_categorical` | `fill_nulls_numeric`, `categorical_columns` | `pl.DataFrame` |
| `null_filled_df` | `fill_nulls_numeric`, `fill_nulls_categorical`, `numeric_columns`, `categorical_columns` | `pl.DataFrame` |
| `decomposed_dates_df` | `null_filled_df`, `date_columns` | `pl.DataFrame` |
| `encoded_df` | `decomposed_dates_df`, `categorical_columns` | `pl.DataFrame` |
| `processed_dataframe` | `encoded_df` | `pl.DataFrame` |

See [preprocessing.md](preprocessing.md) for method details.

### 2. Data Preparation

**Source:** `pipelines/nodes/data_preparation.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `validate_target` | `processed_dataframe`, `target_col` | `pl.DataFrame` |
| `quality_cleaned_df` | `validate_target`, `target_col`, `run_quality_audit`, `auto_clean_noise`, `noise_quality_threshold` | `tuple[df, cleaned, n_dropped]` |
| `adapter_result` | `quality_cleaned_df`, `target_col` | `(X, y)` numpy arrays |
| `feature_names` | `quality_cleaned_df`, `target_col` | `list[str]` |
| `leakage_report` | `adapter_result`, `run_leakage_audit`, `task` | `LeakageReport \| None` |
| `target_transform_result` | `adapter_result`, `target_transform`, `target_skewness_threshold` | `(y, transformer, method, skew_orig, skew_trans, applied)` |
| `data_prep_result` | `adapter_result`, `target_transform_result`, `feature_names`, `leakage_report`, `quality_cleaned_df` | `DataPrepResult` |

### 3. Model Selection

**Source:** `pipelines/nodes/model_selection.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `models_to_run` | `data_prep_result`, `task`, `vram_gb`, `config_models` | `list[str]` |

If `config_models="auto"`, uses `ModelSelector.select()`. Otherwise uses the provided list.

### 4. Baselines

**Source:** `pipelines/nodes/baselines.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `baseline_models` | (none) | `dict[str, type]` |
| `baseline_scores` | `data_prep_result`, `baseline_models`, `task`, `cv_folds`, `cv_strategy`, `metrics` | `dict[str, dict[str, float]]` |

### 5. Feature Engineering (Config Variant)

**Source:** `pipelines/nodes/feature_engineering.py`

Uses `@config.when()` to activate different implementations:

| Node | Config Condition | Behavior |
|------|-----------------|----------|
| `training_features__default` | `afe_enabled != True` | Pass-through: returns `(X, feature_names)` from `data_prep_result` |
| `training_features__afe_enabled` | `afe_enabled == True` | Runs full AFE: top-K → interactions → pruning |

Config is set via `builder.with_config({"afe_enabled": True/False})`.

### 6. Model Training

**Source:** `pipelines/nodes/model_training.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `training_results` | `training_features`, `data_prep_result`, `models_to_run`, `baseline_scores`, `task`, `cv_folds`, `cv_strategy`, `metrics`, `calibration`, `workspace_dir`, `run_id` | `list[ModelResult]` |

For each model in `models_to_run` (excluding baselines):
1. Resolve class via `get_model_class(name)`
2. Cross-validate via `Evaluator.evaluate()`
3. Fit on full data (with optional calibration)
4. Save artifact
5. Compute lift over baselines

### 7. State Generation

**Source:** `pipelines/nodes/state_generation.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `training_state` | `training_results`, `baseline_scores`, `metrics`, `run_id`, `experiment_name`, `task`, `workspace_dir` | `TrainingState` |

Aggregates all results, builds leaderboard, promotes champion to registry.

---

## Drift Detection Pipeline

**Source:** `pipelines/nodes/drift_detection.py`

Uses `@config.when(drift_method=...)` for three variants:

| Node | Config | Method |
|------|--------|--------|
| `drift_report__psi` | `drift_method="psi"` | PSI drift detection |
| `drift_report__domain` | `drift_method="domain_classifier"` | Domain classifier AUC |
| `drift_report__both` | `drift_method="both"` | Both methods, merged report |

Config is set via `builder.with_config({"drift_method": "psi"})`.

---

## Tracking Hooks

**Source:** `src/tabular_blueprint/pipelines/hooks/tracking_hook.py`

**Class:** `TrackingHook` — a Hamilton lifecycle adapter that logs node execution to the experiment tracker.

### Hook Points

| Method | Trigger | Action |
|--------|---------|--------|
| `run_on_node_success` | Node completes | Logs `{"event": "node_completed", "node": name, "duration_seconds": ...}` |
| `run_on_node_error` | Node raises exception | Logs `{"event": "node_error", "node": name, "error": str(exception)}` |
| `run_before_node_execution` | Before node runs | No-op (placeholder for future use) |
| `run_after_node_execution` | After node runs | No-op (placeholder for future use) |

### Attachment

Hooks are attached when building the Hamilton driver:

```python
builder = driver.Builder().with_modules(*modules)
hook = TrackingHook(tracker, run_id)
builder = builder.with_adapters(hook)
dr = builder.build()
```

---

## Configuration

**Source:** `src/tabular_blueprint/config.py`

### `ExperimentConfig` — Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | required | `CLASSIFICATION` or `REGRESSION` |
| `target_col` | required | Target column name |
| `data_path` | required | Path to data file |
| `cv_folds` | 5 | Cross-validation folds |
| `cv_strategy` | auto (STRATIFIED for classification, KFOLD for regression) | CV splitting strategy |
| `models` | `"auto"` | `"auto"` for hardware-aware selection, or list of model names |
| `metrics` | auto (`["roc_auc", "f1_macro"]` or `["rmse", "r2"]`) | Evaluation metrics |
| `afe_enabled` | `False` | Enable automated feature engineering |
| `afe_top_k` | 10 | Top-K features for AFE |
| `afe_lift_threshold` | 0.01 | Minimum lift to keep an interaction |
| `afe_pruning` | `False` | Enable feature pruning after AFE |
| `target_transform` | `"none"` | Target transformation: `"none"`, `"auto"`, `"log1p"`, `"yeo-johnson"`, `"box-cox"` |
| `calibration` | `"none"` | Probability calibration: `"none"`, `"platt"`, `"isotonic"` |
| `drift_detection` | `"psi"` | Drift method: `"none"`, `"psi"`, `"domain_classifier"`, `"both"` |
| `shap_enabled` | `False` | Enable SHAP explainability |
| `max_workers` | 1 | Concurrent model training (auto-reduced to 1 for low-VRAM GPUs) |
| `run_quality_audit` | `True` | Enable Cleanlab label noise detection |
| `auto_clean_noise` | `False` | Auto-drop noisy labels |
| `tracker` | `JSONL` | Tracker backend: `JSONL`, `WANDB`, `MLFLOW` |

---

## How to Extend the DAG

### Adding a New Node

1. Create a function in the appropriate module under `pipelines/nodes/`
2. Its parameters are automatically resolved as dependencies by Hamilton
3. The function name becomes the node name in the DAG

```python
# In pipelines/nodes/my_module.py
def my_custom_feature(processed_dataframe: pl.DataFrame) -> pl.DataFrame:
    # processed_dataframe is auto-resolved from the preprocessing module
    return processed_dataframe.with_columns(...)
```

### Adding a Config Variant

Use `@config.when()` to create conditional node implementations:

```python
from hamilton.function_modifiers import config

@config.when(feature_method="advanced")
def training_features__advanced(
    data_prep_result: object,
    task: str,
) -> tuple[np.ndarray, list[str]]:
    # Advanced feature engineering
    ...

@config.when_not(feature_method="advanced")
def training_features__default(
    data_prep_result: object,
) -> tuple[np.ndarray, list[str]]:
    return data_prep_result.X, data_prep_result.feature_names
```

Then set the config when building: `builder.with_config({"feature_method": "advanced"})`.

### Adding a New Pipeline Mode

1. Add a new value to `PipelineMode` enum in `executor.py`
2. Register modules and final variables in `_MODE_MODULES` and `_MODE_FINAL_VARS`
3. Add a `run_<mode>()` method to `PipelineExecutor`

### Adding a New Tracking Hook

Extend `TrackingHook` or create a new adapter implementing the Hamilton lifecycle methods:

- `run_before_node_execution`
- `run_after_node_execution`
- `run_on_node_error`
- `run_on_node_success`

Attach via `builder.with_adapters(hook)`.

---

## DAG Visualization

The executor can generate a Mermaid diagram of the current DAG:

```python
executor = PipelineExecutor(mode=PipelineMode.TRAINING)
print(executor.get_mermaid_graph())
```

This is also logged automatically in the `experiment_started` event under `pipeline_lineage`.
