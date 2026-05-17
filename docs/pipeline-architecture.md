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

**Source:** `src/iter8ml/engine/trainer.py:50`

1. Calls `PipelineExecutor.run_training()` — builds and executes the full DAG

---

## Pipeline Executor

**Source:** `src/iter8ml/engine/pipelines/executor.py:122`

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
| `run_training(config, df, ...)` | `["training_state"]` | Full training pipeline (spec-driven modules) |
| `run_drift(reference_df, live_df, method)` | `["drift_report"]` | Drift detection pipeline |
| `get_mermaid_graph(spec)` | — | Returns Mermaid diagram of the DAG (spec-aware if provided) |
| `describe_pipeline(spec)` | — | Returns list of dicts with step name, enabled, params |
| `execute(inputs, final_vars, overrides)` | Custom | Generic execution with custom targets |

---

## Training Pipeline DAG (Spec-Driven Modules)

**Source:** `engine/pipelines/executor.py:56`

The full training pipeline composes node modules based on the `PipelineSpec` attached to `ExperimentConfig`:

```python
modules = [prep]
if spec.is_enabled(StepName.FEATURE_ENGINEERING):
    modules.append(features)
modules.append(train)
```

Each module contains Hamilton nodes that may use `@config.when()` variants, resolved from `_resolve_hamilton_config()` which reads step params from the `PipelineSpec`.

### Default Steps

| Step | `StepName` | Default | Configurable Params |
|------|-----------|---------|-------------------|
| Data Prep | `DATA_PREP` | enabled | — |
| Quality Audit | `QUALITY_AUDIT` | enabled | `auto_clean_noise: bool`, `noise_quality_threshold: float` |
| Leakage Audit | `LEAKAGE_AUDIT` | enabled | — |
| Target Transform | `TARGET_TRANSFORM` | enabled | `method: "none"\|"auto"\|"log1p"\|"yeo-johnson"\|"box-cox"`, `skewness_threshold: float` |
| Feature Engineering | `FEATURE_ENGINEERING` | enabled | `strategy: "none"\|"default"\|...` |
| Model Training | `MODEL_TRAINING` | enabled | — |
| Calibration | `CALIBRATION` | enabled | `method: "none"\|"platt"\|"isotonic"` |
| Evaluation | `EVALUATION` | enabled | — |
| HPO | `HPO` | disabled | — |

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
 │     quality_cleaned_df   (Hamilton config:
 │              │            run_quality_audit,
 │              │            auto_clean_noise,
 │              │            noise_quality_threshold)
 │              │
 │     ┌────────┴────────┐
 │     │                 │
 │  adapter_result    feature_names
 │     │
 │  leakage_report ──→ (Hamilton config: run_leakage_audit, task)
 │     │
 │  target_transform_result ──→ (Hamilton config: target_transform,
 │     │                        target_skewness_threshold)
 │     │
 │  data_prep_result
 │     │
 │  models_to_run ──→ (config_models, vram_gb, task)
 │     │
 │  baseline_models → baseline_scores
 │     │                 │
 │  training_features ──┤  ← @config.when(feature_strategy=...)
 │     │                 │
 │  training_results ←───┘  (per-model loop, Hamilton config: calibration)
 │     │
 │  training_state ──→ (leaderboard, registry, champion)
```

---

## Module Details

### 1. Preprocessing

**Source:** `engine/pipelines/nodes/prep.py`

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

**Source:** `engine/pipelines/nodes/prep.py`

These nodes use `@config.when()` variants resolved from `_resolve_hamilton_config()`, which reads step params from `PipelineSpec`.

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `validate_target` | `processed_dataframe`, `target_col` | `pl.DataFrame` |
| `quality_cleaned_df` | `validate_target`, `target_col` + Hamilton config (`run_quality_audit`, `auto_clean_noise`, `noise_quality_threshold`) | `tuple[df, cleaned, n_dropped]` |
| `adapter_result` | `quality_cleaned_df`, `target_col` | `(X, y)` numpy arrays |
| `feature_names` | `quality_cleaned_df`, `target_col` | `list[str]` |
| `leakage_report` | `adapter_result`, `task` + Hamilton config (`run_leakage_audit`) | `LeakageReport \| None` |
| `target_transform_result` | `adapter_result` + Hamilton config (`target_transform`, `target_skewness_threshold`) | `(y, transformer, method, skew_orig, skew_trans, applied)` |
| `data_prep_result` | `adapter_result`, `target_transform_result`, `feature_names`, `leakage_report`, `quality_cleaned_df` | `DataPrepResult` |

### 3. Model Selection

**Source:** `engine/pipelines/nodes/train.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `models_to_run` | `data_prep_result`, `task`, `vram_gb`, `config_models` | `list[str]` |

If `config_models="auto"`, uses `ModelSelector.select()`. Otherwise uses the provided list.

### 4. Baselines

**Source:** `engine/pipelines/nodes/train.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `baseline_models` | (none) | `dict[str, type]` |
| `baseline_scores` | `data_prep_result`, `baseline_models`, `task`, `cv_folds`, `cv_strategy`, `metrics` | `dict[str, dict[str, float]]` |

### 5. Feature Engineering (Config Variant)

**Source:** `engine/pipelines/nodes/features.py`

The features module is only loaded when `PipelineSpec.is_enabled(StepName.FEATURE_ENGINEERING)` is true. Uses `@config.when()` to activate different implementations:

| Node | Config Condition | Behavior |
|------|-----------------|----------|
| `training_features__default` | `feature_strategy != "auto"` and not in `afe` variants | Pass-through: returns `(X, feature_names)` from `data_prep_result` |
| `training_features__afe_enabled` | `afe_enabled == True` | Runs full AFE: top-K → interactions → pruning |

Config is set via `_resolve_hamilton_config()` which reads `feature_strategy` from `PipelineSpec.step_params(StepName.FEATURE_ENGINEERING)`.

### 6. Model Training

**Source:** `engine/pipelines/nodes/train.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `training_results` | `training_features`, `data_prep_result`, `models_to_run`, `baseline_scores`, `task`, `cv_folds`, `cv_strategy`, `metrics`, `workspace`, `run_id` + Hamilton config (`calibration`) | `list[ModelResult]` |

For each model in `models_to_run` (excluding baselines):
1. Resolve class via `get_model_class(name)`
2. Cross-validate via `Evaluator.evaluate()`
3. Fit on full data (with optional calibration via Hamilton config)
4. Save artifact
5. Compute lift over baselines

### 7. State Generation

**Source:** `engine/pipelines/nodes/train.py`

| Node | Input Dependencies | Output |
|------|-------------------|--------|
| `training_state` | `training_results`, `baseline_scores`, `metrics`, `run_id`, `experiment_name`, `task`, `workspace` | `TrainingState` |

Aggregates all results, builds leaderboard, promotes champion to registry.

---

## Drift Detection Pipeline

**Source:** `engine/pipelines/nodes/drift_detection.py`

Uses `@config.when(drift_method=...)` for three variants:

| Node | Config | Method |
|------|--------|--------|
| `drift_report__psi` | `drift_method="psi"` | PSI drift detection |
| `drift_report__domain` | `drift_method="domain_classifier"` | Domain classifier AUC |
| `drift_report__both` | `drift_method="both"` | Both methods, merged report |

Config is set via `builder.with_config({"drift_method": "psi"})`.

---

## Tracking Hooks

**Source:** `src/iter8ml/engine/pipelines/hooks/tracking_hook.py`

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

**Source:** `src/iter8ml/config.py`

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
| `pipeline` | `PipelineSpec()` (all steps enabled except HPO) | Pipeline step configuration (see below) |
| `drift_detection` | `"psi"` | Drift method: `"none"`, `"psi"`, `"domain_classifier"`, `"both"` |
| `shap_enabled` | `False` | Enable SHAP explainability |
| `max_workers` | 1 | Concurrent model training (auto-reduced to 1 for low-VRAM GPUs) |
| `tracker` | `JSONL` | Tracker backend: `JSONL`, `WANDB`, `MLFLOW` |

### `PipelineSpec` — Step Configuration

Pipeline behavior is controlled via the `pipeline` field on `ExperimentConfig`:

```python
from iter8ml import ExperimentConfig, PipelineSpec, PipelineStep, StepName

config = ExperimentConfig(
    name="my_experiment",
    task="classification",
    target_col="label",
    data_path="data.csv",
    pipeline=PipelineSpec(steps=[
        PipelineStep(name=StepName.DATA_PREP),
        PipelineStep(name=StepName.QUALITY_AUDIT, params={"auto_clean_noise": True, "noise_quality_threshold": 0.5}),
        PipelineStep(name=StepName.LEAKAGE_AUDIT),
        PipelineStep(name=StepName.TARGET_TRANSFORM, params={"method": "auto", "skewness_threshold": 1.0}),
        PipelineStep(name=StepName.FEATURE_ENGINEERING, params={"strategy": "auto"}),
        PipelineStep(name=StepName.MODEL_TRAINING),
        PipelineStep(name=StepName.CALIBRATION, params={"method": "platt"}),
        PipelineStep(name=StepName.EVALUATION),
    ]),
)
```

Disable a step by setting `enabled=False`:

```python
PipelineStep(name=StepName.LEAKAGE_AUDIT, enabled=False)
```

Query at runtime:

```python
config.pipeline.is_enabled(StepName.CALIBRATION)  # True
config.pipeline.step_params(StepName.TARGET_TRANSFORM)  # {"method": "auto", "skewness_threshold": 1.0}
```

Inspect or visualize:

```python
from iter8ml.engine.pipelines import describe_pipeline, visualize_pipeline

steps = describe_pipeline(config.pipeline)  # list[dict] with name, enabled, params
graph = visualize_pipeline(spec=config.pipeline)  # Mermaid diagram
```

---

## How to Extend the DAG

### Adding a New Node

1. Create a function in the appropriate module under `engine/pipelines/nodes/`
2. Its parameters are automatically resolved as dependencies by Hamilton
3. The function name becomes the node name in the DAG

```python
# In engine/pipelines/nodes/my_module.py
def my_custom_feature(processed_dataframe: pl.DataFrame) -> pl.DataFrame:
    # processed_dataframe is auto-resolved from the preprocessing module
    return processed_dataframe.with_columns(...)
```

### Adding a Config Variant

Use `@config.when()` to create conditional node implementations. These are resolved from `_resolve_hamilton_config()`, which reads step params from `PipelineSpec`:

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

The config key is set automatically by `_resolve_hamilton_config()` reading from `PipelineSpec.step_params()`. To add a new config key, add it to `_resolve_hamilton_config()` in `executor.py` and document it as a param on the corresponding `PipelineStep`.

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
from iter8ml.engine.pipelines import visualize_pipeline

# Spec-aware diagram (annotates disabled steps)
graph = visualize_pipeline(spec=config.pipeline)
print(graph)

# Full Hamilton DAG (requires Hamilton installed)
from iter8ml.engine.pipelines.executor import PipelineExecutor
executor = PipelineExecutor()
print(executor.get_mermaid_graph())
```

Or inspect the pipeline steps programmatically:

```python
from iter8ml.engine.pipelines import describe_pipeline

for step in describe_pipeline(config.pipeline):
    print(f"  {step['step']}: enabled={step['enabled']}, params={step['params']}")
```

This is also logged automatically in the `experiment_started` event under `pipeline_lineage`.
