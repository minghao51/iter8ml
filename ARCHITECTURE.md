# Architecture

## Overview

Tabular Blueprint is a single-node tabular ML framework with a CLI-first workflow.
Core code lives under `src/tabular_blueprint` and follows a thin orchestration model:
- `cli.py` handles user entrypoints.
- `engine/` coordinates run orchestration, evaluation, HPO, tracking, and state generation.
- `data/` loads and prepares datasets (adapter, leakage, quality checks, feature engineering).
- `models/` contains model wrappers and selection logic.
- `monitoring/` provides drift and explainability primitives.
- `services/` manages reporting, registry, and export packaging.
- `pipelines/` defines Hamilton DAG nodes, hooks, and the multi-mode executor.

## Hamilton DAG Orchestration

The pipeline is orchestrated by `sf-hamilton` as a function-based DAG. Function
signatures define the dependency graph. `PipelineExecutor` builds mode-specific
drivers from node modules.

### Pipeline Modes (`PipelineMode`)

| Mode | Modules | Terminal Node |
|------|---------|---------------|
| `TRAINING` | preprocessing, data_preparation, model_selection, baselines, feature_engineering, model_training, state_generation | `training_state` |
| `DRIFT` | preprocessing, drift_detection | `drift_report` |
| `EXPORT` | preprocessing (via shipped predictor) | `processed_dataframe` |
| `HPO` | preprocessing | `processed_dataframe` |
| `INFERENCE` | preprocessing | `processed_dataframe` |

### Node Modules

- **preprocessing** (9 nodes): null imputation, date decomposition, categorical encoding.
- **data_preparation** (7 nodes): target validation, quality cleaning, adapter transform, leakage detection, target transform.
- **model_selection** (1 node): auto or explicit model list selection.
- **baselines** (2 nodes): naive + linear baseline evaluation.
- **feature_engineering** (1 node, conditional): passthrough or AFE via `@config.when(afe_enabled=True)`.
- **model_training** (1 node): sequential training loop producing `list[ModelResult]`.
- **state_generation** (1 node): terminal node with leaderboard + registry update.
- **drift_detection** (conditional): PSI / domain classifier / both via `@config.when(drift_method=...)`.

### Hooks

- **TrackingHook**: `NodeExecutionHook` adapting the `Tracker` protocol. Registered via `with_adapters()` on the driver builder. Logs `node_completed` and `node_error` events.

## Data and Training Flow

1. CLI or MCP tool builds `ExperimentConfig`.
2. `Trainer.run()` tries Hamilton DAG first (`_try_hamilton_training()`), falls back to imperative path.
3. `DataPreparationService` runs preprocessing + data preparation via Hamilton, with `TrackingHook` for lifecycle events.
4. Model training, baseline evaluation, and state generation execute as DAG nodes.
5. `DriftChecker` runs as standalone `DRIFT` mode DAG.

## Export

- Exported predictor packages include `model.artifact`, `metadata.json`, `predictor.py`, and `pipelines/preprocessing.py`.
- The predictor builds a Hamilton driver from the shipped preprocessing module, guaranteeing parity between training-time and export-time preprocessing.
- Falls back to direct function calls if Hamilton is not installed at inference time.

## Persistence and Interfaces

- Experiment/events are appended to `workspace/experiments.jsonl`.
- Champion metadata is stored in `workspace/registry.json` with file locking.
- Optional integrations (`wandb`, `mlflow`, `llm`, `mcp`) are additive and controlled by extras/config.

## Guardrails

- Safe deserialization uses a restricted unpickler allowlist.
- HPO warmstart uses historical completion events with additive `params` fields.
- Metric directionality and registry promotion logic are centralized in `services/report_service.py`.
