# Session Handoff Plan

## Slug

`hamilton-dag-phase5-export-hooks`

## Readable Summary

Hamilton DAG Enhancement: Phase 5 (Export Consistency + Tracking Hooks) + ARCHITECTURE.md Update

## 1. Primary Request and Intent

The user wants to enhance Hamilton's role in the `iter8ml` (tabular-blueprint) project. Hamilton now orchestrates the entire training pipeline as a DAG. Phases 1-4 are **complete**. Phase 5 (Export Consistency + Tracking Hooks) is partially done. The remaining work is:

- **Export predictor template**: Rewrite `PREDICTOR_TEMPLATE` in `export_service.py` to use Hamilton driver instead of calling preprocessing functions directly (lines 43-61 currently bypass the DAG)
- **Remove manual tracking from data_preparation.py**: The `_log_hamilton_events()` method should be replaced by the `TrackingHook` (already created)
- **Export integration test**: Verify parity between training-time and export-time preprocessing
- **Update `ARCHITECTURE.md`**: Reflect new pipeline structure per `AGENTS.md` rules

## 2. Key Technical Concepts

- **Hamilton DAG orchestration** (`sf-hamilton>=1.70`): Function-based DAG where function signatures define the dependency graph
- **`@config.when()` / `@config.when_not()`**: Build-time DAG shape variation based on config dict. Used for AFE conditional (`afe_enabled`) and drift method selection (`drift_method`)
- **`PipelineMode` enum**: `TRAINING`, `DRIFT`, `EXPORT`, `HPO`, `INFERENCE` — controls which modules are loaded
- **`PipelineExecutor`**: Multi-mode driver builder. Constructor builds preprocessing-only driver; `run_training()` and `run_drift()` build fresh drivers with mode-specific modules
- **`TrackingHook(NodeExecutionHook)`**: Lifecycle hook adapting `Tracker` protocol. Registered via `with_adapters()` on driver builder
- **`ModelResult`**: Has both `model_name` (display name like "CatBoost") and `input_name` (input name like "catboost") — critical for dict key parity with imperative path
- **Graceful fallback**: When Hamilton not installed, returns original df / empty dict / None. Imperative path preserved as fallback in `Trainer._run_imperative()`

## 3. Files and Code Sections

### Phase 3 — Created Files

#### `src/tabular_blueprint/pipelines/nodes/model_selection.py`
- **Why important**: Wraps `ModelSelector.select()` as a pure Hamilton node. Derives `n_rows` from `data_prep_result` instead of requiring it as a separate input.
- Key function: `models_to_run(data_prep_result, task, vram_gb, config_models, include_baselines=True) → list[str]`
- `config_models` accepts `list[str] | str` (`Any` type) — passes through explicit model lists, falls back to auto-selection

#### `src/tabular_blueprint/pipelines/nodes/baselines.py`
- **Why important**: Evaluates naive + linear baselines as Hamilton nodes
- Key function: `baseline_scores(data_prep_result, baseline_models, task, cv_folds, cv_strategy, metrics) → dict`
- Creates temporary `ExperimentConfig` + `Evaluator` internally

#### `src/tabular_blueprint/pipelines/nodes/feature_engineering.py`
- **Why important**: AFE conditional via `@config.when()`. Two variants resolve to same node name `training_features`.
- `training_features__default` with `@config.when_not(afe_enabled=True)` — passthrough
- `training_features__afe_enabled` with `@config.when(afe_enabled=True)` — runs full AFE pipeline
- Uses `contextlib.suppress(ImportError)` for Hamilton import

#### `src/tabular_blueprint/pipelines/nodes/model_training.py`
- **Why important**: Sequential model training loop replacing `ThreadPoolExecutor`. Originally tried `Parallelizable[T]`/`Collect[T]` but Hamilton's default executor doesn't support them.
- Key class: `ModelResult(model_name, input_name, cv_scores, artifact_path, duration_seconds, lift_over_baselines, error)`
- Key function: `training_results(training_features, data_prep_result, models_to_run, ...) → list[ModelResult]`
- Skips baseline model names (naive_baseline, linear_baseline) from training loop
- `_train_one()`: evaluate → fit → save → compute lift over baselines

#### `src/tabular_blueprint/pipelines/nodes/state_generation.py`
- **Why important**: Terminal node producing `TrainingState` with leaderboard + registry update
- Key class: `TrainingState(results, leaderboard, best_model, best_score, best_metric)`
- Uses `input_name` as dict key (lowercase "catboost"), stores `model_name` (display "CatBoost") in entry values
- Auto-updates `RegistryService` with best model

#### `tests/unit/test_training_nodes.py`
- 10 tests: model_selection (4), model_result (2), state_generation (4)
- Uses `MockDataPrepResult` with `n_rows`, `X`, `y`, `feature_names`

### Phase 3 — Modified Files

#### `src/tabular_blueprint/pipelines/executor.py`
- **Critical design**: Constructor always builds preprocessing-only driver. `run_training()` and `run_drift()` build fresh drivers with mode-specific modules.
- Added `_get_training_modules()` returning all 7 modules (preprocessing, data_preparation, model_selection, baselines, feature_engineering, model_training, state_generation)
- Added `run_training(...)` with ~25 explicit input parameters. Builds driver with `with_config({"afe_enabled": afe_enabled})`
- Added `tracker` parameter to constructor. In `run_training()`, creates `TrackingHook` and registers via `with_adapters()`
- `_MODE_FINAL_VARS[TRAINING]` kept as `["processed_dataframe"]` (not `["training_state"]`) to not break preprocessing-only usage

#### `src/tabular_blueprint/engine/trainer.py`
- `Trainer.run()` now tries `_try_hamilton_training()` first, falls back to `_run_imperative()`
- `_try_hamilton_training()`: creates `PipelineExecutor(mode=TRAINING)`, calls `run_training()`, catches all exceptions
- `_log_hamilton_state_events()`: logs experiment_started + model_completed/model_failed events from Hamilton DAG results. Uses `entry.get("model_name", model_name)` for display name in tracking events
- Drift removed from both paths (moved to Phase 4 standalone mode)
- `_run_imperative()`: extracted from old `run()`, preserves exact original logic

### Phase 4 — Created Files

#### `src/tabular_blueprint/pipelines/nodes/drift_detection.py`
- **Why important**: Standalone drift detection as Hamilton DAG mode
- `reference_df_input` / `live_df_input`: passthrough nodes for the two DataFrames
- `reference_features` / `live_features`: select numeric columns
- `psi_drift_report__psi` with `@config.when(drift_method="psi")` — wraps `PSIDriftDetector`
- `domain_drift_report__domain` with `@config.when(drift_method="domain_classifier")` — wraps `DomainClassifierDriftDetector`
- Both variants for `@config.when(drift_method="both")`
- `drift_report__*` terminal nodes combining results into `DriftReport` dataclass

#### `tests/unit/test_drift_nodes.py`
- 9 tests: node unit tests (3), dataclass test (1), DAG integration tests (5)

### Phase 4 — Modified Files

#### `src/tabular_blueprint/pipelines/executor.py`
- Added `run_drift(reference_df, live_df, drift_method) → DriftReport`
- Builds driver with preprocessing + drift_detection modules, config `{"drift_method": drift_method}`
- `_MODE_FINAL_VARS[DRIFT]` set to `["drift_report"]`

#### `src/tabular_blueprint/engine/trainer.py`
- Removed `self._drift.check(df, run_id)` from both Hamilton and imperative paths

#### `src/tabular_blueprint/cli.py`
- `drift` command: for psi/domain/both methods, tries `executor.run_drift()` first; falls back to direct detector instantiation. KS method unchanged.

### Phase 5 — Created Files

#### `src/tabular_blueprint/pipelines/hooks/tracking_hook.py`
- **Why important**: Adapts `Tracker` protocol to Hamilton lifecycle hooks
- `TrackingHook.__init__(tracker, run_id=None)`
- `run_on_node_success()`: logs `{"event": "node_completed", "node": node_name, "duration_seconds": ...}`
- `run_on_node_error()`: logs `{"event": "node_error", "node": node_name, "error": ...}`
- `run_before_node_execution()` and `run_after_node_execution()`: no-ops

#### `src/tabular_blueprint/pipelines/executor.py` (Phase 5 updates)
- Constructor accepts optional `tracker` parameter
- `run_training()` creates `TrackingHook` and registers via `builder.with_adapters(hook)`

### Key Reference Files (Unmodified, needed for remaining Phase 5 work)

#### `src/tabular_blueprint/services/export_service.py` (196 lines)
- `PREDICTOR_TEMPLATE` lines 43-61 call preprocessing functions directly (`fill_nulls_numeric`, `fill_nulls_categorical`, `decomposed_dates_df`, `encoded_df`) — **MUST be rewritten** to use Hamilton driver
- `_copy_preprocessing()` copies `preprocessing.py` to export package — needs to ship Hamilton driver config instead
- The exported predictor should construct a Hamilton driver and call `execute(["processed_dataframe"], inputs={"df": df})` for guaranteed parity

#### `src/tabular_blueprint/engine/data_preparation.py` (183 lines)
- `_log_hamilton_events()` (lines 78-105): manual tracking that should be replaced by `TrackingHook`
- `_prepare_via_hamilton()` (lines 45-76): calls `executor.run_data_prep()` then `_log_hamilton_events()`

## 4. Problem Solving

- **Constructor loaded all training modules**: Broke preprocessing-only tests. Fixed: constructor always builds preprocessing-only driver; `run_training()` builds fresh driver.
- **`_MODE_FINAL_VARS[TRAINING]` set to `["training_state"]`**: Broke `execute()` on preprocessing-only driver. Fixed: kept as `["processed_dataframe"]`.
- **`Parallelizable[T]`/`Collect[T]`**: Required special executor (`InvalidExecutorException`). Fixed: rewrote as sequential loop. Future optimization: add `MultiThreadingExecutor` or `RayTaskSource`.
- **`n_rows` not passed**: `models_to_run()` needed `n_rows` as separate input. Fixed: derive from `data_prep_result.n_rows`.
- **Model name casing**: "CatBoost" vs "catboost" as dict keys broke MCP tests. Fixed: added `input_name` field to `ModelResult`, use as dict key.
- **Leaderboard empty after Hamilton path**: Tracking events not logged. Fixed: added `_log_hamilton_state_events()` in trainer.
- **`config_models` list converted to "auto"**: MCP tools pass explicit model lists. Fixed: pass through directly, accept `Any` type.

## 5. Pending Tasks

1. **Export predictor template**: Update `PREDICTOR_TEMPLATE` in `export_service.py` to use Hamilton driver for preprocessing instead of calling functions directly
2. **Export `_copy_preprocessing()`**: Ship Hamilton driver config in export package
3. **Remove manual tracking from data_preparation.py**: Replace `_log_hamilton_events()` with `TrackingHook` registration in `run_data_prep()`
4. **Export integration test**: Verify parity between training-time and export-time preprocessing
5. **Update `ARCHITECTURE.md`**: Reflect new pipeline structure per `AGENTS.md` rules

## 6. Current Work

Phase 5 was started but not completed. The `TrackingHook` was created and registered in `run_training()`. The remaining work is the export service rewrite and cleanup.

**Test status**: 294 passed, 3 failed (all pre-existing TabPFN GPU tests). All lint clean.

The Hamilton DAG now covers:
- **Preprocessing** (9 nodes): null imputation → date decomposition → categorical encoding
- **Data Preparation** (7 nodes): target validation → quality cleaning → adapter transform → leakage detection → target transform → composed result
- **Model Selection** (1 node): auto or explicit model list
- **Baselines** (2 nodes): baseline_models dict + baseline_scores evaluation
- **Feature Engineering** (1 node, conditional): passthrough or AFE with `@config.when(afe_enabled=True)`
- **Model Training** (1 node): sequential training loop producing `list[ModelResult]`
- **State Generation** (1 node): terminal node with leaderboard + registry update
- **Drift Detection** (multiple conditional nodes): PSI / domain classifier / both via `@config.when(drift_method=...)`
- **Tracking Hook**: Lifecycle hook registered via `with_adapters()`

## 7. Next Step

**Phase 5 remaining work**: Rewrite `PREDICTOR_TEMPLATE` in `export_service.py` to use Hamilton driver for preprocessing. The key change is replacing lines 43-61 (direct function calls) with a Hamilton driver that executes `["processed_dataframe"]`. Then update `_copy_preprocessing()` to ship the necessary Hamilton config, add an export integration test, remove `_log_hamilton_events()` from `data_preparation.py`, and update `ARCHITECTURE.md`.
