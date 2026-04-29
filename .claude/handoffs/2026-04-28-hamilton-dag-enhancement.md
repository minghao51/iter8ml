# Session Handoff Plan

## Slug

`hamilton-dag-enhancement`

## Readable Summary

Hamilton DAG Enhancement: Phases 3-5 (Model Training, Drift Detection, Export Consistency)

## 1. Primary Request and Intent

The user wants to enhance Hamilton's role in the `iter8ml` (tabular-blueprint) project. Hamilton currently orchestrates only a 7-node preprocessing chain (~3% of the pipeline). The goal is to extend it to orchestrate the **entire training pipeline** as a DAG with:

- **Multi-mode execution**: Same DAG, different `final_vars` for training / drift / export / HPO / inference
- **Declarative parallelism**: Replace ad-hoc `ThreadPoolExecutor` with Hamilton's `Parallelizable[T]`/`Collect[T]`
- **Drift detection as separate production mode**: Not part of training — standalone execution path sharing only preprocessing nodes
- **Export consistency**: Fix exported predictors that bypass the DAG by calling functions directly
- **Configurable pipelines**: Use `@config.when()` for conditional steps, `overrides` for injecting fitted artifacts

Phases 1 and 2 are **complete**. Phases 3, 4, and 5 remain.

## 2. Key Technical Concepts

- **Hamilton DAG orchestration** (`sf-hamilton>=1.70`): Function-based DAG where function signatures define the dependency graph
- **`driver.execute(final_vars, inputs)`**: Only computes the minimum subgraph needed to produce the requested terminal nodes
- **`@config.when()` / `@config.when_not()`**: Build-time DAG shape variation based on config dict
- **`Parallelizable[T]` / `Collect[T]`**: Hamilton's map-reduce semantics for parallel execution
- **`overrides`**: Inject pre-computed values (e.g., fitted scaler) to skip re-computation
- **`@subdag`**: Compose reusable modules as sub-graphs
- **`NodeExecutionHook`**: Lifecycle hooks for side-effects (tracking, logging)
- **`PipelineMode` enum**: `TRAINING`, `DRIFT`, `EXPORT`, `HPO`, `INFERENCE` — controls which modules are loaded
- **PipelineExecutor**: New class that builds mode-specific Hamilton drivers with graceful fallback when Hamilton not installed
- **Polars-native**: All preprocessing uses `pl.DataFrame` with `pl.Expr` operations
- **`DataPrepResult`**: Rich dataclass bridging Polars world and NumPy world (Polars→NumPy boundary)

## 3. Files and Code Sections

### Created Files

#### `src/tabular_blueprint/pipelines/nodes/__init__.py`
- Empty init for nodes package

#### `src/tabular_blueprint/pipelines/nodes/preprocessing.py`
- **Why important**: Fixed preprocessing DAG nodes. Two bugs fixed: `null_filled_df` now properly merges numeric+categorical fills (was a passthrough), `decomposed_dates_df` now drops original date columns.
- Key functions: `raw_dataframe`, `numeric_columns`, `categorical_columns`, `date_columns`, `fill_nulls_numeric`, `fill_nulls_categorical`, `null_filled_df`, `decomposed_dates_df`, `encoded_df`, `processed_dataframe`

#### `src/tabular_blueprint/pipelines/nodes/data_preparation.py`
- **Why important**: Data preparation as pure Hamilton nodes. Replaces the imperative `DataPreparationService.prepare()` logic.
- Key functions: `validate_target`, `quality_cleaned_df`, `adapter_result`, `feature_names`, `leakage_report`, `target_transform_result`, `data_prep_result`
- All config is passed as explicit inputs (not via `ExperimentConfig`), making nodes pure
- `DataPrepResult` here has more fields than the engine version: `target_transform_method`, `target_original_skewness`, `target_transformed_skewness`, `target_transform_applied`, `noise_cleaned`, `n_noise_dropped`

```python
@dataclass
class DataPrepResult:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    leakage_report: LeakageReport | None
    target_transformer: Any | None
    n_rows: int
    n_features: int
    target_transform_method: str
    target_original_skewness: float
    target_transformed_skewness: float
    target_transform_applied: bool
    noise_cleaned: bool
    n_noise_dropped: int
```

#### `src/tabular_blueprint/pipelines/executor.py`
- **Why important**: Core multi-mode driver builder. Replaces the old `HamiltonExecutor`.
- `PipelineMode` enum: `TRAINING`, `DRIFT`, `EXPORT`, `HPO`, `INFERENCE`
- `PipelineExecutor` class with `run_preprocessing()` and `run_data_prep()` methods
- `_MODE_MODULES` dict (currently empty for all modes — to be populated in Phases 3-4)
- `_MODE_FINAL_VARS` dict (currently `["processed_dataframe"]` for all modes — to be updated)
- Graceful fallback: when Hamilton not installed, returns original df / empty dict

#### `src/tabular_blueprint/pipelines/hooks/__init__.py`
- Empty init for hooks package (for Phase 5)

#### `tests/unit/test_pipeline_executor.py`
- 12 tests: 8 for normal execution, 4 for fallback without Hamilton (uses monkeypatch)

#### `tests/unit/test_data_prep_nodes.py`
- 12 tests covering: validate_target (valid/invalid), adapter_result, feature_names, leakage_report (enabled/disabled), target_transform (none/log1p), data_prep_result (full/with_leakage/with_transform)

### Modified Files

#### `src/tabular_blueprint/pipelines/preprocessing.py`
- Now re-exports all functions from `nodes.preprocessing.py` for backward compat
- Tests and export service that `from tabular_blueprint.pipelines import preprocessing` still work

#### `src/tabular_blueprint/pipelines/hamilton_executor.py`
- Deprecated. Delegates to `PipelineExecutor`. Issues `DeprecationWarning` on `run()`.

#### `src/tabular_blueprint/pipelines/__init__.py`
- Exports `PipelineExecutor`, `PipelineMode`, `HamiltonExecutor` (deprecated), `visualize_pipeline()`

#### `src/tabular_blueprint/engine/trainer.py`
- Changed import from `HamiltonExecutor` to `PipelineExecutor`
- `self.executor = PipelineExecutor()`
- `df = self.executor.run_preprocessing(df)` guarded by `if self.executor.available`

#### `src/tabular_blueprint/engine/data_preparation.py`
- `DataPreparationService.prepare()` now tries Hamilton first via `executor.run_data_prep()`
- Falls back to `_prepare_imperative()` if Hamilton unavailable
- Tracking events extracted to `_log_hamilton_events()` to keep nodes pure
- `_prepare_imperative()` preserves the original logic exactly

#### `pyproject.toml`
- `sf-hamilton>=1.70` moved from core `dependencies` to `[project.optional-dependencies]` as `hamilton = ["sf-hamilton>=1.70"]`
- `all` extra now includes `hamilton`

#### `tests/unit/test_processors.py`
- Rewritten to import from `nodes.preprocessing`
- 9 tests including new: `test_fill_nulls_string_columns`, `test_decompose_dates_drops_original`, `test_null_filled_df_merges_columns`, `test_full_pipeline_with_strings`, `test_no_dates_passthrough`

### Key Reference Files (Unmodified, needed for Phase 3+)

#### `src/tabular_blueprint/engine/model_trainer.py` (331 lines)
- `ModelTrainer` class with `run_baselines()`, `train_all()`, `_train_sequential()`, `_train_concurrent()`, `_train_single_model()`
- Uses `ThreadPoolExecutor` for concurrent mode — target for `Parallelizable[T]` replacement
- `_train_single_model()` does: evaluate → fit → save → registry update

#### `src/tabular_blueprint/engine/feature_engineer.py` (115 lines)
- `FeatureEngineer.run_afe()`: fit importance model → extract top-k → discover interactions → augment → optional pruning
- Conditional via `@config.when(afe_enabled=True)` in Phase 3

#### `src/tabular_blueprint/models/selector.py`
- `ModelSelector.select(n_rows, task, vram_gb)` → `list[str]` (hardware-aware model routing)

#### `src/tabular_blueprint/engine/drift_checker.py` (59 lines)
- Currently splits training data 80/20 (misleading) — Phase 4 makes it a standalone mode
- PSI + domain classifier methods

#### `src/tabular_blueprint/services/export_service.py` (196 lines)
- `PREDICTOR_TEMPLATE` calls preprocessing functions directly (lines 51-60) — Phase 5 fixes this
- `_copy_preprocessing()` copies `preprocessing.py` to export package

#### `src/tabular_blueprint/engine/tracker.py`
- `Tracker` protocol with `log_event()`, `log_metrics()`, `finish()`
- `JSONLTracker` implementation — Phase 5 wraps this in a `NodeExecutionHook`

## 4. Problem Solving

- **`null_filled_df` bug**: Was just returning `fill_nulls_categorical` without merging. Fixed to properly select and concat numeric + categorical + other columns
- **Date columns not dropped**: `decomposed_dates_df` now drops original date columns after creating year/month/day/weekday features
- **Hamilton import**: Made optional with graceful fallback — `PipelineExecutor.available` property and monkeypatch-based tests
- **DataPrepResult type mismatch**: Hamilton nodes return a richer `DataPrepResult` than the engine version. Bridge via `_log_hamilton_events()` extracting metadata for tracking
- **Test for validate_target**: Failed initially because `validate_target` receives the preprocessed DataFrame (transformed by Hamilton), not the raw input. Fixed assertion to check columns/rows instead of exact equality

## 5. Pending Tasks

### Phase 3: Model Selection + Training Sub-DAG
1. Create `nodes/model_selection.py` — `select_models(n_rows, task, vram_gb, config_models) → list[str]`
2. Create `nodes/baselines.py` — `run_baselines(X, y, evaluator, ...) → dict[str, dict]`
3. Create `nodes/feature_engineering.py` — AFE nodes with `@config.when(afe_enabled=True)` / `@config.when_not(afe_enabled=True)`
4. Create `nodes/model_training.py` — `Parallelizable[str]` / `Collect[ModelResult]` for per-model fan-out. Replace `ThreadPoolExecutor`
5. Enable `MultiThreadingExecutor` on driver when `max_workers > 1`
6. Create `nodes/state_generation.py` — terminal node producing state + leaderboard
7. Update `_MODE_MODULES[TRAINING]` and `_MODE_FINAL_VARS[TRAINING]` in executor.py
8. Integration test with small dataset running full training DAG

### Phase 4: Drift Detection as Separate Mode
1. Create `nodes/drift_detection.py` — `reference_features`, `live_features`, `psi_drift_report`, `domain_drift_report`, `combined_drift_report`
2. PSI and domain classifier as parallel nodes
3. Use `@config.when(drift_method="psi")` / `"domain_classifier"` / `"both"`
4. Update `_MODE_MODULES[DRIFT]` and `_MODE_FINAL_VARS[DRIFT]`
5. Remove drift from `Trainer.run()` (currently at `trainer.py:127-128`)
6. Wire `cli.py:drift` command to Hamilton driver: `dr.execute(["drift_report"], inputs={"reference_df": ref, "live_df": live})`
7. Add `overrides` support for fitted preprocessing state
8. Tests for drift nodes independently + CLI integration

### Phase 5: Export Consistency + Tracking Hooks
1. Update `ExportService._copy_preprocessing()` — ship Hamilton driver config in export package
2. Update `PREDICTOR_TEMPLATE` — use Hamilton driver with `mode="export"` instead of calling functions directly
3. Create `hooks/tracking_hook.py` — adapt `Tracker` protocol to `NodeExecutionHook`
4. Remove manual `tracker.log_event()` from service logic
5. Optional: Add `HamiltonTracker` for Hamilton UI
6. Export integration test verifying parity

### Also Needed
- Update `ARCHITECTURE.md` to reflect new structure (per `AGENTS.md` rules)

## 6. Current Work

Phases 1 and 2 are fully implemented and tested. The last action was running the full test suite after Phase 2 completion: **249 passed, 0 failed, all lint clean**.

The Hamilton DAG now covers:
- **Preprocessing** (9 nodes): null imputation → date decomposition → categorical encoding
- **Data Preparation** (7 nodes): target validation → quality cleaning → adapter transform → leakage detection → target transform → composed result

The infrastructure is in place for Phase 3+:
- `PipelineExecutor` with `run_data_prep()` method showing the pattern for adding new DAG stages
- `_MODE_MODULES` and `_MODE_FINAL_VARS` dicts ready to be populated
- `nodes/` and `hooks/` directories created

## 7. Next Step

**Phase 3: Model Selection + Training Sub-DAG** — Start with creating `nodes/model_selection.py` converting `ModelSelector.select()` to a Hamilton node, then `nodes/baselines.py`, then `nodes/model_training.py` with `Parallelizable[T]` for concurrent model training. This directly continues the approved plan.

Key patterns to follow from Phase 2:
- Pure functions with explicit inputs (no `ExperimentConfig` dependency)
- Config passed as input parameters to nodes
- Service classes delegate to Hamilton with imperative fallback
- Tests for individual nodes + composed pipeline
