# Architecture

## Overall Pattern: Thin Orchestration with Hamilton DAG

`tabular-blueprint` follows a **thin orchestration** model where `Trainer` in `engine/` coordinates the run lifecycle. The critical-path pipeline is driven by **sf-hamilton** as a function-based DAG — function signatures define the dependency graph and `PipelineExecutor` builds mode-specific drivers from node modules. An imperative fallback exists when Hamilton is unavailable.

## Layers

### 1. CLI Layer (`src/tabular_blueprint/cli.py`)
- Typer app (`tabblueprint`) registered in `pyproject.toml:59` as `[project.scripts]`
- Commands: `init`, `run`, `leaderboard`, `registry`, `hardware`, `drift`, `state`, `hpo`, `diff`, `export`
- `run` builds `ExperimentConfig` from file or CLI flags, calls `Trainer.run(df)`

### 2. MCP Layer (`src/tabular_blueprint/mcp/tools.py`)
- FastMCP server (`tabular-blueprint`) exposing atomic tools for LLM agents
- Tools: `get_experiment_state`, `get_column_stats`, `run_baseline`, `run_hpo`, `get_event_log`, `registry_show`, `registry_promote`, `detect_drift`, `export_champion`
- Each tool is self-contained: loads data, builds config, calls Trainer/pipeline

### 3. Config Layer (`config.py`, `constants.py`, `exceptions.py`)
- **ExperimentConfig** (Pydantic BaseModel at `config.py:19`): central experiment configuration with file loading (YAML/TOML/JSON/PY), enum serialization, task-defaults validator
- **HardwareProfile** (`config.py:136`): auto-detected GPU/CPU/RAM profile
- **Enums** (`constants.py`): `TaskType`, `CVStrategy`, `ModelName`, `EmbeddingMethod`, `TrackerType` — all `str` enums for JSON compatibility
- **Exceptions** (`exceptions.py`): `TabularBlueprintError` base, `DataLoadError`, `ModelFitError`, `RegistryError`; `track_errors()` decorator catches/logs/re-raises typed errors

### 4. Data Layer (`src/tabular_blueprint/data/`)
- **loaders.py**: `load_data()`, `load_csv()`, `load_parquet()`, `load_sqlite()`, `get_data_hash()`
- **adapter.py**: `DataAdapter` converts Polars DataFrame -> numpy arrays
- **feature_engine.py**: AFE (automatic feature engineering), interaction discovery, target transformation
- **leakage.py**: `LeakageReport` / `detect_leakage()` — flags suspiciously predictive features
- **quality.py**: `audit_data_quality()`, `clean_noise()` — data quality auditing and noise cleaning
- **cache.py**: `PreprocessingCache` — hash-based preprocessing cache on disk
- **embedding_engine.py**: high-cardinality categorical embedding utilities

### 5. Engine Layer (`src/tabular_blueprint/engine/`)
- **trainer.py** (`Trainer` at line 41): slim orchestrator. `run()` tries Hamilton DAG first (`_try_hamilton_training()`), falls back to `_run_imperative()`. Manages run_id, tracker lifecycle, event logging
- **data_preparation.py** (`DataPreparationService`): orchestrates preprocessing + noise cleaning + adapter transform + leakage detection + target transform via Hamilton or imperative
- **model_trainer.py** (`ModelTrainer`): baseline evaluation + model training (sequential or concurrent via `ThreadPoolExecutor`), calibration, champion update
- **evaluator.py** (`Evaluator`): cross-validation evaluation (CV loop)
- **tracker.py**: `Tracker` protocol + `JSONLTracker` (default, with log rotation), `WandbTracker`, `MLflowTracker`
- **hpo.py**: Optuna hyperparameter optimization
- **hpo_warmstart.py**: injects historical trials from JSONL log
- **hpo_importance.py**: parameter importance (PedAnova)
- **calibration.py**: `CalibratedModel` wrapper (Platt/Isotonic)
- **drift_checker.py**: standalone drift detection checker
- **embedding_trainer.py**: trains entity embeddings or autoencoders
- **explainability_service.py**: SHAP-based feature importance
- **feature_engineer.py**: AFE orchestration (imperative path)
- **state_observer.py**: generates `current_state.md` from logs + registry, optional LLM commentary

### 6. Pipeline Layer (`src/tabular_blueprint/pipelines/`)
- **executor.py** (`PipelineExecutor` at line 77): builds Hamilton drivers per mode. Modes via `PipelineMode` enum:
  - `TRAINING`: 7 node modules -> `training_state`
  - `DRIFT`: preprocessing + drift_detection -> `drift_report`
  - `EXPORT`/`HPO`/`INFERENCE`: preprocessing -> `processed_dataframe`
- **hamilton_executor.py**: deprecated wrapper around PipelineExecutor
- **hooks/tracking_hook.py** (`TrackingHook`): `NodeExecutionHook` adapts `Tracker` protocol -> Hamilton adapter. Logs `node_completed` / `node_error`

### 7. Pipeline Nodes (`src/tabular_blueprint/pipelines/nodes/`)
Each node module is a plain-Python module where function names define the DAG:
- **preprocessing.py**: 9 nodes — null imputation (numeric median, categorical mode), date decomposition (year/month/day/weekday), categorical encoding
- **data_preparation.py**: 7 nodes — target validation, quality cleaning, adapter transform, leakage detection, target transform -> `DataPrepResult`
- **model_selection.py**: 1 node — auto or explicit model list
- **baselines.py**: 2 nodes — naive + linear baseline evaluation
- **feature_engineering.py**: conditional nodes via `@config.when(afe_enabled=True)` / `@config.when(embedding_enabled=True)` — passthrough, AFE, or embedding
- **model_training.py**: 1 node -> `list[ModelResult]`, sequential training loop
- **state_generation.py**: terminal node -> `TrainingState` (results dict, leaderboard, best model, registry update)
- **drift_detection.py**: conditional nodes via `@config.when(drift_method=...)` — PSI, domain classifier, or both -> `DriftReport`

### 8. Models Layer (`src/tabular_blueprint/models/`)
- **base.py**: `AbstractModel` Protocol — `fit`, `predict`, `predict_proba`, `save`, `load`, `model_name`
- **factory.py**: string-keyed registry with lazy imports (`_MODEL_REGISTRY`), cached lookups
- **selector.py** (`ModelSelector`): hardware/data-size-aware model routing (TabPFN < 50k + GPU, CatBoost/LightGBM/XGBoost, FT-Transformer > 50k + 12GB VRAM, TabNet > 8GB VRAM)
- **baselines.py**: `NaiveBaseline`, `LinearBaseline`
- **gbdt_base.py**: shared GBDT base class
- **model_configs.py**: model hyperparameter configs
- **conventional/**: `CatBoostModel`, `LightGBMModel`, `XGBoostModel`
- **deep/**: `FTTransformerModel` (PyTorch), `TabNetModel` (pytorch-tabular), `SparseEmbedder`, `TextEncoder`
- **tabular_foundation/**: `TabPFNModel` (TabPFN v2)

### 9. Monitoring Layer (`src/tabular_blueprint/monitoring/`)
- **drift.py**: `DriftDetector` (KS test for numeric, chi2 for categorical)
- **psi_drift.py**: `PSIDriftDetector` (Population Stability Index)
- **domain_classifier.py**: `DomainClassifierDriftDetector` (train a classifier to distinguish reference vs live)
- **explainability.py**: SHAP-based explainer

### 10. Services Layer (`src/tabular_blueprint/services/`)
- **registry_service.py** (`RegistryService`): file-locked JSON registry for champion models. Thread/process-safe via `filelock`. Atomic saves via temp + rename. `update_if_better()`, `promote_run()`
- **report_service.py** (`ReportService`): builds `ExperimentReport` / `LeaderboardEntry` from JSONL events. Metric direction logic: `metric_higher_is_better()`, `metric_value_is_better()`, `resolve_primary_score()`
- **export_service.py** (`ExportService`): packages champion model + preprocessing nodes + predictor script into portable directory

### 11. LLM Layer (`src/tabular_blueprint/llm/__init__.py`)
- **TabularAgent**: natural language explanations for SHAP, performance, feature importance via `litellm`

### 12. Utils (`src/tabular_blueprint/utils/`)
- **jsonl.py**: `load_events()`, `iter_events()` — JSONL log reading
- **safe_pickle.py**: `RestrictedUnpickler` — allowlist-based safe deserialization (sklearn, numpy, scipy, catboost, lightgbm, xgboost, tabpfn, collections, builtins)

## Data Flow

1. CLI/MCP builds `ExperimentConfig` and loads data as `pl.DataFrame`
2. `Trainer.run(df)` creates run_id, tries Hamilton DAG
3. DAG executes: preprocessing -> data_preparation -> model_selection -> baselines -> feature_engineering -> model_training -> state_generation
4. `TrackingHook` logs `node_completed` / `node_error` events to Tracker
5. Terminal node `training_state` generates leaderboard, updates registry
6. StateObserver writes `current_state.md` and `leaderboard.md` to workspace
7. On Hamilton failure, `Trainer._run_imperative()` runs equivalent logic imperatively

## Guardrails

- **Safe deserialization** (`utils/safe_pickle.py`): `RestrictedUnpickler` blocks unpickling of non-allowlisted classes
- **HPO warmstart** (`engine/hpo_warmstart.py`): validates historical trials before injection
- **Metric direction** (`services/report_service.py`): `metric_value_is_better()` centralized for leaderboard ranking and registry promotion
- **Config safety** (`config.py`): `.py` config files disabled by default (`allow_unsafe_python=False`)
- **Model calibration** (`engine/calibration.py`): only applies to classification tasks
- **TabPFN guard**: warning emitted when n_rows > 50k

## Entry Points

| Entry Point | Mechanism | Location |
|---|---|---|
| CLI | `tabblueprint` console script | `pyproject.toml:59` -> `cli.py:18` |
| MCP Server | FastMCP | `mcp/tools.py:17` |
| Direct Python | `from tabular_blueprint import ...` | `__init__.py` |

## Key Files

| File | Purpose |
|---|---|
| `src/tabular_blueprint/cli.py` | Typer CLI entry point with 10 commands |
| `src/tabular_blueprint/config.py` | ExperimentConfig + HardwareProfile |
| `src/tabular_blueprint/constants.py` | Enums and conversion utilities |
| `src/tabular_blueprint/exceptions.py` | Exception hierarchy + track_errors decorator |
| `src/tabular_blueprint/engine/trainer.py` | Main orchestrator |
| `src/tabular_blueprint/pipelines/executor.py` | Hamilton driver builder per mode |
| `src/tabular_blueprint/engine/tracker.py` | Tracker protocol + implementations |
| `src/tabular_blueprint/models/base.py` | AbstractModel Protocol |
| `src/tabular_blueprint/models/factory.py` | Model class registry |
| `src/tabular_blueprint/services/registry_service.py` | File-locked champion registry |
| `src/tabular_blueprint/utils/safe_pickle.py` | Restricted unpickler |
