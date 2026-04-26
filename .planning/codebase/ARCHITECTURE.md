# Architecture

> Last updated: 2026-04-23

## Pattern

**Layered pipeline architecture** with a CLI-first interface. The system is structured as a sequential ML experimentation framework — not a web app — organized into clearly separated layers: CLI entry point, engine orchestration, data processing, model management, services, and monitoring.

The core pattern is a **train-evaluate-track loop**: data flows in from files, passes through a Hamilton DAG for preprocessing, gets split into NumPy arrays, trains multiple models with cross-validation, evaluates them, logs results as JSONL events, and persists champions to a file-backed model registry.

## Layers

### 1. Presentation Layer (CLI + MCP)
- **CLI** (`src/tabular_blueprint/cli.py`): Typer-based CLI with commands: `init`, `run`, `leaderboard`, `registry`, `hardware`, `drift`, `state`, `hpo`, `diff`, `export`
- **MCP Server** (`src/tabular_blueprint/mcp/tools.py`): FastMCP-based server exposing atomic tools for LLM agents (get_experiment_state, run_baseline, run_hpo, detect_drift, etc.)
- **LLM Agent** (`src/tabular_blueprint/llm/__init__.py`): LiteLLM-backed agent for natural language explanations of SHAP results and model performance

### 2. Engine / Orchestration Layer
- **Trainer** (`src/tabular_blueprint/engine/trainer.py`): Central orchestrator. Coordinates the full experiment lifecycle: preprocessing → data adaptation → leakage audit → baselines → optional AFE → model training (sequential or concurrent) → drift detection → state update
- **Evaluator** (`src/tabular_blueprint/engine/evaluator.py`): Cross-validation engine. Creates fresh model instances per fold, computes configurable metrics (roc_auc, f1_macro, rmse, r2, etc.)
- **Tracker** (`src/tabular_blueprint/engine/tracker.py`): Pluggable experiment tracking via `Tracker` Protocol. Three implementations: `JSONLTracker` (default, with log rotation), `WandbTracker`, `MLflowTracker`
- **HPO** (`src/tabular_blueprint/engine/hpo.py`): Optuna-based hyperparameter optimization with warmstart from historical JSONL events
- **StateObserver** (`src/tabular_blueprint/engine/state_observer.py`): Generates `current_state.md` and `leaderboard.md` after every run — LLM-readable experiment state summaries

### 3. Data Layer
- **Loaders** (`src/tabular_blueprint/data/loaders.py`): Polars-based ingestion from CSV, Parquet, and SQLite
- **DataAdapter** (`src/tabular_blueprint/data/adapter.py`): Converts Polars DataFrames to model-specific formats (NumPy, PyTorch tensors, HuggingFace Dataset)
- **Feature Engine** (`src/tabular_blueprint/data/feature_engine.py`): Target transformation (log1p, yeo-johnson, box-cox), interaction discovery (multiply, ratio), and feature pruning via permutation importance
- **Quality** (`src/tabular_blueprint/data/quality.py`): Cleanlab-based noise detection and cleaning
- **Leakage** (`src/tabular_blueprint/data/leakage.py`): Feature leakage detection
- **Preprocessing Pipeline** (`src/tabular_blueprint/pipelines/preprocessing.py`): Hamilton DAG — null filling → date decomposition → categorical encoding

### 4. Model Layer
- **AbstractModel** (`src/tabular_blueprint/models/base.py`): Protocol defining the model contract (`fit`, `predict`, `predict_proba`, `save`, `load`, `model_name`)
- **Factory** (`src/tabular_blueprint/models/factory.py`): Lazy-loading model registry mapping string names to `(module_path, class_name)` tuples
- **Selector** (`src/tabular_blueprint/models/selector.py`): Hardware/data-size-aware model routing (e.g., TabPFN only with GPU, FT-Transformer only with >12GB VRAM and >50k rows)
- **Model hierarchy**:
  - `BaseGBDTModel` → `CatBoostModel`, `LightGBMModel`, `XGBoostModel` (conventional/)
  - `TabPFNModel` (tabular_foundation/)
  - `FTTransformerModel`, `TabNetModel` (deep/)
  - `NaiveBaseline`, `LinearBaseline` (baselines.py)
- **Model Configs** (`src/tabular_blueprint/models/model_configs.py`): Pydantic models with default hyperparameters and HPO search spaces per model

### 5. Services Layer
- **RegistryService** (`src/tabular_blueprint/services/registry_service.py`): Thread-safe model registry with file locking (`fcntl`). Stores champions in `workspace/registry.json`, auto-promotes better-scoring models
- **ReportService** (`src/tabular_blueprint/services/report_service.py`): Builds structured experiment reports from JSONL logs + registry. Powers leaderboard generation (console and markdown)
- **ExportService** (`src/tabular_blueprint/services/export_service.py`): Packages champion models as portable prediction directories (model artifact + preprocessing pipeline + predictor script + metadata)

### 6. Monitoring Layer
- **Drift Detection** (`src/tabular_blueprint/monitoring/drift.py`): KS-test (numeric) and Chi-squared (categorical) drift detection
- **PSI Drift** (`src/tabular_blueprint/monitoring/psi_drift.py`): Population Stability Index drift detection
- **Domain Classifier** (`src/tabular_blueprint/monitoring/domain_classifier.py`): Classifier-based drift detection
- **Explainability** (`src/tabular_blueprint/monitoring/explainability.py`): SHAP-based feature importance with plot generation

## Data Flow

```
CLI / MCP Tool
    │
    ▼
Trainer.run(df: Polars DataFrame)
    │
    ├── HamiltonExecutor.run(df) ── Hamilton DAG ──► preprocessed Polars DataFrame
    │
    ├── DataAdapter.transform(df) ──► (X: np.ndarray, y: np.ndarray)
    │
    ├── [Optional] Quality audit + noise cleaning
    ├── [Optional] Leakage detection
    ├── [Optional] Target transformation
    │
    ├── Baseline models (NaiveBaseline, LinearBaseline)
    │
    ├── [Optional] AFE: fit GBDT → permutation importance → interaction discovery → pruning
    │
    ├── ModelSelector.select() ──► ordered model list
    │
    ├── For each model:
    │   ├── Evaluator.evaluate(model_cls, X, y) ──► CV scores
    │   ├── model.fit(X, y)
    │   ├── [Optional] Calibration
    │   ├── model.save(artifact_path)
    │   ├── [Optional] SHAP explainability
    │   └── Tracker.log_event(...)
    │
    ├── [Optional] Drift detection (PSI / Domain Classifier)
    │
    ├── RegistryService.update_if_better() ──► champion promotion
    │
    └── StateObserver.generate() ──► workspace/current_state.md + leaderboard.md
```

## Entry Points

| Entry Point | File | Description |
|---|---|---|
| CLI `tabblueprint` | `src/tabular_blueprint/cli.py` | Main CLI entry point (registered as `tabblueprint` script in pyproject.toml) |
| MCP Server | `src/tabular_blueprint/mcp/tools.py` | FastMCP server for LLM agent integration |
| Programmatic API | `src/tabular_blueprint/engine/trainer.py:Trainer` | `Trainer(config).run(df)` for library usage |
| Example configs | `examples/credit_risk.py`, `examples/zenml_pipeline.py` | Runnable experiment configurations |
| Docker | `Dockerfile`, `docker-compose.yml` | GPU-enabled container with MLflow sidecar |

## Abstractions

| Abstraction | Type | File | Purpose |
|---|---|---|---|
| `AbstractModel` | Protocol | `models/base.py` | Structural subtyping contract for all models |
| `Tracker` | Protocol | `engine/tracker.py` | Pluggable experiment tracking (JSONL, W&B, MLflow) |
| `BaseGBDTModel` | Abstract base class | `models/gbdt_base.py` | Shared GBDT behavior (fit/save/load/predict) |
| `ExperimentConfig` | Pydantic BaseModel | `config.py` | Single source of truth for all experiment settings |
| `HardwareProfile` | Pydantic BaseModel | `config.py` | Auto-detected hardware capabilities |
| `DataAdapter` | Class | `data/adapter.py` | Format-agnostic data conversion layer |
| `HamiltonExecutor` | Class | `pipelines/hamilton_executor.py` | DAG-based preprocessing orchestration |

## State Management

This is a **CLI tool / library**, not a web application. State is managed via:

1. **Experiment log** (`workspace/experiments.jsonl`): Append-only JSONL event log. Each run writes structured events (experiment_started, model_completed, baseline_completed, drift_check, etc.) with timestamps and run IDs. Thread-safe via `JSONLTracker._lock`. Supports log rotation (100MB default, 5 backups).

2. **Model registry** (`workspace/registry.json`): JSON file tracking champion models per key (e.g., `experiment_name:classification`). Thread-safe via `fcntl` file locking.

3. **State files** (generated after each run):
   - `workspace/current_state.md`: LLM-readable experiment state summary
   - `workspace/leaderboard.md`: Ranked model performance table

4. **Model artifacts** (`workspace/artifacts/`): Serialized trained models saved per model per run.

5. **Configuration**: Passed as `ExperimentConfig` (Pydantic model) — either constructed from CLI flags or loaded from a Python config module via `importlib`.

## Routing

Not applicable in the traditional web sense. The system routes via:

- **CLI commands**: Typer routes commands to handler functions in `cli.py` (`run`, `hpo`, `drift`, `leaderboard`, etc.)
- **MCP tools**: FastMCP routes tool invocations to handler functions in `mcp/tools.py`
- **Model routing**: `ModelSelector.select()` routes models based on dataset size and hardware profile
- **Model factory**: `get_model_class(name)` lazily resolves model names to implementation classes

## Configuration

Configuration is managed through `ExperimentConfig` (`src/tabular_blueprint/config.py`), a Pydantic BaseModel with validation:

- **Defaults**: Sensible defaults for all fields (5-fold stratified CV, auto model selection, JSONL tracking)
- **Task-aware defaults**: Metrics and CV strategy auto-adjust based on task type (classification vs regression) via `model_validator`
- **Loading**: Config can be provided via:
  1. CLI flags (`--data`, `--target`, `--task`, `--models`, `--config`)
  2. Python config module (`--config path/to/config.py`) loaded via `importlib`
  3. Programmatic construction (`ExperimentConfig(...)`)
- **Hardware profile**: `HardwareProfile.detect()` auto-detects GPU, VRAM, RAM, CPU cores at runtime
- **Enums**: Type-safe enums in `constants.py` for task types, CV strategies, model names, tracker types
- **Serialization**: Custom field serializers for enums and Path objects for JSON output
