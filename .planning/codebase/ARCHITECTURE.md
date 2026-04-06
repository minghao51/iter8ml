# Architecture

## Overall Architectural Pattern

**Layered Pipeline Architecture** with a CLI-driven entry point. The codebase follows a modular, loosely-coupled design where each layer has a single responsibility:

```
CLI (Typer) → Config → Data → Engine → Models → Tracking
```

The project name is **tabular-blueprint** — a high-velocity iteration framework for tabular ML that automates model selection, training, evaluation, and tracking across multiple model families.

## Component Layers and Responsibilities

### 1. Entry Layer (`main.py`)
- **File:** `main.py`
- **Role:** CLI entry point using Typer framework
- **Commands:** `init`, `run`, `leaderboard`, `registry`, `hardware`, `drift`, `state`, `hpo`
- **Entry point script:** `tabblueprint = "main:app"` (defined in `pyproject.toml:40`)

### 2. Configuration Layer (`configs/`)
- **Files:** `configs/experiment.py`, `configs/hardware.py`, `configs/model_configs.py`
- **Role:** Pydantic-based configuration models
- **Key classes:**
  - `ExperimentConfig` — experiment parameters (task, target, models, CV settings, tracker type)
  - `HardwareProfile` — auto-detected GPU/CPU/RAM profile for model routing
  - `ModelConfigs` — per-model default hyperparameters and HPO search spaces
  - Individual model configs: `CatBoostConfig`, `LightGBMConfig`, `XGBoostConfig`, `TabPFNConfig`, `FTTransformerConfig`

### 3. Data Layer (`core/data/`)
- **Files:** `core/data/loaders.py`, `core/data/processors.py`, `core/data/adapter.py`, `core/data/quality.py`
- **Role:** Data ingestion, preprocessing, format conversion, and quality auditing
- **Key abstractions:**
  - `load_csv()`, `load_parquet()`, `load_sqlite()` — Polars-based data loaders
  - `DataAdapter` — converts Polars DataFrames to numpy/tensors/HuggingFace datasets
  - `fill_nulls()`, `decompose_dates()`, `encode_categoricals()` — preprocessing functions
  - `pipeline()` — composable preprocessing pipeline
  - `audit_data_quality()` — Cleanlab-based label noise detection

### 4. Engine Layer (`core/engine/`)
- **Files:** `core/engine/trainer.py`, `core/engine/evaluator.py`, `core/engine/hpo.py`, `core/engine/tracker.py`, `core/engine/state_observer.py`
- **Role:** Core orchestration — experiment execution, cross-validation, HPO, tracking
- **Key classes:**
  - `Trainer` — orchestrates full experiment runs: config → data → model selection → evaluation → tracking → registry
  - `Evaluator` — cross-validation with configurable strategies (kfold, stratified, timeseries) and metrics
  - `optimize_model()` — Optuna-based hyperparameter optimization
  - `Tracker` (Protocol) — pluggable tracking interface
  - `JSONLTracker`, `WandbTracker`, `MLflowTracker` — tracking implementations
  - `StateObserver` — generates LLM-readable experiment state summaries

### 5. Model Layer (`core/models/`)
- **Files:** Organized by model family
  - `core/models/base.py` — `AbstractModel` Protocol (structural subtyping)
  - `core/models/selector.py` — `ModelSelector` for hardware/data-size-aware routing
  - `core/models/conventional/` — CatBoost, LightGBM, XGBoost wrappers
  - `core/models/tabular_foundation/` — TabPFN wrapper
  - `core/models/deep/` — FT-Transformer, TextEncoder (DeBERTa)
- **Key abstraction:** `AbstractModel` Protocol defines the interface: `fit()`, `predict()`, `predict_proba()`, `save()`, `load()`, `model_name`
- **Model registry:** String-to-class mapping in `core/engine/trainer.py:23-29`

### 6. Monitoring Layer (`core/monitoring/`)
- **Files:** `core/monitoring/drift.py`
- **Role:** Production drift detection
- **Key class:** `DriftDetector` — KS test (numeric) and Chi-squared test (categorical) with Pydantic result models

### 7. MCP Server Layer (`mcp_server/`)
- **Files:** `mcp_server/tools.py`
- **Role:** Exposes experiment operations as MCP tools for LLM agents
- **Tools:** `get_experiment_state`, `get_column_stats`, `run_baseline`, `run_hpo`, `get_event_log`, `registry_show`, `registry_promote`, `detect_drift`

### 8. Pipelines Layer (`pipelines/`)
- **Files:** `pipelines/__init__.py`
- **Role:** Placeholder for Hamilton DAG-based feature engineering (optional dependency)

## Data Flow Between Components

```
1. CLI command (main.py)
   ↓
2. Load ExperimentConfig (configs/experiment.py)
   ↓
3. Detect HardwareProfile (configs/hardware.py)
   ↓
4. Load data via Polars (core/data/loaders.py)
   ↓
5. Trainer.run(df) (core/engine/trainer.py)
   ├── ModelSelector.select() → determines which models to run
   ├── DataAdapter.transform() → converts to numpy/tensors
   ├── Evaluator.evaluate() → cross-validation with metrics
   │   └── For each fold: model_cls().fit() → model_cls().predict()
   ├── Model.save() → workspace/artifacts/
   ├── JSONLTracker.log_event() → workspace/experiments.jsonl
   ├── _update_registry() → workspace/registry.json (champion tracking)
   └── StateObserver.generate() → workspace/current_state.md
```

## Key Abstractions and Interfaces

### AbstractModel Protocol (`core/models/base.py:8-16`)
```python
class AbstractModel(Protocol):
    def fit(self, X, y, **kwargs) -> None
    def predict(self, X) -> np.ndarray
    def predict_proba(self, X) -> np.ndarray | None
    def save(self, path: str) -> None
    def load(self, path: str) -> None
    @property
    def model_name(self) -> str
```

### Tracker Protocol (`core/engine/tracker.py:9-14`)
```python
class Tracker(Protocol):
    def log_metrics(self, metrics: dict, step: int | None) -> None
    def log_params(self, params: dict) -> None
    def log_artifact(self, path: str) -> None
    def log_event(self, event: dict) -> None
    def finish(self) -> None
```

### Model Selector (`core/models/selector.py:6-39`)
Routes models based on data size and hardware:
- n_rows < 10k → [TabPFN, CatBoost, LightGBM]
- 10k ≤ n_rows < 500k → [CatBoost, LightGBM, XGBoost]
- n_rows ≥ 500k → [LightGBM, XGBoost]
- vram_gb > 12 and n_rows ≥ 50k → append FT-Transformer

### Metrics Registry (`core/engine/evaluator.py:15-27`)
- Classification: roc_auc, f1_macro, accuracy, log_loss
- Regression: rmse, mae, r2

## Entry Points and Initialization Flow

### Primary Entry: `main.py` (Typer CLI)
- `tabblueprint init` — creates workspace/ directory structure
- `tabblueprint run` — runs full experiment (config → data → trainer → results)
- `tabblueprint leaderboard` — displays results from experiments.jsonl
- `tabblueprint registry` — shows/promotes model champions
- `tabblueprint hardware` — shows detected hardware profile
- `tabblueprint drift` — detects distribution drift between datasets
- `tabblueprint state` — generates current experiment state
- `tabblueprint hpo` — runs hyperparameter optimization

### Programmatic Entry: `Trainer.run(df)`
```python
config = ExperimentConfig(...)
trainer = Trainer(config)
results = trainer.run(df)  # Returns dict[model_name, cv_scores]
```

### MCP Server Entry: `mcp_server/tools.py`
FastMCP server exposing 8 tools for LLM agent interaction with the experiment system.

## Workspace Directory Structure (runtime)
- `workspace/experiments.jsonl` — append-only event log
- `workspace/registry.json` — champion model registry
- `workspace/registry.lock` — file lock for registry updates
- `workspace/artifacts/` — saved model artifacts
- `workspace/leaderboard.md` — auto-generated markdown leaderboard
- `workspace/current_state.md` — LLM-readable state summary
