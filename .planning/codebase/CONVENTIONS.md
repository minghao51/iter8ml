# Codebase Conventions

## Code Style

### Formatting
- **Tool**: Ruff (`ruff format`) with line-length = 100, target Python 3.11+
- **Config**: `pyproject.toml:42-44`
- **Pre-commit**: Runs `ruff format` and `ruff check --fix` via `uv run`
- **Pre-commit config**: `.pre-commit-config.yaml:1-19`

### Linting Rules
- **Enabled rule sets**: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify)
- **Config**: `pyproject.toml:46-47`
- **Per-file ignores**:
  - `main.py`: `B008` (typer requires `Option()` in function signatures)
  - `notebooks/*`: `E402`, `I001` (notebook-style imports per cell)
- **Config**: `pyproject.toml:49-51`

### Import Style
- Standard library imports first, then third-party, then local
- `isort` rules enforced via Ruff (`I` in select)
- Lazy imports used for optional/heavy dependencies (e.g., `import torch` inside functions, `import sqlite3` inside `load_sqlite`)
- Examples:
  - `core/data/loaders.py:32` — `import sqlite3` inside function
  - `core/data/adapter.py:45` — `import torch` inside `_to_tensor`
  - `core/data/quality.py:28-29` — `from cleanlab.*` inside function with try/except

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `ExperimentConfig`, `DriftDetector`, `JSONLTracker`, `DataAdapter`)
- **Functions/Methods**: `snake_case` (e.g., `load_csv`, `fill_nulls`, `evaluate`, `detect`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `METRICS_REGISTRY` at `core/engine/evaluator.py:15`)
- **Private methods**: Leading underscore (e.g., `_to_numpy`, `_ks_test`, `_load_events`)
- **Test files**: `test_<module>.py` naming (e.g., `test_config.py`, `test_processors.py`)

## Typing Conventions

### Type Hint Style
- Uses modern Python 3.10+ union syntax: `str | Path` instead of `Union[str, Path]`
- Uses `dict[str, float]`, `list[str]`, `tuple` built-in generics
- **Config**: `pyproject.toml:5` — `requires-python = ">=3.11"`
- No `from __future__ import annotations` needed (Python 3.11+ supports native syntax)

### Type Usage
- **Function parameters**: Fully typed with type hints on all public functions
- **Return types**: Always specified (e.g., `-> pl.DataFrame`, `-> DriftReport`, `-> dict[str, float]`)
- **Pydantic models**: Used for config and result validation
  - `configs/experiment.py:1-19` — `ExperimentConfig` with `Literal` types and `Field`
  - `core/monitoring/drift.py:8-19` — `ColumnDriftResult`, `DriftReport` as Pydantic `BaseModel`
- **Protocols**: `typing.Protocol` used for interface definitions
  - `core/engine/tracker.py:9-14` — `Tracker` protocol defining tracker interface
- **Literal types**: Used for constrained string values
  - `configs/experiment.py:8` — `Literal["classification", "regression"]`
  - `core/data/adapter.py:20` — `Literal["numpy", "tensor", "dataset"]`
- **py.typed marker**: Present at `core/py.typed` (signals PEP 561 type hints support)

### Typing Strictness
- No `mypy` configured in project
- Type hints are present but not enforced by CI
- Ruff `UP` rules enforce modern type syntax (pyupgrade)

## Error Handling Patterns

### Exception Raising
- **ValueError** for invalid arguments:
  - `core/engine/evaluator.py:39` — `raise ValueError(f"Unknown CV strategy: {strategy}")`
  - `core/data/adapter.py:35` — `raise ValueError(f"Unsupported target format: {self.target_format}")`
- **ImportError** with chained exceptions for optional dependencies:
  - `core/engine/tracker.py:54-55` — `raise ImportError("wandb is required...") from e`
  - `core/engine/tracker.py:87-88` — `raise ImportError("mlflow is required...") from e`
  - `core/data/adapter.py:56-60` — `raise ImportError(...) from e`

### Graceful Degradation
- Optional features return safe defaults instead of raising:
  - `core/data/quality.py:30-31` — Returns `{"enabled": False, "message": "cleanlab not installed"}`
  - `core/engine/hpo.py:61-62` — Catches `Exception` and raises `optuna.TrialPruned()`
  - `core/data/processors.py:40` — Returns original DataFrame if no expressions to apply

### Validation
- **Pydantic** used for config validation:
  - `configs/experiment.py` — `ExperimentConfig` validates `task`, `cv_strategy`, `tracker` via `Literal`
  - Invalid values raise `pydantic.ValidationError` (tested in `tests/unit/test_config.py:23-30`)
- **Pydantic** used for structured results:
  - `core/monitoring/drift.py:8-19` — `ColumnDriftResult`, `DriftReport` enforce schema

## Logging Patterns

### No Standard Logging
- The codebase does **not** use Python's `logging` module
- No `logger = logging.getLogger(__name__)` patterns found
- No log files or log configuration

### Output Mechanisms
- **Rich** for CLI output (dependency listed in `pyproject.toml:27`)
- **JSONL events** for experiment tracking:
  - `core/engine/tracker.py:34-38` — `JSONLTracker.log_event()` writes JSON lines to `experiments.jsonl`
  - Events include `run_id`, `timestamp`, and event-specific data
- **Markdown reports** for state summaries:
  - `core/engine/state_observer.py:20-71` — Generates `current_state.md` leaderboard
- **CLI stdout** via Typer/Rich:
  - `main.py` (entry point) — Commands output formatted text to console

### Tracker Abstraction
- `Tracker` protocol at `core/engine/tracker.py:9-14` defines logging interface
- Implementations: `JSONLTracker`, `WandbTracker`, `MLflowTracker`
- All trackers implement: `log_metrics`, `log_params`, `log_artifact`, `log_event`, `finish`

## Shared Utilities and Their Conventions

### Data Layer (`core/data/`)
- **`loaders.py`** (`core/data/loaders.py:1-44`): Pure functions for data ingestion
  - `load_csv(path, *, separator=",", ...)` — keyword-only optional params
  - `load_parquet(path)` — simple wrapper around `pl.read_parquet`
  - `load_sqlite(db_path, query)` — lazy imports sqlite3
  - `get_data_hash(df)` — deterministic SHA-256 hash
- **`processors.py`** (`core/data/processors.py:1-93`): Pure functions for preprocessing
  - All functions take `pl.DataFrame` and return `pl.DataFrame`
  - Keyword-only config params (`*` separator)
  - Composable: `pipeline()` chains `fill_nulls`, `decompose_dates`, `encode_categoricals`
- **`adapter.py`** (`core/data/adapter.py:1-66`): Format conversion class
  - `DataAdapter` class with `transform()` method
  - Private methods: `_to_numpy`, `_to_tensor`, `_to_dataset`
- **`quality.py`** (`core/data/quality.py:1-59`): Data quality audit
  - Single function `audit_data_quality()` returns dict report
  - Optional output to JSON file

### Engine Layer (`core/engine/`)
- **`evaluator.py`** (`core/engine/evaluator.py:1-97`): Cross-validation and metrics
  - `METRICS_REGISTRY` dict maps task type → metric name → function
  - `get_cv_split()` factory function for CV splitters
  - `Evaluator` class with `evaluate()` method
- **`hpo.py`** (`core/engine/hpo.py:1-70`): Hyperparameter optimization
  - `create_study()` — Optuna study factory
  - `optimize_model()` — runs optimization, returns best params/scores
- **`tracker.py`** (`core/engine/tracker.py:1-107`): Experiment tracking
  - `Tracker` protocol, `JSONLTracker`, `WandbTracker`, `MLflowTracker`
- **`trainer.py`** (`core/engine/trainer.py`): Main training orchestration
- **`state_observer.py`** (`core/engine/state_observer.py:1-86`): State summary generation
  - `StateObserver` class generates markdown from JSONL events

### Monitoring Layer (`core/monitoring/`)
- **`drift.py`** (`core/monitoring/drift.py:1-89`): Drift detection
  - `DriftDetector` class with `detect()` method
  - Returns `DriftReport` (Pydantic model)
  - Uses KS test for numeric, chi-squared for categorical

### Config Layer (`configs/`)
- **`experiment.py`** (`configs/experiment.py:1-19`): `ExperimentConfig` Pydantic model
- **`hardware.py`** (`configs/hardware.py`): Hardware configuration
- **`model_configs.py`** (`configs/model_configs.py`): Model-specific hyperparameter defaults

### File Organization Pattern
- `core/` — library code (importable as `tabular-blueprint`)
- `configs/` — configuration models and defaults
- `main.py` — CLI entry point (Typer app)
- `tests/` — mirrors `core/` structure with `unit/` and `integration/` subdirs
- `workspace/` — runtime output directory (artifacts, experiments.jsonl, registry.json)
- `notebooks/` — exploratory notebooks
- `examples/` — example configs
- `pipelines/` — pipeline definitions

### Function Signature Conventions
- Keyword-only parameters after `*` for configuration options
- Default values for all optional parameters
- Type hints on all parameters and return types
- Docstrings for public functions/classes
