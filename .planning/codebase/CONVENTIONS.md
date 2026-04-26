# Conventions

> Last updated: 2026-04-23

## Code Style

### Formatting

- **Formatter**: Ruff (`ruff format`) — replaces Black
- **Line length**: 100 characters (`pyproject.toml` → `[tool.ruff] line-length = 100`)
- **Target Python**: 3.11+ (`target-version = "py311"`)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes (Ruff default)
- **Trailing commas**: Follows Ruff defaults (multi-line collections)

### Linting

- **Linter**: Ruff (`ruff check`)
- **Config**: `pyproject.toml` → `[tool.ruff.lint]`
- **Selected rules** (`select`): `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `UP` (pyupgrade), `B` (flake8-bugbear), `SIM` (flake8-simplify), `C4` (flake8-comprehensions), `PT` (flake8-pytest-style), `RUF` (Ruff-specific)
- **Fixable**: `["ALL"]` — all rules are auto-fixable via `ruff check --fix`
- **Per-file ignores**:
  - `src/tabular_blueprint/cli.py`: `B008` (function-call-in-default-argument — needed for Typer)
  - `notebooks/*`: `E402` (import not at top), `I001` (import sorting)

### Pre-commit

- **Config**: `.pre-commit-config.yaml`
- **Hooks**:
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `debug-statements` (pre-commit-hooks v6.0.0)
  - `ruff` (with `--fix`), `ruff-format` (astral-sh/ruff-pre-commit v0.11.2)
  - Local `pytest` hook: runs `uv run --group dev --extra llm pytest tests/unit -v` on all Python files

## Type System

- **Config**: `pyproject.toml` → `[tool.mypy]`
- **Python version**: 3.11
- **Strictness**: `disallow_untyped_defs = true` — all function signatures must have type annotations
- **Pragmatism**: `ignore_missing_imports = true` — third-party libs without stubs are not blocked
- **Package marker**: `src/tabular_blueprint/py.typed` (PEP 561 compliant)
- **Modern union syntax**: Uses `X | Y` instead of `Union[X, Y]` (Python 3.11+ style, enforced by `UP` rule)
- **Common type patterns**:
  - `dict[str, Any]` instead of `Dict[str, Any]`
  - `list[str]` instead of `List[str]`
  - `str | None` instead of `Optional[str]`
  - `ClassVar` for class-level constants (e.g., `Trainer._GBDT_PRIORITY`)

## Naming Conventions

### Variables and Functions

- **snake_case**: All variables, functions, methods (e.g., `cv_scores`, `get_model_class`, `detect_leakage`)
- **Private methods**: Single underscore prefix (e.g., `_train_single_model`, `_fit_quick_gbdt`, `_to_numpy`)
- **Double-underscore**: Not used

### Classes

- **PascalCase**: All classes (e.g., `ExperimentConfig`, `DriftDetector`, `JSONLTracker`, `BaseGBDTModel`)
- **Suffix patterns**:
  - `*Model` — model implementations (e.g., `LightGBMModel`, `CatBoostModel`)
  - `*Service` — service layer classes (e.g., `RegistryService`, `ExportService`, `ReportService`)
  - `*Detector` — monitoring components (e.g., `DriftDetector`, `PSIDriftDetector`)
  - `*Tracker` — experiment tracking backends (e.g., `JSONLTracker`, `WandbTracker`, `MLflowTracker`)
  - `*Result` / `*Report` — Pydantic/dataclass result objects (e.g., `DriftReport`, `PromotionResult`)
  - `*Config` — configuration models (e.g., `ExperimentConfig`, `HardwareProfile`)
  - `*Error` — exception hierarchy (e.g., `TabularBlueprintError`, `ModelFitError`)

### Constants and Enums

- **UPPER_SNAKE_CASE**: Module-level constants (e.g., `BASELINE_MODELS`, `WORKSPACE_DIR`, `METRICS_REGISTRY`, `LOWER_IS_BETTER_METRICS`)
- **Enums**: PascalCase class names, UPPER_SNAKE_CASE members (e.g., `TaskType.CLASSIFICATION`, `CVStrategy.STRATIFIED`)
- Enum values are lowercase strings (e.g., `"classification"`, `"kfold"`)

### Files and Directories

- **Source files**: `snake_case.py` (e.g., `feature_engine.py`, `psi_drift.py`, `domain_classifier.py`)
- **Test files**: `test_{module}.py` (e.g., `test_drift.py`, `test_baselines.py`, `test_model_factory.py`)
- **Packages**: `snake_case` directories with `__init__.py` (e.g., `data/`, `models/`, `engine/`, `monitoring/`, `services/`, `utils/`, `pipelines/`, `mcp/`, `llm/`)

## Import Patterns

### Organization

Imports follow Ruff's `I` (isort) rule with these groups (separated by blank lines):

1. **Standard library**: `import json`, `from pathlib import Path`, `from typing import Any`
2. **Third-party**: `import numpy as np`, `import polars as pl`, `from pydantic import BaseModel`
3. **First-party**: `from tabular_blueprint.config import ExperimentConfig`

### Conventions

- **Absolute imports** for first-party code: `from tabular_blueprint.config import ExperimentConfig`
- **Relative imports not used** in source code
- **Lazy imports** for optional/heavy dependencies inside function bodies to avoid import-time failures:
  - `import torch` inside `HardwareProfile.detect()`
  - `from cleanlab.filter import find_label_issues` inside `audit_data_quality()`
  - `import lightgbm as lgb` at module level (but guarded by model-specific paths)
- **`noqa: E402`** used sparingly when module-level initialization forces late imports (e.g., `trainer.py:16-34`)
- **Common aliases**: `numpy as np`, `polars as pl`, `lightgbm as lgb`

## Error Handling

### Exception Hierarchy

Defined in `src/tabular_blueprint/exceptions.py`:

```
TabularBlueprintError (base)
├── DataLoadError     — data loading/validation failures
├── ModelFitError     — model training failures
└── RegistryError     — registry operation failures
```

### Error Context

All custom exceptions accept a `context` keyword dict:
```python
raise ModelFitError("Model catboost failed", context={"model": "catboost", "error": str(e)})
```

### `@track_errors` Decorator

A class-method decorator that:
1. Passes through `TabularBlueprintError` subclasses unchanged
2. Converts `ValueError` → `DataLoadError`
3. Converts `RuntimeError` → `ModelFitError`
4. Converts all other exceptions → `ModelFitError` (with `original_type` context)
5. Logs error events to the instance's tracker before re-raising

Usage:
```python
class Trainer:
    @track_errors()
    def _train_single_model(self, ...):
        ...
```

### CLI Error Handling

- Typer CLI uses `typer.Exit(1)` for graceful exits
- Catches `ValueError` / `FileNotFoundError` and converts to user-friendly messages
- Uses `raise ... from e` pattern for exception chaining throughout

## Logging and Tracking

### Structured Event Logging

- **No use of Python's `logging` module** — all tracking goes through the `Tracker` protocol
- **Tracker Protocol** (`src/tabular_blueprint/engine/tracker.py`):
  - `log_event(event: dict)` — primary method, adds `run_id` and `timestamp` automatically
  - `log_metrics(metrics: dict)` — convenience: wraps as `{"event": "metrics", ...}`
  - `log_params(params: dict)` — convenience: wraps as `{"event": "params", ...}`
  - `log_artifact(path: str)` — convenience: wraps as `{"event": "artifact", ...}`
  - `finish()` — logs `run_completed` and clears run ID

### JSONL Tracker (Default)

- Writes to `workspace/experiments.jsonl`
- Thread-safe (uses `threading.Lock`)
- Log rotation: max 100MB per file, 5 backup files
- Events include structured event types: `experiment_started`, `model_completed`, `baseline_completed`, `drift_check`, `leakage_audit`, `noise_cleaned`, `shap_explainability`, `metrics`, `params`, `run_completed`, etc.

### Console Output

- CLI uses `typer.echo()` for plain text, `rich.console.Console` and `rich.table.Table` for formatted output
- Rich is used for colored warnings and tables (e.g., experiment diff, leaderboard)

## Code Patterns

### Pydantic Models for Configuration and Data

- `ExperimentConfig(BaseModel)` — central config with validators and serializers
- `HardwareProfile(BaseModel)` — hardware detection with `@classmethod detect()`
- Drift/monitoring results use Pydantic `BaseModel` (e.g., `DriftReport`, `ColumnDriftResult`)
- Service results use `@dataclass(frozen=True)` (e.g., `PromotionResult`, `LeaderboardEntry`)

### Protocol-Based Interfaces

- `AbstractModel(Protocol)` in `src/tabular_blueprint/models/base.py` — structural subtyping for models
- `Tracker(Protocol)` in `src/tabular_blueprint/engine/tracker.py` — pluggable tracking backends
- No ABC inheritance; Protocols used for duck-typing contracts

### Template Method for GBDT Models

- `BaseGBDTModel` (ABC) in `src/tabular_blueprint/models/gbdt_base.py`
- Subclasses implement: `_build_params()`, `_create_model()`, `_train_model()`, `_predict_proba_impl()`, `load()`, `model_name`
- Common `fit()`, `predict_proba()`, `save()` provided by base

### Lazy Factory Pattern

- `get_model_class(name)` in `src/tabular_blueprint/models/factory.py` — lazy imports with caching
- Model registry maps string names → `(module_path, class_name)` tuples
- Avoids importing all ML frameworks at startup

### Data Format Adapter

- `DataAdapter` converts Polars DataFrames to model-specific formats (numpy, tensor, dataset)
- Single point of truth for format conversion

### Decorator Pattern

- `@track_errors()` — error translation and tracking (see Error Handling above)

### Singleton Constants

- Enum-based configuration in `constants.py` (e.g., `TaskType`, `CVStrategy`, `ModelName`, `TrackerType`)
- Conversion functions for backward compat: `from_task_type()`, `from_cv_strategy()`, etc.

## Shared Utilities

### `src/tabular_blueprint/utils/jsonl.py`

- `load_events(path)` — loads JSONL files, handles missing files, blank lines, malformed JSON

### `src/tabular_blueprint/services/report_service.py`

- `metric_value_is_better()` — direction-aware metric comparison
- `metric_higher_is_better()` — checks if higher is better for a metric
- `metric_sort_value()` — normalizes scores for descending sort
- `resolve_primary_score()` — picks primary metric from cv_scores dict
- `LOWER_IS_BETTER_METRICS` — set of metric names where lower values are better

### `src/tabular_blueprint/data/loaders.py`

- `load_data(path)` — universal loader (CSV or Parquet)
- `load_csv()`, `load_parquet()`, `load_sqlite()` — format-specific loaders
- `get_data_hash(df)` — deterministic SHA-256 hash of DataFrame contents

### `src/tabular_blueprint/constants.py`

- Enum definitions and string-to-enum conversion utilities
