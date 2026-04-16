# Codebase Conventions

Derived from pyproject.toml, source files, and test files.

## Linting and Formatting

### Ruff (configured in pyproject.toml)

- **Line length:** 100
- **Target version:** Python 3.11
- **Enabled rule sets:**
  - `E` -- pycodestyle errors
  - `F` -- pyflakes
  - `I` -- isort (import ordering)
  - `UP` -- pyupgrade (modernize syntax)
  - `B` -- flake8-bugbear
  - `SIM` -- flake8-simplify
- **Per-file ignores:**
  - `main.py`: `B008` (typer requires `Option()` in function signatures)
  - `notebooks/*`: `E402`, `I001` (notebook-style imports per cell)
- No separate ruff.toml or .ruff.toml file -- all config is in `[tool.ruff]` in pyproject.toml.

### Mypy

- Not configured in pyproject.toml. No mypy section found. Type checking relies on inline annotations only.

## Naming Conventions

| Element            | Convention       | Example                                      |
|--------------------|------------------|----------------------------------------------|
| Modules            | snake_case       | `evaluator.py`, `report_service.py`, `hpo.py`|
| Classes            | PascalCase       | `Evaluator`, `ExperimentConfig`, `Trainer`   |
| Functions          | snake_case       | `create_study()`, `get_model_class()`        |
| Constants (module) | UPPER_SNAKE      | `METRICS_REGISTRY`, `_MODEL_REGISTRY`        |
| Private helpers    | Leading underscore | `_validate_bounds()`, `_is_numeric()`       |
| Enums              | PascalCase members | `TaskType.CLASSIFICATION`, `CVStrategy.KFOLD`|
| Dataclasses        | PascalCase       | `LeaderboardEntry`, `PromotionResult`        |

### Module-level naming

- Registry dicts use `UPPER_SNAKE_CASE` (e.g., `METRICS_REGISTRY`, `_MODEL_REGISTRY`).
- Caches use `UPPER_SNAKE_CASE` with a leading underscore for module-private (e.g., `_MODEL_CLASS_CACHE`).
- Sentinel singletons use `UPPER_SNAKE_CASE` (e.g., `WORKSPACE_DIR`).

## Import Ordering

Ruff's `I` rule enforces isort-style ordering. Observed pattern:

```python
# 1. Standard library
import hashlib
import json
from pathlib import Path

# 2. Third-party
import numpy as np
import polars as pl
from pydantic import BaseModel, Field

# 3. Local / project
from configs.experiment import ExperimentConfig
from core.constants import TaskType
from core.engine.evaluator import Evaluator
```

- Blank line between each group.
- Within a group: alphabetical, with `import X` before `from X import Y`.
- In `trainer.py`, post-configuration imports use `# noqa: E402` to suppress module-level-import-not-at-top warnings, since `HardwareProfile.configure_omp_threads()` must run first.

## Type Hint Usage

### Union syntax (Python 3.10+ style)

The codebase uses the `X | Y` union syntax (not `Union[X, Y]`) everywhere:

```python
def load(path: str | Path) -> dict[str, Any] | None: ...
def evaluate(self, ..., task: str | None = None) -> dict[str, float]: ...
search_space: dict | None = None
```

### Protocols (structural subtyping)

`core/models/base.py` uses `typing.Protocol` for model interface:

```python
class AbstractModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

    @property
    def model_name(self) -> str: ...
```

### Abstract base classes

`core/models/gbdt_base.py` uses `abc.ABC` / `abc.abstractmethod` for GBDT models:

```python
class BaseGBDTModel:
    @abstractmethod
    def _build_params(self) -> dict: ...
    @abstractmethod
    def _create_model(self, params: dict): ...
```

### Pydantic for config

`ExperimentConfig` in `configs/experiment.py` uses Pydantic v2 `BaseModel` with `model_validator` and `field_serializer` decorators. Enums (`TaskType`, `CVStrategy`, `TrackerType`) are used directly as field types.

### Generic containers

All dict/list types use lowercase generics (`dict[str, float]`, `list[str]`), not `Dict`, `List`.

## Error Handling Patterns

### ValueError for business logic validation

The primary exception type. Used for invalid arguments, unknown model names, bad search spaces:

```python
raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
raise ValueError(f"Invalid search space for '{param_name}': bounds must be numeric")
raise ValueError(f"Unsupported file format: {path.suffix}. Supported: .csv, .parquet")
```

### FileNotFoundError for missing files

```python
if not db_path.exists():
    raise FileNotFoundError(f"Database file not found: {db_path}")
```

### Exception chaining with `from e`

Used consistently to preserve original exception context:

```python
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}") from e
except sqlite3.Error as e:
    raise ValueError(f"Database error: {e}") from e
```

### Try/except in HPO objective

The HPO module catches `optuna.TrialPruned` separately to re-raise, and wraps other exceptions:

```python
try:
    scores = evaluator.evaluate(...)
except optuna.TrialPruned:
    raise
except Exception as e:
    raise optuna.TrialPruned(f"Evaluation failed... Error: {e}") from e
```

### Graceful degradation in Trainer

The trainer wraps individual model training in try/except and records failures as events rather than crashing:

```python
except Exception as e:
    self.tracker.log_event({"event": "model_failed", "model": model_name, "error": str(e)})
    return {"error": str(e)}
```

## Logging and Event Tracking

### No stdlib logging

The codebase does **not** use Python's `logging` module. Instead, it uses a custom `Tracker` system:

- `JSONLTracker` writes structured events to `.jsonl` files.
- `Tracker` is the Protocol base class.
- Events are dicts with an `"event"` key (e.g., `"experiment_started"`, `"model_completed"`, `"model_failed"`).

### Event structure

```python
{
    "event": "model_completed",
    "run_id": "exp_1713273600_a1b2c3",
    "model": "CatBoost",
    "task": "classification",
    "cv_scores": {"roc_auc": 0.92, "f1_macro": 0.88},
    "duration_seconds": 12.34,
    "artifact_path": "workspace/artifacts/catboost_exp_...",
    "hardware": {"device": "cuda", "vram_used_gb": 0.0},
}
```

### CLI output

Uses `typer.echo()` for CLI output (no print statements).

## Architecture Patterns

### Factory Pattern

`core/models/factory.py` implements a lazy-import factory with a module-level registry dict and caching:

```python
_MODEL_REGISTRY = {
    "catboost": ("core.models.conventional.catboost_model", "CatBoostModel"),
    ...
}
def get_model_class(model_name: str) -> type:
    # Lazy import + cache
```

### Strategy Pattern

Cross-validation splitting uses a strategy selector:

```python
def get_cv_split(strategy: str, n_splits: int = 5):
    if strategy == "kfold": return KFold(...)
    elif strategy == "stratified": return StratifiedKFold(...)
    elif strategy == "timeseries": return TimeSeriesSplit(...)
```

### Protocol (Structural Subtyping)

`AbstractModel` in `core/models/base.py` is a `typing.Protocol`, not an abstract base class. Models conform structurally.

### Template Method (ABC)

`BaseGBDTModel` defines the skeleton (`fit`, `save`, `predict_proba`) and requires subclasses to implement `_build_params`, `_create_model`, `_train_model`, `predict`, `_predict_proba_impl`, `load`, and `model_name`.

### Enum-based Configuration

`core/constants.py` defines `Enum` classes for type-safe config values (`TaskType`, `CVStrategy`, `ModelName`, `TrackerType`) with converter functions (`from_task_type`, `from_cv_strategy`) that accept both strings and enum values.

### Service Layer

Business logic is organized into services under `core/services/`:

- `RegistryService` -- model registry with file locking (`fcntl`).
- `ReportService` -- builds structured experiment reports from JSONL logs.

### Data Adapter

`core/data/adapter.py` abstracts data format conversion (e.g., Polars DataFrame to NumPy arrays).

## Constants and Config Patterns

### Module-level constants

Constants are defined at module scope with UPPER_SNAKE_CASE:

```python
# core/engine/evaluator.py
METRICS_REGISTRY = {
    "classification": {"roc_auc": roc_auc_score, ...},
    "regression": {"rmse": ..., "mae": ..., "r2": ...},
}

# core/models/selector.py (class-level constants)
TABPFN_ROW_LIMIT = 10_000
FT_TRANSFORMER_ROW_MIN = 50_000
```

### Pydantic settings

Configuration uses Pydantic v2 `BaseModel` with:

- `Field(default_factory=...)` for mutable defaults.
- `model_validator(mode="after")` for cross-field validation and defaults.
- `field_serializer` for enum-to-string JSON serialization.
- `Literal["auto"]` for sentinel values.

### Config convention

Experiment configs live in `configs/` as Python modules (not YAML/TOML). The CLI can load a config module dynamically via `importlib.util.spec_from_file_location`.
