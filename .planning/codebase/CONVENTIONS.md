# Code Conventions

## Code Style (Ruff)

- **Line length:** 100 (`pyproject.toml:62`)
- **Target version:** Python 3.11 (`pyproject.toml:63`)
- **Selected rules:** E, F, I (imports), UP (pyupgrade), B (bugbear), SIM (simplify), C4 (comprehensions), PT (pytest), RUF (ruff-specific) — `pyproject.toml:110`
- **Formatter:** `ruff-format` via pre-commit (`.pre-commit-config.yaml:21-22`)
- All rule sets are fixable (`fixable = ["ALL"]`, line 111)
- **Per-file ignores:** `cli.py` exempt from B008 (no bare function calls in argument defaults); notebooks exempt from E402, I001, B018, E501, RUF001, F841

## Naming Conventions

- **snake_case** for modules, functions, variables, methods (`data/loaders.py`, `load_events`)
- **PascalCase** for classes, enums, Pydantic models (`ExperimentConfig`, `JSONLTracker`, `TaskType`)
- **UPPER_CASE** for module-level constants (`LOWER_IS_BETTER_METRICS`, `METRICS_REGISTRY`, `BASELINE_MODELS`, `_MODEL_REGISTRY`)
- Private attributes/methods prefixed with `_` (`self._lock`, `self._rotate_log`, `_load_completed_models`)
- Test files: `test_<module>.py` (`test_loaders.py`, `test_config.py`)
- Test classes: `Test<Name>` (`TestNaiveBaseline`, `TestCalibratedModel`)
- Test functions: `test_<scenario>` (`test_no_drift_same_distribution`, `test_load_events_handles_empty_file`)

## Type Annotation Practices

- All public functions and methods are fully type-annotated (`disallow_untyped_defs = true` in mypy config, `pyproject.toml:67`)
- Uses `from __future__ import annotations` style (implicit in py311+, union syntax with `|`)
- Prefers `collections.abc` types over typing: `collections.abc.Iterator`, `collections.abc.Callable`
- Uses `typing.Any` and `typing.Protocol` for structural subtyping
- Uses `|` union syntax: `str | Path`, `dict[str, Any] | None`, `int | float`
- Optional params use `X | None = None` pattern (not `Optional[X]`)
- Mypy exclusions for third-party packages without stubs (catboost, lightgbm, xgboost, optuna, polars, hamilton) and for specific internal modules (hpo, hpo_warmstart, hpo_importance, pipelines.nodes.*, trainer) where type checking is too complex
- Protocols defined for duck typing: `AbstractModel` (`models/base.py:8`), `Tracker` (`engine/tracker.py:10`)

## Error Handling Patterns

- **Custom exception hierarchy** in `exceptions.py`:
  - `TabularBlueprintError(Exception)` — base, with optional `context: dict[str, Any] | None`
  - `DataLoadError` — data loading/validation failures
  - `ModelFitError` — model training failures
  - `RegistryError` — registry operation failures
- **`track_errors()` decorator** (`exceptions.py:28`): catches exceptions on methods, reraises typed errors (`DataLoadError`, `ModelFitError`), logs to tracker
- **`pytest.raises`** with `match=` for expected errors in tests
- **Early validation** with `raise ValueError(...)` in constructors/functions (e.g., security validation in `load_sqlite`)
- CLI commands catch and exit cleanly with `typer.Exit(1)`
- ImportError for optional dependencies caught inline with clear messages

## Logging Conventions

- **Not using stdlib logging** — the codebase uses JSONL event-based observability instead
- Only 1 file (`tabpfn_model.py`) uses `logging.getLogger`
- Primary observability: `tracker.log_event(dict)` writes structured JSONL to `workspace/experiments.jsonl`
- JSONL events include: `run_id`, `timestamp` (UTC ISO), `event` type (namespace)
- CLI output via `typer.echo()` (not `print`)
- Rich formatting used for tables (`Console`, `Table`) in CLI `diff` command

## Pydantic Model Patterns

- **Namespace:** All in `config.py`, `models/model_configs.py`, monitoring/ modules
- **Inheritance:** Always `BaseModel` from pydantic v2
- **Defaults:** Use `Field(default_factory=...)` for mutable defaults; simple values use `=`
- **Validators:** `model_validator(mode="after")` for cross-field validation (e.g., `ExperimentConfig.apply_task_defaults`)
- **Serializers:** `@field_serializer` with `when_used="json"` for enum/Path serialization
- **Deserialization:** `cls.model_validate(data)` (not `parse_obj`)
- **Export:** `.model_dump(mode="json")` for serialization (pydantic v2)
- **Protocol for structural subtyping:** `AbstractModel` defines duck-typed interface, no inheritance
- Typical pattern: a config class with `hpo_search_space()` method returning dict
- `model_configs.py` uses nested config composition (`ModelConfigs` wrapping per-model configs with `Field(default_factory=...)`)

## Config Management

- `ExperimentConfig` is the single source of truth for experiment settings
- Loaded from `.yaml`/`.yml`, `.toml`, `.json`, or `.py` (`.py` disabled by default — safety)
- `.py` config requires `--allow-unsafe-config` flag
- `HardwareProfile.detect()` auto-detects GPU/CPU/RAM
- `OMP_NUM_THREADS` configured at startup via `HardwareProfile.configure_omp_threads()`

## Import Organization

Sorted by `ruff` I001 (isort-style). Convention observed:
1. Standard library (top-level imports, then `collections.abc`, `pathlib`, etc.)
2. Third-party libraries (polars, numpy, pydantic, sklearn, etc.)
3. Internal `tabular_blueprint.*` imports
4. Lazy imports at point-of-use for heavy dependencies (torch, wandb, mlflow, typer inside methods)
5. No `from typing import Optional` — uses `| None` syntax
6. `__all__` explicitly declared in `__init__.py` for public API

## Shared Constants and Enums

- All in `constants.py`: `TaskType`, `CVStrategy`, `ModelName`, `EmbeddingMethod`, `TrackerType`
- Conversion functions for each enum: `from_task_type()`, `from_cv_strategy()`, etc.
- Module-level dicts as registries: `_MODEL_REGISTRY` (factory.py), `METRICS_REGISTRY` (evaluator.py), `BASELINE_MODELS` (trainer.py), `LOWER_IS_BETTER_METRICS` (report_service.py)

## Module `__init__` Pattern

Each subpackage has an `__init__.py` that re-exports public API via `__all__`:
- `tabular_blueprint/__init__.py`: minimal (`__all__ = []`)
- `tabular_blueprint/models/__init__.py`: re-exports `AbstractModel`, `ModelSelector`, factory functions
- `tabular_blueprint/data/__init__.py`: re-exports loader and embedding functions
- Pipeline files may use `__all__` to control what Hamilton discovers
