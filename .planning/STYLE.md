# Tabular Blueprint — Code Style & Conventions

## File Organization

### Where Things Go

```
src/tabular_blueprint/
  __init__.py           # Package root, __all__, __typed__
  cli.py                # Typer CLI commands
  config.py             # Pydantic ExperimentConfig, HardwareProfile
  constants.py          # Enums (TaskType, CVStrategy, ModelName, TrackerType, EmbeddingMethod)
  exceptions.py         # Custom exception hierarchy + @track_errors decorator
  data/                 # Data loading, quality, leakage detection, embedding, feature engineering
    loaders.py          # load_csv, load_parquet, load_sqlite, load_data, get_data_hash
    quality.py          # Data quality audit
    leakage.py          # Permutation-based leakage detection → LeakageReport (Pydantic)
    feature_engine.py   # Feature engineering utilities
    embedding_engine.py # Entity/autoencoder embeddings
    cache.py            # Data caching
    adapter.py          # Data adapters
  engine/               # Training orchestration and evaluation
    trainer.py          # Trainer class (slim orchestrator)
    evaluator.py        # Cross-validation, metrics, lift computation
    tracker.py          # Tracker protocol + JSONLTracker, WandbTracker, MLflowTracker
    calibration.py      # Platt/isotonic calibration
    hpo.py              # Hyperparameter optimization via Optuna
    hpo_importance.py   # HPO parameter importance
    hpo_warmstart.py    # Warmstart HPO from historical JSONL runs
    state_observer.py   # Generate current_state.md and leaderboard.md
  models/               # Model wrappers and factory
    base.py             # AbstractModel Protocol (structural subtyping)
    factory.py          # _MODEL_REGISTRY dict, get_model_class(), lazy imports
    selector.py         # ModelSelector — hardware-aware model routing
    gbdt_base.py        # BaseGBDTModel ABC for gradient boosting models
    baselines.py        # NaiveBaseline, LinearBaseline
    model_configs.py    # Pydantic configs per model (CatBoostConfig, LightGBMConfig, etc.)
    conventional/       # GBDT model wrappers
      catboost_model.py
      lightgbm_model.py
      xgboost_model.py
    deep/               # Deep learning model wrappers
      ft_transformer.py
      tabnet_model.py
      sparse_embedder.py
    tabular_foundation/ # Foundation model wrappers
      tabpfn_model.py
  pipelines/            # Hamilton DAG pipelines
    executor.py         # PipelineExecutor (builds/executes Hamilton DAGs)
    preprocessing.py    # Standalone preprocessing (non-Hamilton)
    hooks/
      tracking_hook.py  # Hamilton lifecycle hook for event tracking
    nodes/              # Hamilton node functions (one function = one DAG node)
      preprocessing.py
      data_preparation.py
      model_selection.py
      baselines.py
      feature_engineering.py
      model_training.py
      state_generation.py
      drift_detection.py
  monitoring/           # Drift detection and explainability
    drift.py            # KS/Chi2 drift detection
    psi_drift.py        # PSI-based drift detection
    domain_classifier.py# Domain classifier drift
    explainability.py   # SHAP explanations
  services/             # Business logic services
    registry_service.py # Thread-safe model registry (FileLock, atomic writes)
    report_service.py   # Leaderboard/report generation
    export_service.py   # Export champion models as portable packages
  utils/                # Shared utilities
    jsonl.py            # load_events, iter_events
    safe_pickle.py      # Restricted pickle load/dump
  llm/                  # LLM integration (stub)
  mcp/                  # MCP server tools
    tools.py

tests/
  conftest.py           # Shared fixtures (classification_data, regression_data, tmp_workspace)
  unit/                 # Fast, isolated, no external deps
  integration/          # Multi-component tests, may require optional deps
  e2e/                  # Full workflow smoke tests

notebooks/              # Quarto .qmd tutorials
docs/                   # MkDocs static site (generated from notebooks + mkdocs.yml)
scripts/                # Build/utility scripts
benchmarks/             # Performance benchmarks
workspace/              # Runtime workspace (experiments.jsonl, registry.json, artifacts/)
```

## Naming Conventions

### Python

| Element | Convention | Example |
|---------|-----------|---------|
| Package | `snake_case` | `tabular_blueprint` |
| Module | `snake_case` | `model_configs.py`, `feature_engine.py` |
| Class | `PascalCase` | `ExperimentConfig`, `CatBoostModel`, `PipelineExecutor` |
| Pydantic model | `PascalCase` | `LeakageReport`, `PromotionResult`, `HardwareProfile` |
| Enum | `PascalCase` enum, `UPPER_SNAKE` members | `TaskType.CLASSIFICATION`, `CVStrategy.STRATIFIED` |
| Function | `snake_case` | `load_data()`, `get_cv_split()`, `detect_leakage()` |
| Private function | `_leading_underscore` | `_load_completed_models()`, `_get_module()` |
| Method | `snake_case` | `model.fit()`, `trainer.run()`, `selector.select()` |
| Private method | `_leading_underscore` | `self._build_model()`, `self._rotate_log()` |
| Protocol | `PascalCase` | `Tracker`, `AbstractModel` |
| Constant (module-level) | `UPPER_SNAKE` or `snake_case` | `METRICS_REGISTRY`, `DEFAULT_LLM_MODEL`, `_MODEL_REGISTRY` |
| Dataclass | `PascalCase` | `ModelResult` |
| CLI command | `snake_case` (Typer) | `tabblueprint run`, `tabblueprint hpo`, `tabblueprint drift` |
| Pydantic field | `snake_case` | `cv_folds`, `target_col`, `data_path` |
| Fixture | `snake_case` | `classification_data`, `regression_data`, `tmp_workspace` |
| Test function | `test_` prefix, `snake_case` | `test_default_config()`, `test_load_csv()` |

## Python Patterns

### Pydantic Models
- Use `BaseModel` for all structured data: config (`ExperimentConfig`), reports (`LeakageReport`), service results (`PromotionResult`), hardware profiles (`HardwareProfile`)
- Validation via `@field_validator` and `@model_validator` decorators
- Serialization via `@field_serializer` with `when_used="json"`
- Config loading via `@classmethod` factory methods (`ExperimentConfig.from_file()`)
- Per-model hyperparameter configs use separate Pydantic models with `hpo_search_space()` method

### Protocols vs ABCs
- Use `typing.Protocol` for structural subtyping (duck typing): `AbstractModel`, `Tracker`
- Use `abc.ABC` / `@abstractmethod` for inheritance-based extension: `BaseGBDTModel`
- Models conform to `AbstractModel` Protocol implicitly (no inheritance required)

### Model Factory Pattern
- `_MODEL_REGISTRY` dict maps model name strings to `(module_path, class_name)` tuples
- `get_model_class()` uses `importlib.import_module()` for lazy loading
- Model classes cached in `_MODEL_CLASS_CACHE` after first import
- Enum `ModelName` defines canonical names but registry uses string keys

### Hamilton DAG Nodes
- Each function in `pipelines/nodes/` is a Hamilton node
- Function parameter names = DAG dependencies (resolved by name)
- Return value name = node output name
- `PipelineExecutor` builds driver with `Builder().with_modules(*modules).with_config({...}).build()`
- `PipelineMode` enum selects which nodes to include (TRAINING, DRIFT, EXPORT, HPO, INFERENCE)
- Hamilton adapters used for cross-cutting concerns (`TrackingHook`)

### Exception Hierarchy
- `TabularBlueprintError` is the base exception, accepts `context: dict[str, Any]`
- Domain-specific subclasses: `DataLoadError`, `ModelFitError`, `RegistryError`
- `@track_errors()` decorator catches exceptions and re-raises as typed errors with tracker logging
- Exception chaining with `raise ... from e` throughout

### Tracker Pattern
- `Tracker` Protocol defines: `log_metrics`, `log_params`, `log_artifact`, `log_event`, `finish`
- `JSONLTracker` is default; `WandbTracker` and `MLflowTracker` are optional extras
- All events are dicts with `"event"` key (e.g., `"experiment_started"`, `"model_completed"`)
- `JSONLTracker` includes log rotation (size-based, with backup files)

### Lazy Imports
- Optional heavy dependencies imported inside functions: `import torch`, `import wandb`, `import mlflow`
- `ImportError` caught and handled gracefully (feature disabled or raised with install instructions)
- `TYPE_CHECKING` guard for type-only imports

### CLI (Typer)
- Single `app = typer.Typer()` in `cli.py`, entry point registered in `pyproject.toml` as `tabblueprint`
- Commands are decorated `@app.command()` functions
- `typer.Option()` for flags with `--long-form` names
- `typer.Exit(1)` for error exits, `typer.echo()` for output

### Thread Safety
- `FileLock` (from `filelock` package) for process-safe registry operations
- `threading.Lock` in `JSONLTracker` for thread-safe writes
- Atomic file writes via `tempfile.mkstemp()` + `os.replace()`

### Type Annotations
- Modern union syntax: `str | None` (not `Optional[str]`)
- `from __future__ import annotations` in pipeline nodes
- `type: ignore[...]` comments for untyped third-party libs
- `py.typed` marker file included in package

## Testing

### Python (pytest)
- **Runner**: `uv run pytest` from project root
- **Async**: No async tests (synchronous codebase)
- **File naming**: `test_<module_name>.py` mirroring source module names (e.g., `test_config.py` tests `config.py`)
- **Test organization**: Three-tier directory structure:
  - `tests/unit/` — isolated, fast, no optional deps (46 files)
  - `tests/integration/` — multi-component, may need optional deps (8 files)
  - `tests/e2e/` — full workflow smoke tests
- **Markers**: Auto-applied by `conftest.py:pytest_collection_modifyitems` based on directory:
  - `unit` — auto-applied to `tests/unit/`
  - `integration` + `slow` — auto-applied to `tests/integration/`
  - `e2e` + `slow` — auto-applied to `tests/e2e/`
  - `smoke` — manually applied with `@pytest.mark.smoke`
  - Registered in `pyproject.toml` with `--strict-markers`
- **Fixtures**: Session-scoped for expensive data (`classification_data`, `regression_data`); function-scoped for temp dirs (`tmp_workspace` using `tmp_path`)
- **Mocking**: `unittest.mock.patch` for environment variables and external deps; no mocking framework
- **Config**: `pythonpath = ["src"]`, `--import-mode=importlib`
- **Test data**: Generated via `sklearn.datasets.make_classification` / `make_regression`, wrapped in `polars.DataFrame`

## Linting & Formatting

### Python
- **Ruff** (v0.15.9) — linter + formatter, configured in `pyproject.toml`
  - `line-length = 100`, `target-version = "py311"`
  - Rule selection: `E, F, I, UP, B, SIM, C4, PT, RUF`
  - Auto-fix enabled (`fixable = ["ALL"]`)
  - Notebooks excluded from linting (`exclude: ^notebooks/`)
- **mypy** — strict type checking (`disallow_untyped_defs = true` for `src/`)
  - `tests/`, `benchmarks/`, `notebooks/`, `workspace/` excluded
  - Missing imports ignored for third-party libs via overrides
- **pre-commit** — runs ruff (fix + format), mypy, uv-lock, and quarto-render
  - Notebooks excluded from ruff hooks

## Build/Dev Commands

```
uv sync                        → Install all dependencies from uv.lock
uv run pytest                  → Run all tests
uv run pytest tests/unit/      → Run unit tests only
uv run pytest tests/integration/ → Run integration tests only
uv run pytest -m smoke         → Run smoke tests only
uv run pytest -m "not slow"    → Skip slow tests
uv run ruff check .            → Lint with ruff
uv run ruff format .           → Format with ruff
uv run mypy .                  → Type-check
uv run tabblueprint run --help → CLI help
uv run tabblueprint run -d data.csv -t target --task classification  → Run experiment
uv run tabblueprint run --quick -d data.csv -t target  → Quick experiment (2 folds, 20% data)
uv run tabblueprint hpo -d data.csv -t target -m catboost  → Run HPO
uv run tabblueprint leaderboard → Show experiment leaderboard
uv run tabblueprint drift -r ref.csv -n new.csv  → Drift detection
uv run tabblueprint export experiment:classification  → Export champion model
uv run tabblueprint hardware   → Show hardware profile
make notebooks                 → Render all Quarto notebooks
make docs                      → Build MkDocs site
pre-commit run --all-files     → Run all pre-commit hooks
```
