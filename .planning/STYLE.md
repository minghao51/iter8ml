# iter8ml — Code Style & Conventions

## File Organization

### Where Things Go

```
src/iter8ml/
  __init__.py            # Public API re-exports + __version__
  config.py              # Pydantic settings/models for experiment configuration
  constants.py           # Enums (TaskType, CVStrategy, ModelName, TrackerType, etc.)
  exceptions.py          # Custom exception hierarchy + @track_errors decorator
  session.py             # ExperimentSession — primary programmatic API
  workspace.py           # Workspace dataclass — paths to artifacts, registry, logs
  cli/
    main.py              # Typer app assembly + init/hardware commands
    run.py               # `iter8 run` command
    optimize.py          # `iter8 hpo` command
    analyze.py           # `iter8 analyze` command
    export.py            # `iter8 export` command
  data/
    loader.py            # Polars-based CSV/Parquet ingestion
    quality.py           # Data quality auditing
    leakage.py           # Target leakage detection
    features.py          # Feature engineering engine
    embedding.py         # High-cardinality embedding engine
    adapter.py           # Data adaptation layer
    cache.py             # Data caching utilities
  engine/
    trainer.py           # Trainer — thin orchestrator tying config+data+model
    evaluator.py         # Cross-validation + metric computation
    tracker.py           # Tracker Protocol + JSONLTracker implementation
    hpo.py               # Optuna study factory + optimize_model
    hpo_importance.py    # Hyperparameter importance (PedAnova)
    hpo_warmstart.py     # Warmstart from historical trials
    calibration.py       # Platt/isotonic calibration
    state_observer.py    # State snapshot generation (optionally LLM-enriched)
    models/
      base.py            # AbstractModel Protocol (fit/predict/predict_proba/save/load)
      factory.py         # Plugin-based model class factory with lazy imports
      model_configs.py   # Per-model Pydantic config + HPO search spaces
      catboost_model.py  # CatBoostModel wrapper
      lightgbm_model.py  # LightGBMModel wrapper
      xgboost_model.py   # XGBoostModel wrapper
      tabpfn_model.py    # TabPFNModel wrapper
      tabnet_model.py    # TabNetModel wrapper
      baselines.py       # NaiveBaseline + LinearBaseline
      sparse_embedder.py # Sparse feature embedding
    pipelines/
      executor.py        # PipelineExecutor — Hamilton DAG driver (train/drift modes)
      preprocessing.py   # Pipeline preprocessing steps
      nodes/
        prep.py          # Data preparation DAG nodes
        features.py      # Feature engineering DAG nodes
        train.py         # Training DAG nodes
        drift_detection.py # Drift detection DAG nodes
      hooks/
        tracking_hook.py # Tracker integration hook for DAG nodes
  analysis/
    drift.py             # Statistical drift detection (KS, chi-squared)
    psi.py               # Population Stability Index
    domain_classifier.py # Domain classifier drift method
    explainability.py    # SHAP-based feature importance
  services/
    registry.py          # RegistryService — thread-safe model promotion
    reporting.py         # ReportService — leaderboard + experiment reports
    export.py            # ExportService — artifact packaging
    llm.py               # LLM service for AI commentary
    mcp.py               # MCP server integration
  utils/
    io.py                # JSONL I/O + safe pickle utilities

tests/
  conftest.py            # Shared fixtures (classification_data, regression_data, tmp_workspace)
  strategies.py          # Shared Hypothesis strategy generators
  unit/                  # Fast, isolated tests (57 files)
  integration/           # Multi-component tests (8 files)
  e2e/                   # Full workflow smoke tests (1 file)

benchmarks/              # Performance benchmarks + OpenML sweeps
scripts/                 # Dev scripts (check_legacy_namespace, generate_notebook_docs)
docs/                    # MkDocs documentation source
notebooks/               # Quarto .qmd notebooks
```

## Naming Conventions

### Python

| Element | Convention | Example |
|---------|-----------|---------|
| Package dirs | `snake_case` | `engine/`, `data/`, `analysis/` |
| Module files | `snake_case.py` | `model_configs.py`, `hpo_warmstart.py` |
| Classes (models/configs) | `PascalCase` | `ExperimentConfig`, `CatBoostModel`, `LeaderboardEntry` |
| Classes (services) | `PascalCase` + `Service` suffix | `RegistryService`, `ExportService`, `ReportService` |
| Classes (protocols) | `PascalCase` | `AbstractModel`, `Tracker` |
| Classes (exceptions) | `PascalCase` + `Error` suffix | `DataLoadError`, `ModelFitError`, `TabularBlueprintError` |
| Enums | `PascalCase` | `TaskType`, `CVStrategy`, `ModelName` |
| Enum members | `UPPER_SNAKE_CASE` | `STRATIFIED`, `KFOLD`, `CLASSIFICATION` |
| Functions | `snake_case` | `load_data()`, `get_model_class()`, `validate_model_name()` |
| Private helpers | `_leading_underscore` | `_raise_if_unknown_model_names()`, `_build_pruner()` |
| Constants (module-level) | `UPPER_SNAKE_CASE` | `DEFAULT_LLM_MODEL`, `METRICS_REGISTRY`, `LOWER_IS_BETTER_METRICS` |
| Config fields | `snake_case` | `cv_folds`, `target_col`, `run_hpo` |
| Test files | `test_<module>.py` | `test_config.py`, `test_model_factory.py`, `test_drift.py` |
| Test functions | `test_<description>` | `test_default_config()`, `test_get_model_class_known_model()` |
| Fixtures | `snake_case` | `classification_data`, `tmp_workspace` |

## Python Patterns

### Pydantic Models
- Config models use `BaseModel` with typed fields + `Field(default_factory=...)` for mutable defaults
- Nested configs composed as sub-models (`HPOConfig`, `QualityConfig`, `EmbeddingConfig`)
- Validation via `@field_validator`, `@model_validator(mode="before"|"after")`
- Enums as field types for constrained choices (`TaskType`, `CVStrategy`)
- `@field_serializer` for custom JSON serialization of enums
- Result/data models use `BaseModel` (e.g., `LeaderboardEntry`, `PromotionResult`, `DriftReport`)

### Protocols
- Use `typing.Protocol` for structural subtyping (e.g., `AbstractModel`, `Tracker`)
- `TYPE_CHECKING` guard for imports only needed for type annotations
- `if TYPE_CHECKING: from iter8ml.workspace import Workspace`

### Model Wrappers
- Each model is a plain class conforming to `AbstractModel` Protocol (no inheritance)
- Constructor accepts `task: str`, model-specific kwargs, and `**kwargs: Any`
- Internal state prefixed with `_` (`_model`, `_n_classes`, `_value`)
- `save()`/`load()` for serialization; `model_name` as `@property`

### Plugin Discovery
- Entry points declared in `pyproject.toml` under `[project.entry-points."iter8ml.models"]`
- `factory.py` merges built-in registry with `importlib.metadata.entry_points()`
- Lazy imports: `importlib.import_module()` on first access, cached in `_MODEL_CLASS_CACHE`

### CLI
- Built with `typer` — single `app = typer.Typer()` in `cli/main.py`
- Subcommands as separate modules importing `from .main import app`
- Options via `typer.Option(...)` with `--long` / `-s`hort flags
- Errors reported via `typer.echo()` + `raise typer.Exit(1)`

### Services
- Service classes take `workspace: Workspace` in `__init__`
- File locking via `filelock.FileLock` for thread/process safety
- `classmethod` constructors for alternate construction (`from_workspace()`)

### Error Handling
- Hierarchy: `TabularBlueprintError` → `DataLoadError` / `ModelFitError` / `RegistryError`
- `@track_errors()` decorator catches bare exceptions, logs to tracker, re-raises typed errors
- `context: dict` parameter on base exception for structured error metadata

### Data Layer
- Polars (not Pandas) as the DataFrame library throughout
- `load_data()` dispatches on file suffix (`.csv` → `load_csv`, `.parquet` → `load_parquet`)
- Config loading dispatches on suffix (`.yaml`, `.toml`, `.json`, `.py`) via `ExperimentConfig.from_file()`

### Workspace Pattern
- `Workspace` is a `@dataclass` with `Path` properties for each artifact location
- `workspace.init()` creates directories and touches files idempotently
- Env var `ITER8ML_WORKSPACE` overrides default root (`workspace/`)

## Testing

### Framework & Runner
- **pytest** with `--strict-markers`, `--import-mode=importlib`
- **hypothesis** for property-based tests
- Run via `uv run pytest`

### File Organization
```
tests/
  conftest.py         # Session-scoped fixtures (classification_data, regression_data)
  strategies.py       # Shared @st.composite generators (dataframes, numpy_arrays, jsonl_events)
  unit/               # ~57 files, fast isolated tests
  integration/        # 8 files, multi-component tests, session-scoped conftest
  e2e/                # Full pipeline smoke tests
```

### Markers
Defined in `pyproject.toml`:
- `unit`, `integration`, `e2e` — test tier
- `slow` — >1s tests
- `serial` — cannot run in parallel
- `network` — needs internet/auth
- `smoke` — critical path tests
- `property` — hypothesis property-based tests
- `metamorphic`, `contract`, `differential` — AI/ML testing strategies

### Auto-marking
`conftest.py:pytest_collection_modifyitems` auto-adds markers based on directory:
- `unit/` → `@pytest.mark.unit`
- `integration/` → `@pytest.mark.integration` + `@pytest.mark.slow`
- `e2e/` → `@pytest.mark.e2e` + `@pytest.mark.slow`

### Test Patterns
- Fixtures use `scope="session"` for expensive data generation (`make_classification`, `make_regression`)
- `tmp_workspace` fixture provides `tmp_path / "workspace"` for isolation
- Hypothesis strategies in `tests/strategies.py` shared across property tests
- Tests import from `iter8ml.*` (not relative) due to `pythonpath = ["src"]`

### CI Commands
```bash
uv run pytest tests/unit/ -v --tb=short
uv run pytest tests/integration/ -v --tb=short
uv run pytest tests/e2e/ -v --tb=short
uv run pytest tests/unit/ tests/integration/ tests/e2e/ \
  --cov=src/iter8ml/engine --cov=src/iter8ml/services --cov=src/iter8ml/config.py \
  --cov-report=xml --cov-report=term-missing --cov-fail-under=70
```

## Linting & Formatting

### Ruff
- `line-length = 100`, `target-version = "py311"`
- Rule selection: `E, F, I, UP, B, SIM, C4, PT, RUF` (pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear, simplify, comprehensions, pytest, ruff-specific)
- All rules fixable (`fixable = ["ALL"]`)
- CLI commands exempted from `B008` (function-call-in-default-arg)
- Notebooks exempted from `E402, I001, B018, E501, RUF001, F841`

### mypy
- `python_version = "3.11"`, `disallow_untyped_defs = true`
- Excluded: `tests/`, `benchmarks/`, `notebooks/`, `workspace/`
- `ignore_missing_imports = true` for ML/data libraries (sklearn, polars, torch, optuna, etc.)

### pre-commit hooks
1. **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-merge-conflict**, **debug-statements** (pre-commit-hooks)
2. **check-added-large-files** (max 1024 KB, excludes docs/notebooks)
3. **ruff-check --fix** + **ruff-format** (excludes notebooks)
4. **pip-audit** (`uv run pip-audit --skip-editable`)
5. **mypy** (`uv run mypy .`)
6. **quarto-render** (renders staged `.qmd` notebooks via `make notebooks-staged`)

## Build/Dev Commands

```
uv sync                          → Install all dependencies (prod + dev)
uv sync --extra full             → Install with all optional deps (train + docs)
uv run ruff check .              → Lint with ruff
uv run ruff format .             → Format with ruff
uv run mypy .                    → Type-check with mypy
uv run pytest tests/unit/        → Run unit tests
uv run pytest tests/             → Run all tests
uv run pytest -m property        → Run property-based tests only
uv run pytest -m "not slow"      → Skip slow tests
pre-commit run --all-files       → Run all pre-commit hooks
make docs                        → Render notebooks + build MkDocs site
make check-legacy-namespace      → Verify no legacy namespace imports
uv run iter8 run -c config.yaml  → Run experiment via CLI
```
