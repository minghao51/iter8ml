# Structure

> Last updated: 2026-04-23

## Top-Level Directory Layout

```
iter8ml/
├── src/                        # Source code (src layout)
├── tests/                      # Test suite
├── examples/                   # Example configs and pipelines
├── docs/                       # Documentation and plans
├── workspace/                  # Runtime workspace (experiments, registry, artifacts)
├── notebooks/                  # Jupyter notebooks
├── .github/                    # CI/CD and Dependabot config
├── .claude/                    # Claude handoff sessions
├── .devcontainer/              # Dev container config
├── .planning/                  # Planning and codebase documentation
├── pyproject.toml              # Project metadata, dependencies, tool config
├── Dockerfile                  # GPU-enabled Docker image (CUDA 12.4 + uv)
├── docker-compose.yml          # App + MLflow server
├── .pre-commit-config.yaml     # Pre-commit hooks (ruff, pytest)
├── .gitignore                  # Git ignore rules
├── AGENTS.md                   # Agent workflow instructions
├── CLAUDE.md                   # Claude-specific config
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT license
├── README.md                   # Project overview
├── strategic_roadmap.md        # Strategic planning document
├── technical_roadmap.md        # Technical planning document
└── uv.lock                    # Locked dependency tree
```

## Source Structure

```
src/tabular_blueprint/
├── __init__.py                 # Package init (typed marker)
├── py.typed                    # PEP 561 typed package marker
├── cli.py                      # CLI entry point (Typer app with 10 commands)
├── config.py                   # ExperimentConfig + HardwareProfile (Pydantic)
├── constants.py                # Enums: TaskType, CVStrategy, ModelName, TrackerType
├── exceptions.py               # Custom exception hierarchy + @track_errors decorator
│
├── data/                       # Data ingestion and transformation
│   ├── __init__.py             # Exports: load_csv, load_parquet, load_sqlite, get_data_hash
│   ├── loaders.py              # Polars-based data loading (CSV, Parquet, SQLite)
│   ├── adapter.py              # DataAdapter: Polars → NumPy/Tensor/HF Dataset
│   ├── feature_engine.py       # AFE: target transform, interaction discovery, pruning
│   ├── quality.py              # Data quality audit and noise cleaning (Cleanlab)
│   └── leakage.py              # Feature leakage detection
│
├── engine/                     # Core orchestration and evaluation
│   ├── __init__.py             # Exports: Trainer, Evaluator, JSONLTracker
│   ├── trainer.py              # Trainer: main orchestrator (647 lines, largest file)
│   ├── evaluator.py            # Evaluator: cross-validation + metrics computation
│   ├── tracker.py              # Tracker Protocol + JSONLTracker, WandbTracker, MLflowTracker
│   ├── hpo.py                  # Optuna HPO with warmstart support
│   ├── hpo_warmstart.py        # Historical trial injection from JSONL
│   ├── hpo_importance.py       # Hyperparameter importance (PedAnova)
│   ├── calibration.py          # Platt scaling and isotonic regression calibration
│   └── state_observer.py       # Generates current_state.md + leaderboard.md
│
├── models/                     # Model implementations and management
│   ├── __init__.py             # Exports: AbstractModel, ModelSelector, factory functions
│   ├── base.py                 # AbstractModel Protocol (structural subtyping)
│   ├── factory.py              # Lazy model registry: name → (module, class)
│   ├── selector.py             # Hardware/data-aware model routing
│   ├── gbdt_base.py            # BaseGBDTModel abstract class
│   ├── baselines.py            # NaiveBaseline + LinearBaseline
│   ├── model_configs.py        # Per-model Pydantic configs with HPO search spaces
│   ├── conventional/           # GBDT model implementations
│   │   ├── __init__.py
│   │   ├── catboost_model.py   # CatBoost wrapper
│   │   ├── lightgbm_model.py   # LightGBM wrapper
│   │   └── xgboost_model.py    # XGBoost wrapper
│   ├── tabular_foundation/     # Foundation model implementations
│   │   ├── __init__.py
│   │   └── tabpfn_model.py     # TabPFN wrapper
│   └── deep/                   # Deep learning model implementations
│       ├── __init__.py
│       ├── ft_transformer.py   # FT-Transformer wrapper
│       ├── tabnet_model.py     # TabNet wrapper
│       └── text_encoder.py     # Text encoding for mixed-type features
│
├── pipelines/                  # Data pipeline definitions
│   ├── __init__.py             # Pipeline visualization export
│   ├── hamilton_executor.py    # Hamilton Driver orchestration
│   └── preprocessing.py        # Hamilton DAG functions (null fill → dates → encode)
│
├── monitoring/                 # Model monitoring and explainability
│   ├── __init__.py
│   ├── drift.py                # KS-test / Chi-squared drift detection
│   ├── psi_drift.py            # PSI (Population Stability Index) drift
│   ├── domain_classifier.py    # Classifier-based drift detection
│   └── explainability.py       # SHAP feature importance + plots
│
├── services/                   # Business logic services
│   ├── __init__.py             # Exports: RegistryService, ReportService, dataclasses
│   ├── registry_service.py     # Thread-safe model registry (file-locked JSON)
│   ├── report_service.py       # Leaderboard + experiment report generation
│   └── export_service.py       # Champion model packaging for portable inference
│
├── llm/                        # LLM integration
│   └── __init__.py             # TabularAgent: LiteLLM-backed SHAP/performance explanations
│
├── mcp/                        # MCP server for LLM agent tools
│   ├── __init__.py
│   └── tools.py                # FastMCP server exposing 8 atomic tools
│
└── utils/                      # Shared utilities
    ├── __init__.py
    └── jsonl.py                # JSONL file reader
```

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures (classification_data, regression_data, tmp_workspace)
│                               # Auto-marker: unit/ → @unit, integration/ → @integration @slow
│
├── unit/                       # Fast isolated tests (~30 files)
│   ├── test_config.py          # ExperimentConfig validation
│   ├── test_cli.py             # CLI command tests
│   ├── test_trainer.py         # Trainer unit tests
│   ├── test_loaders.py         # Data loading tests
│   ├── test_adapter.py         # DataAdapter format conversion
│   ├── test_feature_engine.py  # AFE functions
│   ├── test_quality.py         # Quality audit
│   ├── test_leakage.py         # Leakage detection
│   ├── test_baselines.py       # Baseline model tests
│   ├── test_model_factory.py   # Model factory + registry
│   ├── test_model_selector.py  # Model routing logic
│   ├── test_tabpfn.py          # TabPFN model tests
│   ├── test_tabpfn_guardrails.py # TabPFN row limit warnings
│   ├── test_ft_transformer.py  # FT-Transformer tests
│   ├── test_hpo_unit.py        # HPO search space validation
│   ├── test_hpo_importance.py  # Hyperparameter importance
│   ├── test_hpo_warmstart.py   # HPO warmstart injection
│   ├── test_calibration.py     # Calibration tests
│   ├── test_drift.py           # KS/Chi2 drift tests
│   ├── test_psi_drift.py       # PSI drift tests
│   ├── test_domain_classifier.py # Domain classifier drift
│   ├── test_jsonl.py           # JSONL utilities
│   ├── test_exceptions.py      # Exception hierarchy
│   ├── test_tracker_rotation.py # Log rotation tests
│   ├── test_state_observer.py  # State generation
│   ├── test_registry_service.py # Registry CRUD
│   ├── test_report_service.py  # Report formatting
│   ├── test_export_service.py  # Export packaging
│   ├── test_processors.py      # Preprocessing tests
│   ├── test_llm_agent.py       # LLM agent tests
│   └── test_mcp_tools.py       # MCP tool tests
│
├── integration/                # Integration tests (slower, external deps)
│   ├── conftest.py             # Integration-specific fixtures
│   ├── test_full_pipeline.py   # End-to-end Trainer.run() tests
│   ├── test_model_selection.py # Model selection + training integration
│   ├── test_hpo.py             # HPO integration (Optuna)
│   ├── test_gdbt_models.py     # GBDT model training integration
│   └── test_registry_and_drift.py # Registry + drift combined
│
└── e2e/                        # End-to-end workflow tests
```

## Key Locations

| Concern | Location |
|---|---|
| **CLI commands** | `src/tabular_blueprint/cli.py` |
| **Main orchestrator** | `src/tabular_blueprint/engine/trainer.py` |
| **Configuration models** | `src/tabular_blueprint/config.py` |
| **Enums/constants** | `src/tabular_blueprint/constants.py` |
| **Model contract** | `src/tabular_blueprint/models/base.py` |
| **Model factory** | `src/tabular_blueprint/models/factory.py` |
| **Model routing** | `src/tabular_blueprint/models/selector.py` |
| **Data loading** | `src/tabular_blueprint/data/loaders.py` |
| **Preprocessing DAG** | `src/tabular_blueprint/pipelines/preprocessing.py` |
| **Experiment tracking** | `src/tabular_blueprint/engine/tracker.py` |
| **Model registry** | `src/tabular_blueprint/services/registry_service.py` |
| **HPO** | `src/tabular_blueprint/engine/hpo.py` |
| **Drift detection** | `src/tabular_blueprint/monitoring/drift.py` |
| **MCP server** | `src/tabular_blueprint/mcp/tools.py` |
| **LLM agent** | `src/tabular_blueprint/llm/__init__.py` |
| **Error handling** | `src/tabular_blueprint/exceptions.py` |
| **Report generation** | `src/tabular_blueprint/services/report_service.py` |
| **Model export** | `src/tabular_blueprint/services/export_service.py` |
| **Test fixtures** | `tests/conftest.py` |
| **Example configs** | `examples/` |

## Naming Conventions

### Files
- **Modules**: `snake_case.py` (e.g., `feature_engine.py`, `state_observer.py`)
- **Test files**: `test_<module_name>.py` (e.g., `test_trainer.py`, `test_loaders.py`)
- **Config files**: `kebab-case` (e.g., `docker-compose.yml`, `pyproject.toml`)

### Directories
- **Packages**: `snake_case/` (e.g., `tabular_blueprint/`, `tabular_foundation/`)
- **Test tiers**: `unit/`, `integration/`, `e2e/`

### Code
- **Classes**: `PascalCase` (e.g., `ExperimentConfig`, `Trainer`, `BaseGBDTModel`, `JSONLTracker`)
- **Functions/methods**: `snake_case` (e.g., `load_data()`, `get_model_class()`, `detect_leakage()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `BASELINE_MODELS`, `METRICS_REGISTRY`, `LOWER_IS_BETTER_METRICS`)
- **Private members**: `_leading_underscore` (e.g., `_train_model()`, `_safe_ratio()`)
- **Protocols**: `PascalCase` with Protocol suffix implied (e.g., `AbstractModel`, `Tracker`)
- **Pydantic models**: `PascalCase` with descriptive suffix (e.g., `ExperimentConfig`, `DriftReport`, `LeaderboardEntry`)

## Special Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, dependencies (core + optional extras), ruff/mypy/pytest config, setuptools config |
| `Dockerfile` | NVIDIA CUDA 12.4 + Python 3.11 + uv, for GPU training |
| `docker-compose.yml` | App container (GPU) + MLflow server sidecar |
| `.pre-commit-config.yaml` | ruff lint/format + pytest unit tests |
| `.github/workflows/ci.yml` | CI pipeline |
| `.github/dependabot.yml` | Dependency auto-update config |
| `uv.lock` | Locked dependency resolutions |
| `workspace/` | Runtime directory (gitignored except `.gitkeep`) |
| `workspace/experiments.jsonl` | Append-only experiment event log |
| `workspace/registry.json` | Champion model registry |
| `workspace/current_state.md` | LLM-readable state summary (generated) |
| `workspace/leaderboard.md` | Ranked experiment results (generated) |
| `workspace/artifacts/` | Serialized model files |
| `examples/credit_risk.py` | Example experiment config |
| `examples/zenml_pipeline.py` | ZenML integration example |
| `AGENTS.md` | Workflow instructions for AI agents |
| `CLAUDE.md` | Claude-specific configuration |

## Monorepo Structure

This is **not a monorepo**. It is a single Python package (`tabular-blueprint`) using the **src layout** pattern:

- Package source: `src/tabular_blueprint/`
- Installable via: `uv sync` (editable install via setuptools)
- CLI entry point: `tabblueprint` (defined in `pyproject.toml [project.scripts]`)
- Optional dependency groups: `hamilton`, `llm`, `wandb`, `mlflow`, `zenml`, `transformers`, `shap`, `dl`
- Dev dependencies: `pytest`, `ruff`, `pre-commit` (in `[dependency-groups] dev`)
