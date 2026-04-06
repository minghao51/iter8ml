# Structure

## Complete Directory Layout

```
iter8ml/
├── main.py                          # CLI entry point (Typer)
├── pyproject.toml                   # Project config, dependencies, scripts
├── uv.lock                          # Dependency lock file
│
├── configs/                         # Configuration layer
│   ├── __init__.py                  # Exports ExperimentConfig
│   ├── experiment.py                # ExperimentConfig Pydantic model
│   ├── hardware.py                  # HardwareProfile auto-detection
│   ├── model_configs.py             # Per-model defaults + HPO search spaces
│   └── examples/
│       └── credit_risk.py           # Example experiment config
│
├── core/                            # Core ML library
│   ├── __init__.py                  # Package docstring
│   ├── data/                        # Data ingestion & preprocessing
│   │   ├── __init__.py              # Exports loaders
│   │   ├── loaders.py               # CSV, Parquet, SQLite loaders + data hashing
│   │   ├── processors.py            # Polars-native preprocessing pipeline
│   │   ├── adapter.py               # DataFrame → numpy/tensor/dataset converter
│   │   └── quality.py               # Cleanlab-based label noise detection
│   ├── engine/                      # Experiment orchestration
│   │   ├── __init__.py              # Exports Evaluator, JSONLTracker, Trainer
│   │   ├── trainer.py               # Main experiment orchestrator + model registry
│   │   ├── evaluator.py             # Cross-validation + metrics computation
│   │   ├── hpo.py                   # Optuna-based hyperparameter optimization
│   │   ├── tracker.py               # Pluggable tracking (JSONL, W&B, MLflow)
│   │   └── state_observer.py        # LLM-readable state summary generator
│   ├── models/                      # Model implementations
│   │   ├── __init__.py              # Exports AbstractModel, ModelSelector
│   │   ├── base.py                  # AbstractModel Protocol (structural subtyping)
│   │   ├── selector.py              # Hardware/data-size-aware model routing
│   │   ├── conventional/            # GBDT models
│   │   │   ├── __init__.py
│   │   │   ├── catboost_model.py    # CatBoostClassifier/Regressor wrapper
│   │   │   ├── lightgbm_model.py    # LightGBM wrapper
│   │   │   └── xgboost_model.py     # XGBoost wrapper
│   │   ├── tabular_foundation/      # Foundation models for tabular data
│   │   │   ├── __init__.py
│   │   │   └── tabpfn_model.py      # TabPFN v2 wrapper with row guardrail
│   │   └── deep/                    # Deep learning models
│   │       ├── __init__.py
│   │       ├── ft_transformer.py    # FT-Transformer with accelerate support
│   │       └── text_encoder.py      # DeBERTa-v3 text embedding extractor
│   └── monitoring/                  # Production monitoring
│       ├── __init__.py
│       └── drift.py                 # Statistical drift detection (KS, Chi-squared)
│
├── mcp_server/                      # MCP server for LLM agents
│   ├── __init__.py
│   └── tools.py                     # 8 MCP tools (state, stats, baseline, hpo, etc.)
│
├── pipelines/                       # Hamilton DAG pipelines (optional)
│   └── __init__.py                  # Package docstring
│
├── workspace/                       # Runtime workspace
│   ├── .gitkeep
│   ├── artifacts/                   # Saved model artifacts
│   ├── current_state.md             # Current experiment state
│   ├── experiments.jsonl            # Append-only event log
│   ├── leaderboard.md               # Auto-generated leaderboard
│   ├── registry.json                # Champion model registry
│   └── registry.lock                # File lock for registry
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_adapter.py
│   │   ├── test_cli.py
│   │   ├── test_config.py
│   │   ├── test_drift.py
│   │   ├── test_ft_transformer.py
│   │   ├── test_hpo.py
│   │   ├── test_loaders.py
│   │   ├── test_mcp_tools.py
│   │   ├── test_model_selector.py
│   │   ├── test_processors.py
│   │   ├── test_quality.py
│   │   ├── test_state_observer.py
│   │   └── test_tabpfn.py
│   └── integration/
│       ├── __init__.py
│       ├── test_full_pipeline.py
│       └── test_gdbt_models.py
│
├── examples/
│   └── zenml_pipeline.py            # Commented ZenML integration example
│
├── notebooks/
│   └── quick_start.py               # Notebook-style quick start guide
│
├── .planning/
│   └── codebase/
│       ├── ARCHITECTURE.md          # Architecture documentation
│       └── STRUCTURE.md             # This file
│
├── .github/                         # GitHub workflows
├── .devcontainer/                   # Dev container config
├── .venv/                           # Virtual environment
├── .ruff_cache/                     # Ruff linter cache
├── .pytest_cache/                   # Pytest cache
├── __pycache__/                     # Python bytecode cache
├── catboost_info/                   # CatBoost training info
│
├── AGENTS.md                        # AI agent instructions
├── CLAUDE.md                        # Claude-specific instructions
├── README.md                        # Project documentation
├── CHANGELOG.md                     # Version history
├── CONTRIBUTING.md                  # Contribution guidelines
├── technical_roadmap.md             # Technical roadmap
├── LICENSE                          # MIT license
├── Dockerfile                       # Container build
├── docker-compose.yml               # Container orchestration
└── .pre-commit-config.yaml          # Pre-commit hooks
```

## Key File Locations and Their Roles

| File | Role |
|------|------|
| `main.py` | CLI entry point — all user-facing commands |
| `pyproject.toml` | Project metadata, dependencies, scripts, tool configs |
| `configs/experiment.py` | Central experiment configuration model |
| `configs/hardware.py` | GPU/CPU/RAM auto-detection |
| `configs/model_configs.py` | Default hyperparameters + HPO search spaces per model |
| `core/engine/trainer.py` | Main orchestrator — ties config + data + models together |
| `core/engine/evaluator.py` | Cross-validation engine + metrics registry |
| `core/engine/hpo.py` | Optuna study factory |
| `core/engine/tracker.py` | Pluggable tracking protocol + implementations |
| `core/engine/state_observer.py` | Generates LLM-readable state summaries |
| `core/models/base.py` | AbstractModel Protocol — the model interface contract |
| `core/models/selector.py` | Model routing logic based on data size + hardware |
| `core/data/loaders.py` | Polars-based data ingestion |
| `core/data/adapter.py` | Format conversion (Polars → numpy/tensor/dataset) |
| `core/data/processors.py` | Preprocessing functions (nulls, dates, categoricals) |
| `core/data/quality.py` | Cleanlab-based label noise detection |
| `core/monitoring/drift.py` | Production drift detection |
| `mcp_server/tools.py` | MCP tools for LLM agent interaction |

## Naming Conventions Observed

- **Modules:** snake_case (`model_configs.py`, `state_observer.py`)
- **Classes:** PascalCase (`ExperimentConfig`, `Trainer`, `AbstractModel`)
- **Functions:** snake_case (`load_csv`, `optimize_model`, `audit_data_quality`)
- **Model wrappers:** `{Framework}Model` pattern (`CatBoostModel`, `TabPFNModel`)
- **Config classes:** `{Model}Config` pattern (`CatBoostConfig`, `TabPFNConfig`)
- **Test files:** `test_{module}.py` pattern
- **Internal helpers:** leading underscore (`_get_model_class`, `_build_model`, `_to_numpy`)

## Module Organization

### By Responsibility (Layered)
1. **configs/** — Configuration models (Pydantic)
2. **core/data/** — Data ingestion and preprocessing
3. **core/engine/** — Experiment orchestration
4. **core/models/** — Model implementations (by family)
5. **core/monitoring/** — Production monitoring
6. **mcp_server/** — LLM agent interface
7. **pipelines/** — Optional DAG-based pipelines

### By Model Family
- `core/models/conventional/` — CatBoost, LightGBM, XGBoost (GBDTs)
- `core/models/tabular_foundation/` — TabPFN (foundation model)
- `core/models/deep/` — FT-Transformer, TextEncoder (PyTorch)

### Test Organization
- `tests/unit/` — Unit tests per module
- `tests/integration/` — End-to-end pipeline tests

## Workspace Runtime Structure

Created by `tabblueprint init` and populated during experiment runs:

```
workspace/
├── experiments.jsonl    # Append-only JSONL event log (started, completed, failed, metrics)
├── registry.json        # Champion model registry (best score per experiment:task)
├── registry.lock        # File lock for concurrent registry updates
├── artifacts/           # Saved model files (framework-native format or pickle)
├── leaderboard.md       # Auto-generated markdown table of results
└── current_state.md     # LLM-readable experiment state summary
```
