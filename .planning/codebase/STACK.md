# Stack

> Last updated: 2026-04-23

## Languages

- **Python 3.11+** — sole language (specified in `pyproject.toml` `requires-python = ">=3.11"` and `tool.mypy.python_version = "3.11"`)
- No frontend code; no JavaScript/TypeScript in the project

## Runtime

- **Python**: 3.11+ (installed as `python3.11` in Docker image)
- **CUDA**: 12.4.0 (Docker base `nvidia/cuda:12.4.0-runtime-ubuntu22.04`)
- **Ubuntu**: 22.04 (inside Docker container)

## Frameworks

- **Pydantic v2** — config validation, data models (`pydantic>=2.0`, `pydantic-settings>=2.0`)
- **Hamilton** (`sf-hamilton>=1.70`) — DAG-based data pipeline orchestration (`src/tabular_blueprint/pipelines/hamilton_executor.py`)
- **Typer** (`typer>=0.12`) — CLI framework (`src/tabular_blueprint/cli.py`)
- **Rich** (`rich>=13.0`) — terminal formatting and tables
- **HuggingFace Accelerate** (`accelerate>=0.30`) — distributed PyTorch training for deep models
- **HuggingFace Transformers** (`transformers>=4.40`) — transformer-based model support
- **Optuna** (`optuna>=3.6`) — hyperparameter optimization framework

## ML Libraries

### Gradient Boosted Decision Trees (GBDT)
- **CatBoost** (`catboost>=1.2`) — `src/tabular_blueprint/models/conventional/catboost_model.py`
- **LightGBM** (`lightgbm>=4.0`) — `src/tabular_blueprint/models/conventional/lightgbm_model.py`
- **XGBoost** (`xgboost>=2.0`) — `src/tabular_blueprint/models/conventional/xgboost_model.py`

### Tabular Foundation Models
- **TabPFN v2** (`tabpfn>=2.0`) — `src/tabular_blueprint/models/tabular_foundation/tabpfn_model.py`

### Deep Learning
- **PyTorch** (`torch>=2.3`) — core tensor ops, FT-Transformer, TabNet
- **FT-Transformer** (custom) — `src/tabular_blueprint/models/deep/ft_transformer.py`
- **TabNet** (via `pytorch-tabular>=1.0`, optional) — `src/tabular_blueprint/models/deep/tabnet_model.py`

### Data & Feature Engineering
- **Polars** (`polars>=1.0`) — primary DataFrame library (replaces pandas)
- **scikit-learn** (`scikit-learn>=1.4`) — metrics, cross-validation, baselines
- **NumPy** (`numpy>=1.26`) — array operations
- **skrub** (`skrub>=0.3`) — tabular data preparation utilities
- **Cleanlab** (`cleanlab>=2.6`) — label noise detection (`src/tabular_blueprint/data/quality.py`)

### Explainability
- **SHAP** (`shap>=0.44`) — feature importance and explainability (`src/tabular_blueprint/monitoring/explainability.py`)

## Dependencies (Production)

From `pyproject.toml` `[project.dependencies]`:

| Package | Version | Purpose |
|---------|---------|---------|
| polars | >=1.0 | DataFrame library |
| pydantic | >=2.0 | Config/validation models |
| pydantic-settings | >=2.0 | Settings management |
| catboost | >=1.2 | GBDT model |
| lightgbm | >=4.0 | GBDT model |
| xgboost | >=2.0 | GBDT model |
| tabpfn | >=2.0 | Foundation model |
| skrub | >=0.3 | Data preparation |
| cleanlab | >=2.6 | Label noise detection |
| optuna | >=3.6 | Hyperparameter optimization |
| torch | >=2.3 | Deep learning |
| accelerate | >=0.30 | Distributed training |
| transformers | >=4.40 | Transformer models |
| scikit-learn | >=1.4 | ML utilities |
| numpy | >=1.26 | Array operations |
| typer | >=0.12 | CLI framework |
| psutil | >=5.9 | Hardware detection |
| rich | >=13.0 | Terminal output |
| sf-hamilton | >=1.70 | DAG pipeline orchestration |
| shap | >=0.44 | Model explainability |

## Optional Dependencies

From `pyproject.toml` `[project.optional-dependencies]`:

| Extra | Package | Purpose |
|-------|---------|---------|
| hamilton | sf-hamilton>=1.70 | DAG pipelines |
| llm | mcp>=0.9, litellm>=1.40 | LLM agent + MCP server |
| wandb | wandb>=0.17 | W&B experiment tracking |
| mlflow | mlflow>=2.13 | MLflow experiment tracking |
| zenml | zenml>=0.57 | ZenML orchestration |
| transformers | datasets>=2.14 | HuggingFace datasets |
| shap | shap>=0.44 | Explainability |
| dl | pytorch-tabular>=1.0 | Deep tabular models |

## Dev Dependencies

From `pyproject.toml` `[dependency-groups] dev`:

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=8.0 | Testing framework |
| ruff | >=0.4 | Linting + formatting |
| pre-commit | >=3.6 | Git hooks |

Additional dev tools configured in `.pre-commit-config.yaml`:
- **pre-commit-hooks** v6.0.0 — trailing whitespace, end-of-file fixer, YAML check, debug statements
- **ruff-pre-commit** v0.11.2 — lint + format on commit
- **pytest unit tests** — local hook runs `uv run pytest tests/unit -v`

## Config Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, tool config (ruff, mypy, pytest, setuptools) |
| `.pre-commit-config.yaml` | Git hooks: ruff lint/format, pytest unit |
| `Dockerfile` | CUDA 12.4 + Python 3.11 + uv |
| `docker-compose.yml` | App + MLflow server containers |
| `.gitignore` | Python, uv, secrets, ML artifacts, notebooks, workspace |
| `AGENTS.md` | AI agent workflow guidelines |
| `CLAUDE.md` | Claude-specific instructions |
| `src/tabular_blueprint/py.typed` | PEP 561 marker |

## Ruff Configuration

From `pyproject.toml`:
- **Line length**: 100
- **Target**: Python 3.11
- **Rules**: E, F, I, UP, B, SIM, C4, PT, RUF
- **Fixable**: ALL
- **Per-file ignores**: `cli.py` exempts B008; `notebooks/*` exempts E402, I001

## MyPy Configuration

From `pyproject.toml`:
- **Python version**: 3.11
- **Ignore missing imports**: true
- **Disallow untyped defs**: true

## Pytest Configuration

From `pyproject.toml`:
- **Test paths**: `tests/`
- **Python path**: `src/`
- **Strict markers**: enabled
- **Markers**: slow, integration, unit, e2e, serial, network, smoke
- **Import mode**: importlib

## Build/Tooling

- **Package manager**: `uv` (installed via `astral.sh/uv` in Docker)
- **Build system**: setuptools >= 69 with wheel backend
- **Package layout**: src layout (`src/tabular_blueprint/`)
- **Entry point**: `tabblueprint` CLI command → `tabular_blueprint.cli:app`
- **Lock file**: `uv.lock` present
- **Execution**: Always `uv run <command>` per AGENTS.md
- **Sync**: `uv sync`
