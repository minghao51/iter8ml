# Tech Stack

## Languages and Versions

| Language | Version | Source |
|----------|---------|--------|
| Python | >=3.11 (target 3.11) | `pyproject.toml:5`, `pyproject.toml:44` |
| Python (dev env) | 3.13 | `.venv/` path structure |

## Package Manager

| Tool | Version | Source |
|------|---------|--------|
| uv | latest (installed via curl) | `pyproject.toml` (uv.lock present), `Dockerfile:9` |

- Lockfile: `uv.lock` at project root
- Virtual env: `.venv/` managed by uv

## Frameworks and Libraries

### Core Dependencies (`pyproject.toml:9-28`)

| Library | Version | Purpose | Used In |
|---------|---------|---------|---------|
| polars | >=1.0 | DataFrame engine (primary data structure) | `core/data/loaders.py`, `core/data/processors.py`, `core/data/adapter.py`, `core/monitoring/drift.py` |
| pydantic | >=2.0 | Data validation & config models | `configs/experiment.py`, `configs/hardware.py`, `core/monitoring/drift.py` |
| pydantic-settings | >=2.0 | Settings management | `pyproject.toml:12` |
| catboost | >=1.2 | Gradient boosting (CPU, native categorical support) | `core/models/conventional/catboost_model.py` |
| lightgbm | >=4.0 | Gradient boosting (fast training) | `core/models/conventional/lightgbm_model.py` |
| xgboost | >=2.0 | Gradient boosting (hist tree method) | `core/models/conventional/xgboost_model.py` |
| tabpfn | >=2.0 | Tabular foundation model (prior-data fitted network) | `core/models/tabular_foundation/tabpfn_model.py` |
| skrub | >=0.3 | Tabular data preprocessing | `pyproject.toml:17` |
| cleanlab | >=2.6 | Label noise detection / data quality | `core/data/quality.py` |
| optuna | >=3.6 | Hyperparameter optimization | `core/engine/hpo.py` |
| torch | >=2.3 | Deep learning backend | `core/models/deep/ft_transformer.py`, `core/data/adapter.py`, `configs/hardware.py` |
| accelerate | >=0.30 | HuggingFace distributed training | `core/models/deep/ft_transformer.py` |
| transformers | >=4.40 | HuggingFace model hub | `core/data/adapter.py` (HuggingFace Dataset conversion) |
| scikit-learn | >=1.4 | Metrics, CV splits, baseline LR | `core/engine/evaluator.py`, `core/data/quality.py` |
| numpy | >=1.26 | Array operations | Throughout core/ |
| typer | >=0.12 | CLI framework | `main.py` |
| psutil | >=5.9 | System/hardware introspection | `configs/hardware.py` |
| rich | >=13.0 | Terminal formatting | `pyproject.toml:27` |

### Optional Dependencies (`pyproject.toml:30-37`)

| Extra | Package | Version | Purpose | Used In |
|-------|---------|---------|---------|---------|
| hamilton | sf-hamilton | >=1.70 | Dataflow pipeline framework | `pyproject.toml:31` |
| llm | mcp | >=0.9 | Model Context Protocol server | `mcp_server/tools.py` |
| llm | anthropic | >=0.25 | Claude API for LLM integration | `pyproject.toml:32` |
| wandb | wandb | >=0.17 | Experiment tracking (cloud) | `core/engine/tracker.py` (WandbTracker) |
| mlflow | mlflow | >=2.13 | Experiment tracking (local/remote) | `core/engine/tracker.py` (MLflowTracker), `docker-compose.yml` |
| zenml | zenml | >=0.57 | ML pipeline orchestration | `pyproject.toml:35` |
| dev | pytest | >=8.0 | Unit/integration testing | `tests/` |
| dev | ruff | >=0.4 | Linting & formatting | `.pre-commit-config.yaml`, `pyproject.toml:42-50` |
| transformers | datasets | >=2.14 | HuggingFace datasets | `core/data/adapter.py` |

## Build/Dev Tools

| Tool | Config File | Purpose |
|------|-------------|---------|
| ruff | `pyproject.toml:42-50` | Linting (E, F, I, UP, B, SIM rules), formatting, target py311, line-length 100 |
| pytest | `pyproject.toml:53-54` | Testing, testpaths=tests, verbose + short traceback |
| pre-commit | `.pre-commit-config.yaml` | Hooks: ruff format, ruff check --fix, pytest unit tests |
| uv | `uv.lock` | Dependency resolution & lock |

## Runtime/Deployment

| Component | Detail | Source |
|-----------|--------|--------|
| Base image | nvidia/cuda:12.4.0-runtime-ubuntu22.04 | `Dockerfile:1` |
| Runtime Python | python3.11 + python3.11-venv | `Dockerfile:4-5` |
| GPU support | NVIDIA CUDA 12.4, all GPUs | `Dockerfile:1`, `docker-compose.yml:12-15` |
| CLI entry | `tabblueprint` -> `main:app` | `pyproject.toml:40` |
| Commands | init, run, leaderboard, registry, hardware, drift, state, hpo | `main.py` |
| MCP server | FastMCP with 8 tools | `mcp_server/tools.py` |
| Docker Compose | app service + mlflow service (port 5000) | `docker-compose.yml` |
| Volumes | workspace-data, mlflow-data | `docker-compose.yml:25-27` |

## Config Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, ruff/pytest config |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff format, ruff check, pytest) |
| `uv.lock` | Dependency lockfile |
| `.github/workflows/ci.yml` | GitHub Actions CI (lint + test on push/PR) |
| `.github/dependabot.yml` | Weekly dependency & GH Actions updates |
| `configs/experiment.py` | ExperimentConfig Pydantic model |
| `configs/hardware.py` | HardwareProfile auto-detection |
| `configs/model_configs.py` | Model-specific HPO search spaces |
