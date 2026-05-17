# OVERVIEW

Last updated: 2026-05-15

## Project Identity

iter8ml is a CLI-first, single-node framework for high-velocity iteration on tabular ML problems. It automates the full experiment loop — data loading, preprocessing, feature engineering, model training/selection, HPO, drift detection, explainability, and export — with a Hamilton DAG pipeline and config-driven workflow.

## Architecture

**Pattern:** Library package with CLI + optional MCP server. No client-server split; everything runs in-process on a single node.

```
CLI (typer)  ──►  Trainer  ──►  PipelineExecutor (Hamilton DAG)
                      │                    │
                      ▼                    ▼
               ExperimentConfig       Node modules (prep, train, drift, features)
                      │                    │
                      ▼                    ▼
               Data layer (Polars)  ◄──►  Models (8 families)
                                          Engine (evaluator, HPO, calibration, tracker)
                                          Services (registry, reporting, export, LLM, MCP)
```

**Data flow:**
1. CLI or MCP builds `ExperimentConfig` (`src/iter8ml/config.py:134`)
2. `Trainer.run()` dispatches to `PipelineExecutor` (`src/iter8ml/engine/trainer.py:21`)
3. DAG nodes execute in dependency order — preprocessing → data prep → model selection → baselines → feature engineering → training → state generation
4. Results persisted to `workspace/experiments.jsonl` (JSONL tracker) and optionally W&B/MLflow
5. Champion model promoted in `workspace/registry.json` via file-locked `RegistryService`

**Key abstractions:**
- `ExperimentConfig` — Pydantic model; single source of truth for all pipeline behavior (`src/iter8ml/config.py:134`)
- `PipelineExecutor` — builds mode-specific Hamilton drivers from node modules (`src/iter8ml/engine/pipelines/executor.py`)
- `PipelineMode` — 5 modes: TRAINING, DRIFT, EXPORT, HPO, INFERENCE (`src/iter8ml/engine/pipelines/executor.py:35`)
- `Tracker` — protocol for experiment logging (JSONL, W&B, MLflow) (`src/iter8ml/engine/tracker.py:13`)
- `ModelSelector` — hardware-aware model routing (`src/iter8ml/engine/models/selector.py:6`)
- `Workspace` — dataclass pointing to workspace root, experiments.jsonl, registry.json (`src/iter8ml/workspace.py:18`)

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | >=3.11 |
| Package manager | uv + hatchling | hatchling>=1.25 |
| CLI framework | typer | >=0.12 |
| Dataframes | Polars | >=1.0 |
| Validation | Pydantic v2 | >=2.0 |
| ML (core) | scikit-learn, numpy | >=1.4, >=1.26 |
| GBDT models | CatBoost, LightGBM, XGBoost | >=1.2, >=4.0, >=2.0 |
| Deep learning | PyTorch, transformers, accelerate | >=2.3, >=4.40, >=0.30 |
| Tabular foundation | TabPFN | >=2.0 |
| Deep tabular | pytorch-tabular | >=1.0 |
| HPO | Optuna | >=3.6 |
| DAG orchestration | sf-hamilton | >=1.70 |
| Explainability | SHAP | >=0.44 |
| Data quality | cleanlab | >=2.6 |
| Experiment tracking | MLflow, W&B | >=3.11.1, >=0.17 |
| LLM integration | litellm | >=1.83.10 |
| MCP server | mcp (FastMCP) | >=0.9 |
| Testing | pytest, hypothesis | >=9.0.3, >=6.0 |
| Linting | ruff | >=0.4 |
| Type checking | mypy | >=1.10 |
| Docs | MkDocs Material, mkdocstrings, mike | >=9.5 |
| Terminal output | rich | >=13.0 |
| Config formats | YAML, TOML, JSON, .py | PyYAML>=6.0 |

## Infrastructure

| Component | Details |
|-----------|---------|
| Dockerfile | NVIDIA CUDA 12.4 runtime + Python 3.11 + uv (`Dockerfile:1`) |
| docker-compose | 2 services: `app` (GPU passthrough) + `mlflow` server (v2.13.0, port 5000) (`docker-compose.yml`) |
| CI | GitHub Actions: `ci.yml`, `docs.yml`, `benchmarks.yml` (`.github/workflows/`) |
| Publishing | hatch-vcs for versioning from git tags |
| Docs hosting | GitHub Pages via mike (versioned) |

## Integrations

| Integration | Purpose | Status | Entry point |
|-------------|---------|--------|-------------|
| TabPFN | Foundation model for small datasets | Optional (requires token) | `src/iter8ml/engine/models/tabpfn_model.py` |
| MLflow | Experiment tracking server | Optional | `src/iter8ml/engine/tracker.py` |
| Weights & Biases | Experiment tracking | Optional | `src/iter8ml/engine/tracker.py` |
| LiteLLM | LLM commentary on results | Optional | `src/iter8ml/services/llm.py` |
| MCP (FastMCP) | Agentic ML via Claude Desktop | Optional | `src/iter8ml/services/mcp.py` |
| Cleanlab | Label noise detection | Optional | `src/iter8ml/data/quality.py` |
| SHAP | Model explainability | Optional | `src/iter8ml/analysis/explainability.py` |
| Optuna | Hyperparameter optimization | Optional (base extra) | `src/iter8ml/engine/hpo.py` |
| Hamilton | DAG pipeline orchestration | Optional (base extra) | `src/iter8ml/engine/pipelines/executor.py` |
| SHAP + Optuna + Hamilton | Core ML pipeline | Installed via `--extra base` | — |
| PyTorch + transformers | Deep learning models | Optional (opinion extra) | `src/iter8ml/engine/models/deep/` |

## Auth Flow

No user auth system. The framework is a local CLI tool.

- **TabPFN:** License token via `TABPFN_TOKEN` env var or interactive browser login (`src/iter8ml/engine/models/tabpfn_model.py`)
- **MLflow:** Connects to local or remote tracking server; auth handled by MLflow's own config
- **W&B:** Uses `wandb.login()` / `WANDB_API_KEY` env var
- **LLM (litellm):** API key via env var matching the provider (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
- **MCP:** No auth; runs as local stdio server for Claude Desktop

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `TABPFN_TOKEN` | For TabPFN in CI/headless | JWT license token from priorlabs.ai |
| `ITER8ML_WORKSPACE` | No | Override workspace root directory (default: `workspace/`) |
| `ITER8ML_LLM_MODEL` | No | Override LLM model for commentary (default: `claude-sonnet-4-20250514`) |
| `TABBLUEPRINT_LLM_MODEL` | No | Legacy alias for `ITER8ML_LLM_MODEL` (`src/iter8ml/services/llm.py:25`) |
| `WANDB_API_KEY` | For W&B tracking | Weights & Biases authentication |
| `MLFLOW_TRACKING_URI` | For MLflow tracking | MLflow server URL |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | For LLM commentary | LLM provider API key (via litellm) |
| `OMP_NUM_THREADS` | No | Auto-configured by `HardwareProfile.configure_omp_threads()` (`src/iter8ml/config.py:350`) |
| `DOTENV_PUBLIC_KEY` | No | dotenvx public key for encrypted .env (`encrypted`) |
