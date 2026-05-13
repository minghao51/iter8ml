# Tabular Blueprint — Overview

## Architecture
**Pattern:** Single-node tabular ML framework, CLI-first, with Hamilton DAG orchestration and optional MCP server for LLM agents.

```
CLI (typer) ──▶ Trainer ──▶ PipelineExecutor ──▶ Hamilton DAG
                  │              │                     │
                  │              │              ┌──────┴──────┐
                  │              │              │ Node Modules │
                  │              │              │ preprocessing│
                  │              │              │ data_prep    │
                  │              │              │ model_train  │
                  │              │              │ state_gen    │
                  │              │              └─────────────┘
                  │              │
                  ▼              ▼
             JSONLTracker   TrackingHook
                  │
                  ▼
         workspace/experiments.jsonl
         workspace/registry.json
```

### Backend — Python (src layout)
Layered architecture:

| Layer | Location | Pattern |
|-------|----------|---------|
| CLI | `src/tabular_blueprint/cli.py` | Typer app with commands: `run`, `init`, `hpo`, `drift`, `export`, `leaderboard`, `registry`, `diff`, `state`, `hardware` |
| Config | `src/tabular_blueprint/config.py` | Pydantic `ExperimentConfig` + `HardwareProfile`, multi-format loading (YAML/TOML/JSON/PY) |
| Orchestration | `src/tabular_blueprint/engine/trainer.py` | `Trainer.run()` → builds `PipelineExecutor` → delegates to Hamilton DAG |
| DAG Pipeline | `src/tabular_blueprint/pipelines/executor.py` | `PipelineExecutor` wraps Hamilton `Builder` with mode-specific modules |
| DAG Nodes | `src/tabular_blueprint/pipelines/nodes/` | Function-based nodes (preprocessing, data_preparation, model_selection, baselines, feature_engineering, model_training, state_generation, drift_detection) |
| Data | `src/tabular_blueprint/data/` | Polars-based loaders, adapter, leakage detection, quality checks, feature engineering, embeddings |
| Models | `src/tabular_blueprint/models/` | Lazy-imported registry (`factory.py`), Protocol-based `AbstractModel`, GBDT/deep/baseline wrappers |
| Monitoring | `src/tabular_blueprint/monitoring/` | KS/Chi2 drift, PSI drift, domain classifier drift, SHAP explainability |
| Services | `src/tabular_blueprint/services/` | Registry (file-locked champion tracking), Report (leaderboard, metric directionality), Export (portable predictor packages) |
| HPO | `src/tabular_blueprint/engine/hpo.py` | Optuna-based optimization with warmstart from historical JSONL events |
| MCP | `src/tabular_blueprint/mcp/tools.py` | FastMCP server exposing tools for LLM agents |
| Benchmarks | `benchmarks/` | OpenML + synthetic benchmarks with sweep configs |

**Entry point**: `tabblueprint run --data <path> --target <col>` → `src/tabular_blueprint/cli.py:47` (`run` command)

## Key Data Flows

1. **Training**: `cli.py run` → `Trainer.run()` → `PipelineExecutor.run_training()` → Hamilton DAG (7 modules) → `TrainingState` → `RegistryService.update_if_better()` → `workspace/experiments.jsonl` + `registry.json`
2. **Drift Detection**: `cli.py drift` → `PipelineExecutor.run_drift()` → Hamilton DAG (preprocessing + drift_detection) → `DriftReport`
3. **HPO**: `cli.py hpo` → `setup_hpo_components()` → `optimize_model()` with Optuna + warmstart → best params/score
4. **Export**: `cli.py export` → `ExportService.export()` → packages model.artifact + preprocessing.py + predictor.py into portable directory

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Core runtime |
| Package mgr | `uv` + setuptools | Dependency management, build |
| CLI | Typer + Rich | User-facing commands |
| Config | Pydantic v2 | Validation, serialization, multi-format loading |
| Data | Polars | DataFrame operations, CSV/Parquet I/O |
| ML | scikit-learn | Evaluation metrics, cross-validation, baselines |
| GBDT Models | CatBoost, LightGBM, XGBoost | Gradient-boosted decision trees |
| Deep Models | PyTorch, FT-Transformer, TabNet | Neural tabular models |
| Foundation | TabPFN v2 | Foundation model for tabular data |
| DAG | sf-hamilton | Function-based DAG orchestration |
| HPO | Optuna | Hyperparameter optimization with warmstart |
| Tracking | JSONL (built-in), wandb, mlflow | Experiment tracking |
| Monitoring | SHAP, cleanlab | Explainability, data quality auditing |
| MCP | FastMCP | LLM agent tool server |
| LLM | LiteLLM | Multi-provider LLM commentary |
| Docs | MkDocs Material + mkdocstrings + Quarto + mike | Documentation site with versioned notebooks |
| Lint | Ruff | Formatting + linting |
| Types | mypy | Static type checking |
| Tests | pytest | Test runner with markers |
| Container | Docker + docker-compose | GPU-enabled runtime, MLflow server |

## Infrastructure

```bash
# Local dev
uv sync                          # Install all deps
uv run tabblueprint run --data data.csv --target label
uv run pytest                    # Run tests
uv run ruff check src/           # Lint
uv run mypy src/                 # Type check

# Docker (GPU + MLflow)
docker-compose up                # NVIDIA CUDA 12.4 + MLflow v2.13.0 on :5000

# Docs
uv run mkdocs build              # Build docs site
make docs                        # Render Quarto notebooks + build docs
```

## Integrations

| Service | SDK | Purpose | Status |
|---------|-----|---------|--------|
| MLflow | `mlflow>=2.13` | Experiment tracking server | Optional (`[tracking]` extra) |
| W&B | `wandb>=0.17` | Experiment tracking | Optional (`[tracking]` extra) |
| Hamilton | `sf-hamilton>=1.70` | DAG orchestration | Optional (`[base]` extra), graceful fallback |
| Optuna | `optuna>=3.6` | HPO with warmstart | Optional (`[base]` extra) |
| TabPFN | `tabpfn>=2.0` | Foundation tabular model | Optional (`[deep]` extra), requires `TABPFN_TOKEN` |
| SHAP | `shap>=0.44` | Model explainability | Optional (`[audit]` extra) |
| cleanlab | `cleanlab>=2.6` | Data quality auditing | Optional (`[audit]` extra) |
| FastMCP | `mcp>=0.9` | LLM agent tool server | Optional (`[agent]` extra) |
| LiteLLM | `litellm>=1.40` | Multi-provider LLM calls | Optional (`[agent]` extra) |
| PyTorch | `torch>=2.3` | Deep model backends | Optional (`[deep]` extra) |

## Environment Variables

| Variable | Context | Purpose |
|----------|---------|---------|
| `TABPFN_TOKEN` | `.env` / CI | TabPFN license token for headless environments |
| `TABBLUEPRINT_LLM_MODEL` | Config override | LLM model for commentary (default: `claude-sonnet-4-20250514`) |
| `OMP_NUM_THREADS` | Hardware | Auto-configured by `HardwareProfile.configure_omp_threads()` based on platform |
