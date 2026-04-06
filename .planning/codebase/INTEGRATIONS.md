# Integrations

## Databases/Data Stores

| Store | Usage | Source |
|-------|-------|--------|
| SQLite | Data ingestion via `load_sqlite()` | `core/data/loaders.py:30-37` |
| JSONL (file-based) | Experiment event logging (`workspace/experiments.jsonl`) | `core/engine/tracker.py:17-42`, `main.py:92` |
| JSON (file-based) | Model registry (`workspace/registry.json`) | `core/engine/trainer.py:50-83` |
| Parquet files | Data loading & persistence | `core/data/loaders.py:25-27` |
| CSV files | Data loading | `core/data/loaders.py:9-22` |
| MLflow (local) | Experiment tracking server via Docker | `docker-compose.yml:17-23` |

## External APIs/SDKs

| Service | SDK/Package | Purpose | Source |
|---------|-------------|---------|--------|
| Weights & Biases | wandb>=0.17 (optional) | Cloud experiment tracking, artifact logging | `core/engine/tracker.py:45-72` |
| Anthropic Claude | anthropic>=0.25 (optional) | LLM integration for MCP agents | `pyproject.toml:32` |
| HuggingFace Hub | transformers>=4.40, datasets>=2.14 | Model hub access, dataset format conversion | `core/data/adapter.py:52-66` |
| MLflow | mlflow>=2.13 (optional) | Local/remote experiment tracking | `core/engine/tracker.py:75-106`, `docker-compose.yml:17-23` |

## ML Frameworks/Libraries

### Gradient Boosting (Conventional)
| Library | Wrapper File | Features |
|---------|--------------|----------|
| CatBoost | `core/models/conventional/catboost_model.py` | Native categorical support, classifier & regressor |
| LightGBM | `core/models/conventional/lightgbm_model.py` | Fast training, binary/regression objectives |
| XGBoost | `core/models/conventional/xgboost_model.py` | Hist tree method, DMatrix format |

### Deep Learning
| Library | Wrapper File | Features |
|---------|--------------|----------|
| PyTorch | `core/models/deep/ft_transformer.py` | FT-Transformer architecture, AdamW optimizer |
| HuggingFace Accelerate | `core/models/deep/ft_transformer.py:56-74` | Multi-GPU/distributed training support |

### Foundation Models
| Library | Wrapper File | Features |
|---------|--------------|----------|
| TabPFN v2 | `core/models/tabular_foundation/tabpfn_model.py` | Prior-data fitted network, 10K row limit guardrail |

### Data Quality & Preprocessing
| Library | Usage File | Features |
|---------|------------|----------|
| Cleanlab | `core/data/quality.py` | Label noise detection, quality scoring |
| Skrub | `pyproject.toml:17` | Tabular data preprocessing (declared, not yet imported in source) |

### Hyperparameter Optimization
| Library | Usage File | Features |
|---------|------------|----------|
| Optuna | `core/engine/hpo.py` | Median/Hyperband pruners, study-based optimization |

### Evaluation
| Library | Usage File | Features |
|---------|------------|----------|
| scikit-learn | `core/engine/evaluator.py` | KFold, StratifiedKFold, TimeSeriesSplit, metrics (roc_auc, f1, accuracy, log_loss, rmse, mae, r2) |
| scipy | `core/monitoring/drift.py` | KS test, Chi-squared test for drift detection |

## Auth Providers

| Provider | Usage | Source |
|----------|-------|--------|
| None | No authentication implemented | N/A |

Note: W&B and MLflow integrations would use their respective auth mechanisms (API keys) when enabled, but no auth is built into the project itself.

## CI/CD Services

| Service | Config | Triggers | Jobs |
|---------|--------|----------|------|
| GitHub Actions | `.github/workflows/ci.yml` | push to main, PR to main | lint (ruff check + format), test (pytest unit) |
| Dependabot | `.github/dependabot.yml` | Weekly | pip dependency updates (limit 5), github-actions updates |

## Webhooks/Integrations

| Integration | Type | Purpose | Source |
|-------------|------|---------|--------|
| MCP Server | Model Context Protocol | LLM agent tools (8 tools: get_experiment_state, get_column_stats, run_baseline, run_hpo, get_event_log, registry_show, registry_promote, detect_drift) | `mcp_server/tools.py` |
| CLI (Typer) | Command-line interface | 8 commands: init, run, leaderboard, registry, hardware, drift, state, hpo | `main.py` |
| JSONL Event Log | File-based webhook alternative | Structured event streaming for experiment lifecycle | `core/engine/tracker.py`, `core/engine/state_observer.py` |

## Hardware/Runtime Integrations

| Component | Integration | Source |
|-----------|-------------|--------|
| NVIDIA GPU | CUDA 12.4 runtime, auto-detection via torch.cuda | `Dockerfile:1`, `configs/hardware.py:17-26` |
| Apple Silicon | OMP_NUM_THREADS=1 optimization | `core/engine/trainer.py:15-16` |
| System introspection | psutil for RAM/CPU detection | `configs/hardware.py:33-35` |
