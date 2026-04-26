# Integrations

> Last updated: 2026-04-23

## External APIs

### LLM APIs (via LiteLLM)

- **LiteLLM** (`litellm>=1.40`, optional `llm` extra) — unified LLM API client
  - Configured in `src/tabular_blueprint/llm/__init__.py`
  - Default model: `claude-sonnet-4-20250514`
  - Supports any LiteLLM-compatible provider (OpenAI, Anthropic, etc.)
  - API key via environment variable (`LLMAgentConfig.api_key_env`)
  - Optional custom base URL (`LLMAgentConfig.api_base`)
  - Used for: SHAP explanation, performance commentary, feature summaries
  - Graceful degradation when disabled or unavailable

### Model Context Protocol (MCP)

- **MCP SDK** (`mcp>=0.9`, optional `llm` extra) — exposes tools for LLM agent integration
  - Server: `src/tabular_blueprint/mcp/tools.py` using `FastMCP`
  - Exposes 8 tools: `get_experiment_state`, `get_column_stats`, `run_baseline`, `run_hpo`, `get_event_log`, `registry_show`, `registry_promote`, `detect_drift`, `export_champion`
  - Allows external LLM agents to drive the ML workflow programmatically

## Databases

### SQLite

- Used for data ingestion via `src/tabular_blueprint/data/loaders.py::load_sqlite`
- Security-validated: only SELECT queries, no multi-statement
- Connects via `sqlite3` + `pl.read_database()`
- Used for loading training data from `.db` files

### File-Based Storage

- **JSONL** (`workspace/experiments.jsonl`) — primary experiment log
  - Rotation: max 100MB, 5 backups (`src/tabular_blueprint/engine/tracker.py`)
  - Thread-safe with locking
- **JSON** (`workspace/registry.json`) — model champion registry
  - File-lock protected via `fcntl` (`src/tabular_blueprint/services/registry_service.py`)
- **JSON** (`workspace/registry.lock`) — lock file for concurrent registry access

## Experiment Tracking

### W&B (Weights & Biases)

- **wandb** (`wandb>=0.17`, optional `wandb` extra)
  - Tracker: `src/tabular_blueprint/engine/tracker.py::WandbTracker`
  - Mirrors all events to W&B runs
  - Logs metrics, params, artifacts
  - Default project: `tabular-blueprint`
  - Selected via `ExperimentConfig.tracker = TrackerType.WANDB`

### MLflow

- **mlflow** (`mlflow>=2.13`, optional `mlflow` extra)
  - Tracker: `src/tabular_blueprint/engine/tracker.py::MLflowTracker`
  - Logs metrics, params, artifacts, dicts
  - Default experiment: `tabular-blueprint`
  - Docker service: `ghcr.io/mlflow/mlflow:v2.13.0` on port 5000
  - Backend store: `/mlruns` volume
  - Selected via `ExperimentConfig.tracker = TrackerType.MLFLOW`

### ZenML

- **zenml** (`zenml>=0.57`, optional `zenml` extra) — listed but no source integration found yet

## Cloud Services

### NVIDIA GPU (CUDA)

- **Docker**: `nvidia/cuda:12.4.0-runtime-ubuntu22.04` base image
- **GPU passthrough**: configured in `docker-compose.yml` with NVIDIA device driver
- **Detection**: `src/tabular_blueprint/config.py::HardwareProfile.detect()` via `torch.cuda`
- **Models requiring GPU**: TabPFN (enforces CUDA), FT-Transformer (uses Accelerate)
- **Auto-fallback**: ModelSelector routes to CPU models when GPU unavailable or low VRAM

## Data Sources

### File Formats

- **CSV** — `src/tabular_blueprint/data/loaders.py::load_csv` via `pl.read_csv`
- **Parquet** — `src/tabular_blueprint/data/loaders.py::load_parquet` via `pl.read_parquet`
- **SQLite** — `src/tabular_blueprint/data/loaders.py::load_sqlite`

### Data Adapters

- **NumPy** — GBDT models (default)
- **PyTorch Tensor** — deep learning models via Polars native `to_torch()`
- **HuggingFace Dataset** — transformer models via `datasets` library

## Model Persistence

### Artifact Storage

- **Filesystem** — `workspace/artifacts/` directory
- **CatBoost**: native `.cbm` format via `save_model/load_model`
- **LightGBM/XGBoost**: native binary formats
- **PyTorch** (FT-Transformer): state dict via `accelerator.save` or `torch.save`
- **TabPFN**: pickle serialization
- **Export**: `src/tabular_blueprint/services/export_service.py` packages models as portable directories with preprocessing pipeline + predictor script

## Monitoring & Drift Detection

### Statistical Tests (built-in, no external service)

- **KS Test** — numeric column drift (`src/tabular_blueprint/monitoring/drift.py` via `scipy.stats.ks_2samp`)
- **Chi-squared Test** — categorical column drift (`src/tabular_blueprint/monitoring/drift.py` via `scipy.stats.chi2_contingency`)

### PSI Drift

- **PSI (Population Stability Index)** — `src/tabular_blueprint/monitoring/psi_drift.py`
- Thresholds: <0.1 negligible, 0.1–0.25 moderate, >0.25 severe

### Domain Classifier Drift

- **Domain Classifier** — `src/tabular_blueprint/monitoring/domain_classifier.py`
- Trains a classifier to distinguish reference vs. new data
- Uses AUC score to detect drift (threshold default: 0.7)

## Explainability

### SHAP

- **SHAP** library (`shap>=0.44`, optional `shap` extra)
  - `src/tabular_blueprint/monitoring/explainability.py`
  - TreeExplainer for GBDT models
  - KernelExplainer for other models
  - Generates beeswarm and dependence plots (matplotlib, saved as PNG)

## Data Quality

### Cleanlab

- **Cleanlab** (`cleanlab>=2.6`)
  - `src/tabular_blueprint/data/quality.py`
  - Label noise detection via `find_label_issues` and `get_label_quality_scores`
  - Auto-cleaning with configurable threshold

## Webhooks

No incoming or outgoing webhooks. The system is CLI-driven and file-based.

## Email/Notifications

No email or notification integrations. All output is via CLI (Typer/Rich) or file-based reports.

## CDN/Storage

No CDN integration. All storage is local filesystem (`workspace/` directory).
