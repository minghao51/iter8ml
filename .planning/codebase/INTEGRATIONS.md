# Integrations

## Experiment Tracking

Three backends, selected via `config.tracker` (enum `TrackerType`):

### 1. JSONL (default) — `iter8ml.utils.jsonl`
- **File**: `workspace/experiments.jsonl`
- **Import path**: `iter8ml.utils.jsonl` → `load_events()`, `iter_events()`
- **Tracker class**: `iter8ml.engine.tracker.JSONLTracker`
- **Features**: log rotation (100 MB threshold, 5 backups), thread-safe via `threading.Lock`
- **Events**: metrics, params, artifacts, structured event dicts with `run_id` and `timestamp`

### 2. W&B — optional `[wandb]` extra
- **Import path**: `wandb` (lazy inside `iter8ml.engine.tracker.WandbTracker`)
- **Tracker class**: `iter8ml.engine.tracker.WandbTracker`
- **Capabilities**: `wandb.init()`, `wandb.log()`, `wandb.Artifact`, `run.config.update()`
- **Config**: `project="iter8ml"` (overridable via kwargs)

### 3. MLflow — optional `[mlflow]` extra
- **Import path**: `mlflow` (lazy inside `iter8ml.engine.tracker.MLflowTracker`)
- **Tracker class**: `iter8ml.engine.tracker.MLflowTracker`
- **Capabilities**: `mlflow.log_metrics()`, `mlflow.log_params()`, `mlflow.log_artifact()`, `mlflow.log_dict()`
- **Config**: configurable `tracking_uri` + `experiment_name`
- **docker-compose**: ships `ghcr.io/mlflow/mlflow:v2.13.0` sidecar on port 5000

## Model Registry

- **File**: `workspace/registry.json`
- **Service**: `iter8ml.services.registry_service.RegistryService`
- **Concurrency**: cross-process file locking via `filelock.FileLock` → `workspace/registry.lock`
- **Atomic writes**: temp file + `os.replace()` for crash safety
- **Schema per key**: `{model, run_id, score, metric_name, artifact_path, registered_at}`
- **Promotion**: `promote_run()` scans JSONL for best model_completed event, updates if score better

## Persistence (File-Based)

| Artifact | Path | Format | Purpose |
|---|---|---|---|
| Event log | `workspace/experiments.jsonl` | JSONL (JSON lines) | Experiment runs, metrics, params |
| Model registry | `workspace/registry.json` | JSON (dict) | Champion model tracking per task |
| Artifacts | `workspace/artifacts/` | Binary (joblib/onnx/cbm) | Serialized model files |
| Export packages | `workspace/exports/<key>/` | Directory | Portable prediction packages |
| Preprocessing cache | `.iter8/` | JSON + pickle | Skipped preprocessing steps |

## Data Ingestion

`iter8ml.data.loaders` — all return Polars DataFrames:

| Function | Source | Format |
|---|---|---|
| `load_csv()` | File | `.csv` |
| `load_parquet()` | File | `.parquet` |
| `load_sqlite()` | SQLite db | SQL `SELECT` queries (read-only, injection-secured) |
| `load_data()` | Auto-detect | `.csv` or `.parquet` |

## LLM Integration

- **Optional**: `[llm]` extra → `litellm>=1.40`, `mcp>=0.9`
- **Agent**: `iter8ml.llm.TabularAgent`
- **LLM proxy**: `litellm.completion()` — provider-agnostic (OpenAI, Anthropic, etc.)
- **Default model**: `claude-sonnet-4-20250514`
- **Capabilities**: SHAP explanation, performance commentary, feature summary
- **Auth**: API key via environment variable (configurable `api_key_env` + `api_base`)

## MCP Server (Model Context Protocol)

- **File**: `iter8ml.mcp.tools`
- **Framework**: `mcp.server.fastmcp.FastMCP` (from `[llm]` extra)
- **Server name**: `"iter8ml"`
- **Exposed tools** (10 total):
  - `get_experiment_state()` — current_state.md with leaderboard + resource status
  - `get_column_stats(data_path)` — Polars `describe()` output
  - `run_baseline(data_path, target_col, task)` — TabPFN/CatBoost quick run
  - `run_hpo(data_path, target_col, model, task, trials)` — Optuna HPO
  - `get_event_log(n)` — last N JSONL events
  - `registry_show()` — current registry.json content
  - `registry_promote(run_id, key)` — promote run to champion
  - `detect_drift(reference_path, new_path)` — distribution drift detection
  - `export_champion(key, target_col)` — export portable prediction package

## Model Export

- **Service**: `iter8ml.services.export_service.ExportService`
- **Output**: portable directory containing:
  - `model.artifact` — serialized model binary
  - `predictor.py` — standalone predictor class with `predict()` / `predict_proba()`
  - `metadata.json` — model name, task, score, allowlisted model classes
  - `pipelines/preprocessing.py` — copied Hamilton preprocessing nodes
- **Predictor template** embeds Polars, Hamilton, numpy, and uses allowlist security

## Drift Detection

| Method | Import path | Test |
|---|---|---|
| KS/Chi2 | `iter8ml.monitoring.drift.DriftDetector` | Per-column statistical tests |
| PSI | `iter8ml.monitoring.psi_drift.PSIDriftDetector` | Population Stability Index |
| Domain Classifier | `iter8ml.monitoring.domain_classifier.DomainClassifierDriftDetector` | Classifier AUC |
| Hamilton DAG | `iter8ml.pipelines.executor.PipelineExecutor` | Orchestrated drift pipeline |

## Orchestration

- **Hamilton DAG**: `sf-hamilton>=1.70` in `iter8ml.pipelines` — `Driver.Builder().with_modules(module).build()`
  - Nodes: preprocessing, feature engineering, state generation
  - Drift pipeline: PSI + domain classifier as Hamilton nodes

## No External Services (Not Present)

- No FastAPI / Flask web server or API routes
- No Supabase, Firebase, or cloud databases
- No AWS SDK (boto3), GCP, or Azure
- No webhook handlers or HTTP callbacks
- No auth providers (OAuth, JWT, API key auth)
- No Redis, PostgreSQL, MongoDB, or message queues
- No Docker Compose secrets management (no dotenvx)
- No CI secret injection (plain GitHub env from vars)
