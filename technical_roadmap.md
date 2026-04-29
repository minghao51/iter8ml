# Technical Roadmap: Modular Tabular ML Blueprint

> **Status:** Living document — v0.1 draft
> **Audience:** Personal toolkit (near-term) → Open-source template (long-term)
> **Philosophy:** Composable Lego bricks, not a monolith. Every module should be independently usable in a production microservice.

---

## Table of Contents

1. [Vision & Problem Statement](#1-vision--problem-statement)
2. [Design Principles](#2-design-principles)
3. [Tech Stack Rationale](#3-tech-stack-rationale)
4. [Repository Structure](#4-repository-structure)
5. [Core Abstractions](#5-core-abstractions)
6. [Experiment Lifecycle](#6-experiment-lifecycle)
7. [Phase 1 — Core ML Engine](#7-phase-1--core-ml-engine)
8. [Phase 1.5 — Observability & DX Polish](#8-phase-15--observability--dx-polish)
9. [Phase 2 — LLM & Agentic Layer](#9-phase-2--llm--agentic-layer)
10. [Phase 3 — MLOps & Production Readiness](#10-phase-3--mlops--production-readiness)
11. [Hardware-Aware Routing](#11-hardware-aware-routing)
12. [Configuration Strategy](#12-configuration-strategy)
13. [Testing & Validation Strategy](#13-testing--validation-strategy)
14. [Open-Source Readiness Checklist](#14-open-source-readiness-checklist)
15. [Dependency Manifest](#15-dependency-manifest)
16. [Architectural Decision Log](#16-architectural-decision-log)

---

## 1. Vision & Problem Statement

### The Gap

Existing tabular ML tooling falls into two unsatisfying categories:

| Category | Examples | Problem |
|---|---|---|
| **Monolithic AutoML** | PyCaret, H2O, TPOT | Cumbersome, stale, opaque. Abstraction prevents debugging. |
| **Raw Libraries** | scikit-learn + XGBoost | High friction. No coherent iteration structure. |
| **Modern but partial** | TabPFN, Skrub, Hamilton | Excellent individual pieces with no "glue." |

### What This Repo Is

A **high-velocity iteration framework** that:

- Provides a **30-second baseline** (TabPFN / CatBoost) for any tabular task
- Supports **progressive depth**: quick baseline → tuned GBDT → Transformer fine-tune
- Handles **mixed dataset sizes** automatically via hardware-aware model routing
- Is **Polars-native** throughout — no Pandas bottlenecks
- Is structured so every module can be **extracted into a production API** without refactoring
- Is **LLM-ready** (Phase 2): exposes atomic tools via MCP for agentic automation

### Non-Goals (v1)

- Not a deployment platform (no FastAPI/Triton serving layer in Phase 1)
- Not a distributed training framework (single-node GPU focus)
- Not a general NLP or vision toolkit (tabular + text-as-feature only)

---

## 2. Design Principles

### P1 — Functional over Class-heavy
Prefer pure functions with typed signatures over deep inheritance trees. Classes are reserved for stateful objects that genuinely need lifecycle management (`Trainer`, `DataAdapter`). This makes each module trivially extractable.

### P2 — Explicit over Magic
No hidden state. No silent fallbacks. If a model falls back from GPU to CPU, it logs it and says why. If TabPFN is skipped due to row count, it surfaces that decision in the experiment record.

### P3 — Polars as the Single Source of Truth
Data lives in `pl.DataFrame` or `pl.LazyFrame` until it reaches a model boundary. Conversion to NumPy/Tensor happens at the last possible moment inside `DataAdapter`. No Pandas in `core/`.

### P4 — Config is Code
All experiment parameters are `Pydantic` models, not YAML files or argparse dicts. This gives IDE completion, runtime validation, and diff-friendly versioning.

### P5 — Observability First
Every experiment run emits a structured JSONL event. The leaderboard is a derived view over these events, not a separate tracking database. This makes the Phase 2 LLM loop trivial — the agent reads the JSONL, not a GUI.

### P6 — Hardware-Aware by Default
The `ModelSelector` checks dataset size and available VRAM before routing. The user never needs to manually decide "is this too big for TabPFN?"

---

## 3. Tech Stack Rationale

### 3.1 Environment & Packaging

| Tool | Role | Rationale |
|---|---|---|
| **`uv`** | Env + package management | 10–100× faster than pip/conda. Lockfile-first. |
| **`ruff`** | Linting + formatting | Replaces black + flake8 + isort in a single binary. |
| **`pyproject.toml`** | Project manifest | Single source of truth for deps, tool config, scripts. |

### 3.2 Data Layer

| Tool | Role | Rationale |
|---|---|---|
| **Polars** | Core DataFrame engine | Lazy API, Rust-backed, zero-copy Arrow. No Pandas. |
| **Skrub** | Heterogeneous preprocessing | Handles mixed types (text, dates, categoricals) without hand-crafting pipelines. Bridges Polars → sklearn. |
| **Cleanlab** | Data quality audit | Cross-validation-based label noise detection. Runs before any model selection. |

### 3.3 Model Layer

| Tool | Role | Dataset Size |
|---|---|---|
| **TabPFN v2** | Zero-shot baseline (no training) | < 10k rows |
| **CatBoost** | Native categorical GBDT | 10k – 1M rows |
| **LightGBM** | Speed-optimised GBDT | 10k – 5M rows |
| **XGBoost** | Established baseline / GPU-ready | 10k – 1M rows |
| **FT-Transformer** | Tabular Transformer (PyTorch) | > 50k rows, GPU required |
| **DeBERTa-v3** | Text-as-feature encoder | Any size, text columns present |

> **Decision rule:** TabPFN is always run as the first baseline for any dataset < 10k rows. For larger datasets, CatBoost and LightGBM are co-run. FT-Transformer is opt-in, gated by VRAM check.

### 3.4 Training Infrastructure

| Tool | Role | Rationale |
|---|---|---|
| **PyTorch** | Deep model backend | Universal. Supports quantization, custom loss, CUDA. |
| **HuggingFace `accelerate`** | Device management | Handles CPU/GPU/multi-GPU dispatch without boilerplate. |
| **Optuna** | Hyperparameter optimisation | Backend-agnostic (works with CatBoost, LightGBM, PyTorch). Pruning support. |
| **Hamilton** | Pipeline DAG (optional) | Dataflow DAGs for reproducible feature engineering. Swappable — not a hard dependency. |

> **Orchestration note:** Hamilton is the default for structured pipelines but is kept behind an optional `[hamilton]` extras group. Plain functional scripts work fine for quick experiments.

### 3.5 Evaluation & Tracking

| Tool | Role |
|---|---|
| **`engine/evaluator.py`** | Cross-validation, stratified splits, custom metrics |
| **JSONL event log** | Structured experiment history (replaces MLflow for Phase 1) |
| **`workspace/leaderboard.md`** | Human-readable derived view over the event log |

---

## 4. Repository Structure

```
.
├── .venv/                        # Managed by uv (gitignored)
├── .devcontainer/
│   └── devcontainer.json         # Dev container (CUDA-ready)
├── Dockerfile                    # CUDA base image + uv
├── configs/
│   ├── __init__.py
│   ├── experiment.py             # ExperimentConfig (Pydantic)
│   ├── model_configs.py          # Per-model Pydantic configs
│   └── hardware.py               # HardwareProfile (auto-detected)
│
├── core/
│   ├── data/
│   │   ├── loaders.py            # Polars-based ingestion (CSV, Parquet, DB)
│   │   ├── processors.py         # Feature engineering (Polars expressions)
│   │   ├── adapter.py            # DataAdapter: Polars → NumPy / Tensor
│   │   └── quality.py            # Cleanlab wrapper (label noise audit)
│   │
│   ├── models/
│   │   ├── base.py               # AbstractModel protocol (fit/predict/save)
│   │   ├── selector.py           # ModelSelector (hardware + size aware routing)
│   │   ├── conventional/
│   │   │   ├── catboost_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── xgboost_model.py
│   │   ├── tabular_foundation/
│   │   │   └── tabpfn_model.py   # TabPFN v2 wrapper with row-count guardrail
│   │   └── deep/
│   │       ├── ft_transformer.py # FT-Transformer (PyTorch)
│   │       └── text_encoder.py   # DeBERTa-v3 / LLM embedding extractor
│   │
│   ├── engine/
│   │   ├── trainer.py            # Ties config + data + model into a run
│   │   ├── evaluator.py          # CV strategies, metrics registry
│   │   ├── hpo.py                # Optuna study factory
│   │   └── tracker.py            # Pluggable Tracker protocol (JSONL / W&B / MLflow)
│   │
│   └── monitoring/               # Phase 3
│       └── drift.py              # DriftDetector (KS test + Chi²)
│
├── pipelines/                    # Hamilton DAGs (optional, behind extras)
│   ├── feature_engineering.py
│   └── full_experiment.py
│
├── workspace/                    # Runtime state (gitignored except .gitkeep)
│   ├── experiments.jsonl         # Structured event log
│   ├── leaderboard.md            # Derived leaderboard (auto-generated)
│   ├── registry.json             # Model registry — best model per (dataset, task)
│   └── current_state.md          # LLM-readable run summary (Phase 2)
│
├── mcp_server/                   # Phase 2 only — LLM agent interface
│   ├── server.py
│   ├── tools.py
│   └── prompts.py
│
├── examples/
│   └── zenml_pipeline.py         # Phase 3 — ZenML step wrappers (advanced users)
│
├── notebooks/                    # Exploratory notebooks (not part of core)
├── tests/
│   ├── unit/
│   └── integration/
│
├── pyproject.toml
├── README.md
└── main.py                       # CLI entry point (typer subcommands)
```

---

## 5. Core Abstractions

### 5.1 `AbstractModel` Protocol

Every model wrapper — GBDT, TabPFN, Transformer — conforms to this protocol. No inheritance required; structural subtyping via `Protocol`.

```python
# core/models/base.py
from typing import Protocol
import polars as pl
import numpy as np

class AbstractModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    @property
    def model_name(self) -> str: ...
```

### 5.2 `DataAdapter`

The most critical class in the repo. Handles all cross-format conversions and is the only place where Polars frames are converted to other formats.

```python
# core/data/adapter.py

class DataAdapter:
    """
    Single point of truth for format conversion.
    Detects target format from the model type and converts accordingly.

    Supported outputs:
      - "numpy"   → (np.ndarray, np.ndarray) for GBDTs
      - "tensor"  → (torch.Tensor, torch.Tensor) for PyTorch models
      - "dataset" → HuggingFace Dataset for Transformers
    """
    def __init__(self, target_format: Literal["numpy", "tensor", "dataset"]): ...
    def transform(self, df: pl.DataFrame, target_col: str) -> tuple: ...
```

**Rule:** `DataAdapter` is the only file allowed to `import torch` or `import datasets` outside of `core/models/deep/`.

### 5.3 `ModelSelector`

Enforces hardware-aware and data-size-aware routing. Never lets a user accidentally run TabPFN on 500k rows.

```python
# core/models/selector.py

class ModelSelector:
    """
    Given a dataset profile and hardware profile, returns an ordered list
    of models to run, from fastest/cheapest to most expensive.
    """
    TABPFN_ROW_LIMIT = 10_000

    def select(
        self,
        n_rows: int,
        task: Literal["classification", "regression"],
        has_text_cols: bool,
        vram_gb: float,
    ) -> list[str]:
        ...
```

**Routing logic:**

```
n_rows < 10k       → [TabPFN, CatBoost, LightGBM]
10k ≤ n_rows < 500k → [CatBoost, LightGBM, XGBoost]
n_rows ≥ 500k      → [LightGBM, XGBoost]
vram_gb > 12       → append FT-Transformer (any size > 50k)
has_text_cols      → append TextEncoder (DeBERTa embeddings as features)
```

### 5.4 `ExperimentConfig` (Pydantic)

```python
# configs/experiment.py
from pydantic import BaseModel, Field
from typing import Literal

class ExperimentConfig(BaseModel):
    name: str
    task: Literal["classification", "regression"]
    target_col: str
    data_path: str
    cv_folds: int = 5
    cv_strategy: Literal["kfold", "stratified", "timeseries"] = "stratified"
    run_hpo: bool = False
    hpo_n_trials: int = 50
    models: list[str] | Literal["auto"] = "auto"
    random_seed: int = 42
    metrics: list[str] = Field(default_factory=lambda: ["roc_auc", "f1_macro"])
```

### 5.5 JSONL Event Schema

Every experiment event is appended to `workspace/experiments.jsonl`:

```json
{
  "event": "model_completed",
  "run_id": "exp_20260403_001",
  "model": "CatBoost",
  "task": "classification",
  "dataset": "v2_cleaned",
  "data_hash": "sha256:a3f1c9e2...",
  "n_rows": 45000,
  "n_features": 32,
  "cv_scores": {"roc_auc": 0.871, "f1_macro": 0.743},
  "params": {"depth": 6, "learning_rate": 0.05},
  "duration_seconds": 18.4,
  "artifact_path": "./workspace/artifacts/catboost_exp001.cbm",
  "hardware": {"device": "cuda", "vram_used_gb": 0.0},
  "timestamp": "2026-04-03T14:22:01Z"
}
```

### 5.6 Pluggable `Tracker` Protocol

The `Tracker` is a protocol wrapping all telemetry emission. The default is always `JSONLTracker` (zero external service). W&B or MLflow are opt-in via config without touching any trainer logic.

```python
# core/engine/tracker.py
from typing import Protocol

class Tracker(Protocol):
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def log_event(self, event: dict) -> None: ...   # writes to JSONL
    def finish(self) -> None: ...

class JSONLTracker:
    """Default. Writes structured events to workspace/experiments.jsonl."""
    ...

class WandbTracker:
    """Optional [wandb] extra. Mirrors all events to W&B run."""
    ...

class MLflowTracker:
    """Optional [mlflow] extra. Logs to a local or remote MLflow server."""
    ...
```

**Routing rule:** `ExperimentConfig.tracker` accepts `"jsonl"` (default), `"wandb"`, or `"mlflow"`. Multiple can be active simultaneously — the `TrainerRunner` fans out to all enabled trackers in parallel.

> **Key constraint:** `JSONLTracker` always runs, even when W&B is enabled. The JSONL file is the source of truth for the leaderboard and Phase 2 agent context. W&B is an additive mirror, never the primary store.

### 5.7 Data Hash Helper

Data lineage without DVC. Every loader call computes a deterministic SHA-256 hash of the DataFrame and stores it in the JSONL event under `data_hash`. This lets you detect silent dataset mutations between runs.

```python
# core/data/loaders.py
import hashlib
import polars as pl

def get_data_hash(df: pl.DataFrame) -> str:
    """
    Computes a deterministic SHA-256 hash of a Polars DataFrame.
    Uses hash_rows() to avoid materialising the full data as bytes.
    """
    row_hashes = df.hash_rows()
    combined = str(sorted(row_hashes.to_list())).encode()
    return "sha256:" + hashlib.sha256(combined).hexdigest()[:16]
```

### 5.8 Simple Model Registry

Tracks the best model artifact per `(dataset_name, task)` pair. Updated automatically by `trainer.py` after every evaluation if the new model beats the current champion on the primary metric.

```json
// workspace/registry.json
{
  "credit_risk:classification": {
    "model": "CatBoost",
    "run_id": "exp_20260403_001",
    "roc_auc": 0.891,
    "artifact_path": "./workspace/artifacts/catboost_exp001.cbm",
    "registered_at": "2026-04-03T16:44:00Z"
  }
}
```

The registry is the target for Phase 2 MCP tools like `registry promote` and the eventual Phase 3 serving layer.

---

## 6. Experiment Lifecycle

```
Raw Data (CSV / Parquet / DB)
         │
         ▼
  ┌─────────────────────┐
  │  1. Data Quality     │  ← Cleanlab: detect label noise, flag issues
  │     Audit            │     Output: quality_report.json
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  2. Preprocessing    │  ← Polars expressions + Skrub
  │     & Feature Eng.   │     Output: pl.DataFrame (versioned parquet)
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  3. Model Selection  │  ← ModelSelector reads dataset profile + HardwareProfile
  │                      │     Output: ordered model list
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  4. DataAdapter      │  ← Converts Polars → target format per model
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  5. Trainer Loop     │  ← Runs CV for each model in selection list
  │                      │     Emits JSONL event per model completion
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  6. Evaluator        │  ← Aggregates CV results, computes final metrics
  │                      │     Updates leaderboard.md
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  7. Registry Update  │  ← If new model beats champion → update registry.json
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  8. HPO (optional)   │  ← Optuna study on top-K models from leaderboard
  │                      │     Emits hpo_completed event to JSONL
  └─────────────────────┘
```

---

## 7. Phase 1 — Core ML Engine

### Milestone 1.0 — Skeleton & Tooling (Week 1)
- [ ] `pyproject.toml` with `uv` + all Phase 1 dependencies
- [ ] `ruff` + pre-commit config
- [ ] `ExperimentConfig` Pydantic model
- [ ] `HardwareProfile` auto-detection (VRAM, CPU cores, RAM)
- [ ] Empty module stubs with docstrings for all `core/` files
- [ ] `workspace/` directory with `.gitkeep` and gitignore rules

### Milestone 1.1 — Data Layer (Week 1–2)
- [ ] `loaders.py`: CSV, Parquet, and SQLite ingestion via Polars
- [ ] `get_data_hash()` helper in `loaders.py` — SHA-256 of DataFrame, stored in every JSONL event
- [ ] `quality.py`: Cleanlab integration — returns noise index + quality report
- [ ] `processors.py`: Polars-native null handling, type casting, date decomposition
- [ ] `adapter.py`: `DataAdapter` class with `numpy` and `tensor` output modes
- [ ] Unit tests for each loader + adapter round-trip

### Milestone 1.2 — Conventional Model Wrappers (Week 2–3)
- [ ] `AbstractModel` Protocol definition
- [ ] `CatBoostModel` wrapper (native categoricals, GPU flag)
- [ ] `LightGBMModel` wrapper (DART mode option)
- [ ] `XGBoostModel` wrapper (hist method, GPU flag)
- [ ] `ModelSelector` with routing logic + guardrails
- [ ] Integration test: CSV → adapter → CatBoost → metrics → JSONL event

### Milestone 1.3 — Foundation Model Baseline (Week 3)
- [ ] `TabPFNModel` wrapper with hard row-count guardrail (raises `DataSizeError` > 10k)
- [ ] Automatic fallback log when TabPFN is skipped
- [ ] Integration test: TabPFN baseline on synthetic classification dataset

### Milestone 1.4 — Trainer & Evaluator (Week 3–4)
- [ ] `evaluator.py`: CV strategies (KFold, StratifiedKFold, TimeSeriesSplit)
- [ ] Metrics registry: classification (ROC-AUC, F1, log-loss) + regression (RMSE, MAE, R²)
- [ ] `tracker.py`: `Tracker` protocol + `JSONLTracker` implementation
- [ ] `trainer.py`: full loop — config → selector → adapter → model → evaluator → tracker → JSONL
- [ ] `workspace/registry.json`: auto-updated after each run if new champion detected
- [ ] `leaderboard.md` auto-generation from JSONL
- [ ] `main.py` CLI skeleton with `typer`: `run`, `leaderboard`, `registry` subcommands

### Milestone 1.5 — Deep Model Layer (Week 4–5)
- [ ] `FT-Transformer` wrapper (PyTorch + HuggingFace `accelerate`)
- [ ] VRAM-gated entry in `ModelSelector`
- [ ] `text_encoder.py`: DeBERTa-v3 embedding extractor — returns embeddings as Polars columns
- [ ] `DataAdapter` extended with `dataset` output mode (HuggingFace `Dataset`)

### Milestone 1.6 — HPO Layer (Week 5–6)
- [ ] `hpo.py`: Optuna study factory — wraps any `AbstractModel` with a search space
- [ ] Per-model default search spaces defined in `configs/model_configs.py`
- [ ] HPO results emitted as `hpo_completed` JSONL events
- [ ] Pruning enabled by default (MedianPruner)

### Milestone 1.7 — Hamilton Pipelines (Week 6, optional extras)
- [ ] `pipelines/feature_engineering.py`: Hamilton DAG for reproducible feature transforms
- [ ] `pipelines/full_experiment.py`: end-to-end DAG from raw file to leaderboard
- [ ] Gated behind `[hamilton]` optional extras group in `pyproject.toml`
- [ ] Hamilton remains optional — all core functionality works without it

---

## 8. Phase 1.5 — Observability & DX Polish

> **Goal:** Make the repo feel polished for both personal use and eventual open-source handoff. No new ML capability — purely ergonomics, collaboration-readiness, and richer telemetry.

### Milestone 1.5.1 — CLI Subcommands (typer)

Replace the single-entrypoint `main.py` with a proper `typer` CLI. Every command maps 1:1 to an existing `core/` function — no new logic here.

```bash
# Environment
uv run tabblueprint init --data path/to/data.csv

# Experiment execution
uv run tabblueprint run --config configs/credit_risk.py
uv run tabblueprint run --config configs/credit_risk.py --models catboost lightgbm

# Results inspection
uv run tabblueprint leaderboard                        # prints leaderboard.md to terminal
uv run tabblueprint leaderboard --top 5 --metric roc_auc

# Registry management
uv run tabblueprint registry show
uv run tabblueprint registry promote --run-id exp_001 --to champion

# HPO
uv run tabblueprint hpo --config configs/credit_risk.py --model catboost --trials 100
```

- [ ] `typer` app with all subcommands above
- [ ] `--help` output is the primary "getting started" documentation
- [ ] Shell completion via `typer` (`--install-completion`)

### Milestone 1.5.2 — Pluggable Tracker (W&B / MLflow)

- [ ] `WandbTracker` implementation behind `[wandb]` extras
- [ ] `MLflowTracker` implementation behind `[mlflow]` extras
- [ ] `ExperimentConfig.tracker` field: `Literal["jsonl", "wandb", "mlflow"] = "jsonl"`
- [ ] Multi-tracker fan-out: JSONL always active; others additive
- [ ] W&B: log confusion matrix + feature importance plots as rich media artifacts
- [ ] MLflow: log to local tracking server (default `./mlruns`) or remote URI via env var
- [ ] Integration test: assert `WandbTracker` and `JSONLTracker` emit identical metric dicts

### Milestone 1.5.3 — Docker & Dev Container

```dockerfile
# Dockerfile (minimal, CUDA-ready)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 curl
RUN curl -Ls https://astral.sh/uv/install.sh | sh
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
```

```json
// .devcontainer/devcontainer.json
{
  "name": "tabular-blueprint",
  "image": "tabular-blueprint:latest",
  "features": { "ghcr.io/devcontainers/features/cuda:1": {} },
  "postCreateCommand": "uv sync"
}
```

- [ ] `Dockerfile` with CUDA 12.4 base + `uv` installed
- [ ] `.devcontainer/devcontainer.json` for VS Code / GitHub Codespaces
- [ ] `docker-compose.yml` with optional MLflow tracking server service
- [ ] README section: "Running in Docker"

---

## 9. Phase 2 — LLM & Agentic Layer

> **Prerequisite:** Phase 1 complete and stable. The JSONL event log is the primary interface the LLM reads.

### 9.1 State Observer

Generates an LLM-readable `workspace/current_state.md` after every trainer run:

```markdown
## Current Experiment State

**Task:** Binary Classification | **Dataset:** v2_cleaned (45,231 rows, 32 features)

### Leaderboard (sorted by ROC-AUC)
| Model | ROC-AUC | F1 | Duration |
|---|---|---|---|
| CatBoost (tuned) | 0.891 | 0.762 | 47s |
| LightGBM | 0.874 | 0.743 | 12s |
| TabPFN | 0.831 | 0.701 | 2s |

### Resource Status
VRAM: 4.2 / 16 GB used | Suggested next: FT-Transformer (VRAM available)

### Data Lineage
v2_cleaned ← handle_nulls(v1_raw) ← original.csv
Label noise: 2.3% flagged by Cleanlab (see quality_report.json)
```

### 9.2 MCP Server

Exposes atomic tools for LLM agents (Claude Desktop, custom agents):

| Tool | Description |
|---|---|
| `get_experiment_state` | Returns `current_state.md` content |
| `get_column_stats` | Returns Polars `.describe()` for a dataset |
| `run_baseline` | Triggers a TabPFN/CatBoost quick run |
| `suggest_features` | LLM-in-the-loop feature suggestion (see below) |
| `run_hpo` | Triggers Optuna study for a named model |
| `get_event_log` | Returns last N JSONL events |
| `registry_show` | Returns current `registry.json` content |
| `registry_promote` | Promotes a run_id to champion in the registry |

### 9.3 LLM-in-the-Loop Feature Engineering

The agentic feature engineering feedback loop:

```
1. LLM reads current_state.md + column stats
2. LLM suggests: "Create income_to_debt_ratio = income / debt"
3. MCP tool generates Polars .with_columns() snippet
4. Sandbox executor runs snippet on a 1k-row sample
5. Validator checks: null rate > 20%? Zero variance? Division by zero?
6. If OK → applies to full dataset, emits feature_added JSONL event
7. If fails → sends traceback back to LLM context → loop continues
```

### 9.4 Phase 2 Milestones

- [ ] `StateObserver` class: generates `current_state.md` post-run
- [ ] `mcp_server/server.py`: MCP server with basic tools (`get_state`, `get_stats`)
- [ ] `mcp_server/tools.py`: all 8 tools implemented (including registry tools)
- [ ] `mcp_server/prompts.py`: system prompt for the data science agent persona
- [ ] Sandbox executor with validation step
- [ ] Integration test: full agentic loop (suggest → generate → validate → apply)

---

## 10. Phase 3 — MLOps & Production Readiness

> **Goal:** Add the monitoring, orchestration compatibility, and production-serving concerns that become relevant once models are being used beyond a single experiment session.

### 10.1 Drift Detection (`core/monitoring/drift.py`)

Once a champion model is registered, you need to know if incoming data differs from the training distribution. This module is standalone — it doesn't touch the trainer or evaluator.

```python
# core/monitoring/drift.py
class DriftDetector:
    """
    Compares a reference DataFrame (training) against a new DataFrame (production).

    Numeric columns : Kolmogorov-Smirnov test
    Categorical cols : Chi-squared test

    Returns a DriftReport with per-column p-values and a global drift flag.
    """
    def __init__(self, reference_df: pl.DataFrame, alpha: float = 0.05): ...
    def detect(self, new_df: pl.DataFrame) -> DriftReport: ...
```

**Integration points:**
- Callable as a standalone CLI: `tabblueprint drift --reference train.parquet --new batch.parquet`
- Exposed as an MCP tool in Phase 2 server (`detect_drift`)
- `DriftReport` emitted as a `drift_checked` JSONL event for history tracking

**Milestone:**
- [ ] `DriftDetector` class with KS + Chi² tests
- [ ] `DriftReport` Pydantic model (per-column results + global flag)
- [ ] `drift` CLI subcommand
- [ ] Unit tests with synthetic distribution shift

### 10.2 ZenML Compatibility (`examples/zenml_pipeline.py`)

The repo's `core/` functions are already plain Python — they can be wrapped as ZenML steps with zero refactoring of the underlying logic. This example is purely for advanced users who need scheduled retraining or remote execution.

```python
# examples/zenml_pipeline.py
from zenml import step, pipeline
from core.data.loaders import load_parquet
from core.engine.trainer import run_experiment

@step
def load_data(path: str) -> pl.DataFrame:
    return load_parquet(path)

@step
def train_models(df: pl.DataFrame, config: ExperimentConfig) -> dict:
    return run_experiment(df, config)

@pipeline
def retraining_pipeline(data_path: str):
    df = load_data(data_path)
    train_models(df, config=ExperimentConfig(...))
```

**Milestone:**
- [ ] `examples/zenml_pipeline.py` with `load → train → evaluate` steps
- [ ] README note: "ZenML is not a dependency. This example shows how `core/` functions drop into ZenML with no changes."
- [ ] ADR-013 documenting the decision to keep ZenML out of core

### 10.3 LightEx Migration Path (ADR only)

LightEx (Amazon's lightweight experiment framework) provides battle-tested dataset snapshotting and run comparison queries that overlap significantly with the custom JSONL + leaderboard logic built in Phase 1. It is **not** included in Phase 1 or Phase 3, but is documented here as a potential migration path if the custom JSONL query logic becomes burdensome.

**Trigger condition for migration:** If maintaining `leaderboard.md` generation and run comparison requires >2 days of work to extend, evaluate LightEx as a drop-in replacement for `workspace/` logic.

### 10.4 Phase 3 Milestones

- [ ] `core/monitoring/drift.py` — `DriftDetector` + `DriftReport`
- [ ] `drift` CLI subcommand
- [ ] `examples/zenml_pipeline.py`
- [ ] ADR-013: ZenML kept out of core
- [ ] ADR-014: LightEx migration trigger condition documented

---

## 11. Hardware-Aware Routing

The `HardwareProfile` is auto-detected at runtime and informs all routing decisions:

```python
# configs/hardware.py
import torch
import psutil

class HardwareProfile(BaseModel):
    vram_gb: float          # 0.0 if no GPU
    system_ram_gb: float
    cpu_cores: int
    has_gpu: bool
    gpu_name: str | None

    @classmethod
    def detect(cls) -> "HardwareProfile":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
        return cls(
            vram_gb=round(vram, 1),
            system_ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
            cpu_cores=psutil.cpu_count(logical=False),
            has_gpu=torch.cuda.is_available(),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
```

**Routing matrix:**

| Condition | Action |
|---|---|
| `n_rows > 10k` | Skip TabPFN, log reason |
| `vram_gb < 8` | Skip FT-Transformer, log reason |
| `vram_gb >= 12` | Enable FT-Transformer + larger batch sizes |
| `has_text_cols and vram_gb >= 8` | Enable DeBERTa embedding extraction |
| `system_ram_gb < 16 and n_rows > 1M` | Force LightGBM (memory-safe) over XGBoost |

---

## 12. Configuration Strategy

### pyproject.toml (all phases)

```toml
[project]
name = "tabular-blueprint"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "polars>=1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "catboost>=1.2",
    "lightgbm>=4.0",
    "xgboost>=2.0",
    "tabpfn>=2.0",
    "skrub>=0.3",
    "cleanlab>=2.6",
    "optuna>=3.6",
    "torch>=2.3",
    "accelerate>=0.30",
    "transformers>=4.40",
    "scikit-learn>=1.4",
    "numpy>=1.26",
    "typer>=0.12",       # CLI subcommands
    "ruff>=0.4",
    "psutil>=5.9",       # HardwareProfile detection
]

[project.optional-dependencies]
hamilton = ["sf-hamilton>=1.70", "pyo3-polars>=0.14"]
llm     = ["mcp>=0.9", "anthropic>=0.25"]
wandb   = ["wandb>=0.17"]
mlflow  = ["mlflow>=2.13"]
zenml   = ["zenml>=0.57"]   # examples/ only, never imported in core/

[project.scripts]
tabblueprint = "main:app"   # exposes `uv run tabblueprint` CLI

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "-v --tb=short"
```

### Experiment Config Example

```python
# configs/my_experiment.py
from configs.experiment import ExperimentConfig

config = ExperimentConfig(
    name="credit_risk_v3",
    task="classification",
    target_col="default",
    data_path="data/credit_v3.parquet",
    cv_folds=5,
    cv_strategy="stratified",
    run_hpo=True,
    hpo_n_trials=100,
    models="auto",              # ModelSelector decides
    tracker="jsonl",            # "jsonl" | "wandb" | "mlflow"
    metrics=["roc_auc", "f1_macro", "log_loss"],
    run_quality_audit=True,     # set False to skip Cleanlab on large datasets
)
```

---

## 13. Testing & Validation Strategy

### Unit Tests (`tests/unit/`)
- `DataAdapter` round-trip: Polars → NumPy → back, assert no precision loss
- `ModelSelector` routing: assert correct model list for each size/hardware combo
- `Cleanlab` wrapper: synthetic noisy labels, assert flagged rows match expected
- `ExperimentConfig` validation: assert Pydantic catches invalid field combos
- `get_data_hash()`: assert identical DataFrames produce identical hashes; assert mutated frame produces different hash
- `DriftDetector`: synthetic distribution shift, assert `drift_detected=True` on shifted columns
- `Tracker` fan-out: assert both `JSONLTracker` and `WandbTracker` receive identical metric dicts in multi-tracker mode

### Integration Tests (`tests/integration/`)
- Full pipeline on `sklearn.datasets.make_classification` (1k rows, 20 features)
- Assert JSONL event is written with correct schema (including `data_hash`) after each model run
- Assert `leaderboard.md` updates after run
- Assert `registry.json` updates when a new champion is detected
- TabPFN guardrail: assert `DataSizeError` raised on 15k-row input
- CLI: `tabblueprint leaderboard` exits 0 and prints expected table headers

### Synthetic Benchmark
Use `sklearn.datasets` to create a standard benchmark suite run on every release tag:
- Binary classification (10k rows, mixed types)
- Multiclass classification (50k rows)
- Regression (100k rows)

This ensures every release can be validated against known baselines and catches regressions in model wrapper behaviour.

---

## 14. Open-Source Readiness Checklist

To be completed before public release:

**Documentation**
- [ ] `README.md` with 60-second quickstart (`uv sync && uv run tabblueprint run --config ...`)
- [ ] `CONTRIBUTING.md` with PR guidelines and code style rules
- [ ] `LICENSE` (MIT recommended)
- [ ] `CHANGELOG.md` initialized
- [ ] All Pydantic models have docstrings and field descriptions
- [ ] All public functions have type hints and docstrings
- [ ] Example configs for 3 real-world dataset types (classification, regression, text)
- [ ] README section: "Running in Docker"
- [ ] README section: "Optional integrations (W&B, MLflow, Hamilton, ZenML)"

**Code Quality**
- [ ] Remove all hardcoded local paths (use `pathlib.Path` + config)
- [ ] `workspace/` fully gitignored (only `.gitkeep` committed)
- [ ] `ruff` passes with zero warnings on full repo
- [ ] No `import pandas` anywhere in `core/` (enforced via `ruff` custom rule)
- [ ] No `import torch` outside `core/models/deep/` and `core/data/adapter.py`

**CI/CD**
- [ ] GitHub Actions: `ruff check` + `uv run pytest tests/unit/` on every PR
- [ ] GitHub Actions: synthetic benchmark suite on every release tag
- [ ] Dependabot config for weekly dep updates
- [ ] Pre-commit hooks: `ruff format`, `ruff check`, `pytest tests/unit/`

**Environment**
- [ ] `Dockerfile` with CUDA 12.4 base + `uv` installed
- [ ] `.devcontainer/devcontainer.json` for VS Code / Codespaces
- [ ] `docker-compose.yml` with optional MLflow tracking server service

---

## 15. Dependency Manifest

| Package | Version Floor | Phase | Optional | Extras Group |
|---|---|---|---|---|
| `polars` | ≥ 1.0 | 1 | No | — |
| `pydantic` | ≥ 2.0 | 1 | No | — |
| `pydantic-settings` | ≥ 2.0 | 1 | No | — |
| `catboost` | ≥ 1.2 | 1 | No | — |
| `lightgbm` | ≥ 4.0 | 1 | No | — |
| `xgboost` | ≥ 2.0 | 1 | No | — |
| `tabpfn` | ≥ 2.0 | 1 | No | — |
| `skrub` | ≥ 0.3 | 1 | No | — |
| `cleanlab` | ≥ 2.6 | 1 | No | — |
| `optuna` | ≥ 3.6 | 1 | No | — |
| `torch` | ≥ 2.3 | 1 | No | — |
| `accelerate` | ≥ 0.30 | 1 | No | — |
| `transformers` | ≥ 4.40 | 1 | No | — |
| `scikit-learn` | ≥ 1.4 | 1 | No | — |
| `numpy` | ≥ 1.26 | 1 | No | — |
| `typer` | ≥ 0.12 | 1.5 | No | — |
| `psutil` | ≥ 5.9 | 1 | No | — |
| `sf-hamilton` | ≥ 1.70 | 1 | Yes | `[hamilton]` |
| `pyo3-polars` | ≥ 0.14 | 1 | Yes | `[hamilton]` |
| `wandb` | ≥ 0.17 | 1.5 | Yes | `[wandb]` |
| `mlflow` | ≥ 2.13 | 1.5 | Yes | `[mlflow]` |
| `mcp` | ≥ 0.9 | 2 | Yes | `[llm]` |
| `anthropic` | ≥ 0.25 | 2 | Yes | `[llm]` |
| `zenml` | ≥ 0.57 | 3 | Yes | `[zenml]` |

---

## 16. Architectural Decision Log

| ID | Decision | Rationale | Alternatives Rejected |
|---|---|---|---|
| ADR-001 | Polars as sole DataFrame engine | Speed, lazy API, Arrow-native. No Pandas in `core/`. | Pandas (slow), DuckDB (less ergonomic for feature eng.) |
| ADR-002 | `AbstractModel` as Protocol, not ABC | Structural subtyping — no inheritance tax. Third-party models can conform without wrapping. | `abc.ABC` (requires inheritance), duck typing (no IDE support) |
| ADR-003 | JSONL as primary event store | Zero infra dependency. LLM-readable in Phase 2. Queryable via Polars directly. W&B/MLflow are additive mirrors. | MLflow as primary (heavy), W&B as primary (requires account) |
| ADR-004 | Optuna for HPO | Backend-agnostic, supports pruning, integrates with every model in stack. | Ray Tune (overkill for single-node), Hyperopt (less maintained) |
| ADR-005 | Hamilton optional, not required | Core ML loop must work without DAG overhead. Hamilton adds value for reproducibility, not iteration speed. | Making Hamilton mandatory (too much friction for quick runs) |
| ADR-006 | Pydantic configs, not YAML | IDE completion, runtime validation, `ruff`-friendly, diffable. | Hydra (complex), YAML (no validation), argparse (no structure) |
| ADR-007 | LLM layer in Phase 2 | ML engine must be rock-solid before adding agentic complexity. | Building both simultaneously (too many moving parts) |
| ADR-008 | TabPFN row-count hard guardrail | Silent degradation on large datasets is worse than a clear error. | Soft warning (user ignores it) |
| ADR-009 | Pluggable `Tracker` protocol over baking in W&B | JSONL always works with zero config. Teams can opt into W&B without any trainer refactor. | Baking W&B directly into trainer (breaks offline use), no tracking at all |
| ADR-010 | Data hash in JSONL, not DVC | A SHA-256 of `hash_rows()` gives lineage for free with no infra. DVC is warranted only when datasets need to be stored outside git at scale. | DVC (overkill for single-user), no lineage (silent mutations) |
| ADR-011 | Simple `registry.json` over MLflow Model Registry | Zero-dependency champion tracking is sufficient for Phase 1–2. MLflow registry adds a server requirement for a problem that doesn't need one yet. | MLflow Model Registry (needs server), no registry (can't promote models) |
| ADR-012 | `typer` CLI over argparse / click | typer generates `--help` docs from type hints automatically, making it self-documenting. Shell completion built-in. | argparse (verbose, no types), click (more boilerplate), bare `sys.argv` |
| ADR-013 | ZenML kept out of `core/`, example only | Plain functions in `core/` are already ZenML-compatible as steps. Adding ZenML as a core dep would violate P1 (Functional over Class-heavy) and add significant import overhead for users who don't need it. | ZenML as core dep (too heavy), no ZenML compat at all |
| ADR-014 | LightEx deferred — migration trigger documented | LightEx overlaps with JSONL + leaderboard logic. Migration is worth evaluating only if extending the query engine exceeds 2 days of effort. | Adopting LightEx in Phase 1 (unnecessary dep), ignoring it entirely |
| ADR-015 | `Cleanlab` audit skippable via config flag | On large datasets (>500k rows), Cleanlab's cross-validation step can add minutes. `run_quality_audit: bool = True` keeps it default-on but lets users bypass it explicitly. | Always mandatory (breaks large-dataset iteration speed) |

---

*Last updated: 2026-04-04 | Next review: after Phase 1 Milestone 1.4 completion*
