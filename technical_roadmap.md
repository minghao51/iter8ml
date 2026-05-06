# Tabular Blueprint — Roadmap

> **Status:** Living document — v2.0 (merged from technical + strategic roadmaps)
> **Audience:** Personal toolkit (near-term) → Open-source template (long-term)
> **Philosophy:** Composable Lego bricks, not a monolith. Every module should be independently usable.
> **Last audited:** 2026-05-05 | 68 files, 7,161 lines in `src/`

---

## Table of Contents

### Part I — Foundation
1. [Vision & Problem Statement](#1-vision--problem-statement)
2. [Design Principles](#2-design-principles)
3. [Core Abstractions](#3-core-abstractions)
4. [Experiment Lifecycle](#4-experiment-lifecycle)
5. [Architecture Decision Log](#5-architecture-decision-log)

### Part II — Current State
6. [Implemented Features](#6-implemented-features)
7. [Audit Findings](#7-audit-findings)
8. [Codebase Metrics](#8-codebase-metrics)

### Part III — Forward Plan
9. [Phase A — Consolidate to DAG-Only Execution](#phase-a--consolidate-to-dag-only-execution)
10. [Phase B — Dead Code & Orphan Removal](#phase-b--dead-code--orphan-removal)
11. [Phase C — Benchmark-Validated Defaults](#phase-c--benchmark-validated-defaults)
12. [Phase D — Config Hygiene](#phase-d--config-hygiene)
13. [Phase E — Dependency Audit](#phase-e--dependency-audit)
14. [Phase F — Test Coverage Gaps](#phase-f--test-coverage-gaps)

### Part IV — Appendices
15. [Repository Structure](#15-repository-structure)
16. [Tech Stack & Dependency Manifest](#16-tech-stack--dependency-manifest)
17. [Testing & Validation Strategy](#17-testing--validation-strategy)
18. [Open-Source Readiness Checklist](#18-open-source-readiness-checklist)

---

# Part I — Foundation

> Timeless context: vision, principles, abstractions, and architectural decisions.
> These sections change infrequently. Update only when fundamental direction shifts.

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
- Executes via **Hamilton DAG** — every pipeline is a dataflow graph with explicit dependencies
- Is **LLM-ready**: exposes atomic tools via MCP for agentic automation
- Is structured so every module can be **extracted into a production API** without refactoring

### Non-Goals (v1)

- Not a deployment platform (no FastAPI/Triton serving layer)
- Not a distributed training framework (single-node GPU focus)
- Not a general NLP or vision toolkit (tabular + text-as-feature only)
- Not a "one-click AutoML" — user controls the iteration loop

---

## 2. Design Principles

### P1 — Functional over Class-heavy
Prefer pure functions with typed signatures over deep inheritance trees. Classes are reserved for stateful objects that genuinely need lifecycle management (`Trainer`, `DataAdapter`). This makes each module trivially extractable.

### P2 — Explicit over Magic
No hidden state. No silent fallbacks. If a model falls back from GPU to CPU, it logs it and says why. If TabPFN is skipped due to row count, it surfaces that decision in the experiment record.

### P3 — Polars as the Single Source of Truth
Data lives in `pl.DataFrame` or `pl.LazyFrame` until it reaches a model boundary. Conversion to NumPy happens at the last possible moment inside `DataAdapter`. No Pandas in `src/`.

### P4 — Config is Code
All experiment parameters are `Pydantic` models, not YAML files or argparse dicts. This gives IDE completion, runtime validation, and diff-friendly versioning.

### P5 — DAG-Native Execution
Every pipeline is a Hamilton DAG. Nodes are plain functions whose signatures define the dependency graph. No imperative fallback — the DAG IS the execution path. This ensures reproducibility, observability, and extensibility by default.

### P6 — Hardware-Aware by Default
The `ModelSelector` checks dataset size and available VRAM before routing. The user never needs to manually decide "is this too big for TabPFN?"

### P7 — Benchmark-Validated Defaults
Default hyperparameters are not guessed — they are validated against a curated OpenML benchmark suite. Every default change must include benchmark evidence.

---

## 3. Core Abstractions

### 3.1 `AbstractModel` Protocol

Every model wrapper — GBDT, TabPFN, Transformer — conforms to this protocol. No inheritance required; structural subtyping via `Protocol`.

```python
# models/base.py
from typing import Protocol
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

### 3.2 `DataAdapter`

Single point of truth for format conversion. Handles Polars → NumPy at the model boundary.

```python
# data/adapter.py
class DataAdapter:
    def transform(self, df: pl.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]: ...
```

### 3.3 `ModelSelector`

Enforces hardware-aware and data-size-aware routing.

```python
# models/selector.py
class ModelSelector:
    TABPFN_ROW_LIMIT = 50_000
    FT_TRANSFORMER_ROW_MIN = 50_000

    def select(
        self,
        n_rows: int,
        task: Literal["classification", "regression"],
        vram_gb: float = 0.0,
        include_baselines: bool = True,
    ) -> list[str]: ...
```

**Current routing logic:**

```
GPU present       → include TabPFN
n_rows < 500k     → [CatBoost, LightGBM, XGBoost]
n_rows >= 500k    → [LightGBM, XGBoost]
vram_gb > 12      → append FT-Transformer (n_rows > 50k)
vram_gb > 8       → append TabNet
```

### 3.4 `ExperimentConfig` (Pydantic)

Flat config with ~40 fields, grouped by concern:

```python
# config.py
class ExperimentConfig(BaseModel):
    # --- Core ---
    name: str
    task: TaskType
    target_col: str
    data_path: str
    cv_folds: int = 5
    cv_strategy: CVStrategy = ...
    models: list[str] | Literal["auto"] = "auto"
    metrics: list[str] = ...
    random_seed: int = 42
    # --- HPO ---
    run_hpo: bool = False
    hpo_n_trials: int = 50
    # --- Data Quality ---
    run_quality_audit: bool = True
    auto_clean_noise: bool = False
    noise_quality_threshold: float = 0.5
    # --- Feature Engineering ---
    afe_enabled: bool = False
    afe_top_k: int = 10
    afe_lift_threshold: float = 0.01
    afe_pruning: bool = False
    afe_prune_min_importance: float = 0.001
    # --- Embedding ---
    embedding_enabled: bool = False
    embedding_method: EmbeddingMethod = ...
    embedding_dim: int = 16
    # ... (9 embedding fields)
    # --- Tracking & Output ---
    tracker: TrackerType = TrackerType.JSONL
    workspace_dir: Path = Path("workspace")
    # --- Advanced ---
    max_workers: int = 1
    data_sample: float = 1.0
    calibration: Literal["none", "platt", "isotonic"] = "none"
    target_transform: Literal[...] = "none"
    # --- LLM ---
    llm_enabled: bool = False
    llm_model: str = "claude-sonnet-4-20250514"
```

### 3.5 JSONL Event Schema

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

### 3.6 Pluggable `Tracker` Protocol

The `Tracker` protocol wraps all telemetry emission. JSONL is always active. W&B/MLflow are opt-in additive mirrors.

```python
# engine/tracker.py
class Tracker(Protocol):
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def log_event(self, event: dict) -> None: ...
    def finish(self) -> None: ...
```

> **Key constraint:** `JSONLTracker` always runs, even when W&B is enabled. The JSONL file is the source of truth for the leaderboard and agent context. W&B is an additive mirror, never the primary store.

### 3.7 Data Hash Helper

Data lineage without DVC. Every loader call computes a deterministic SHA-256 hash of the DataFrame and stores it in the JSONL event under `data_hash`.

```python
# data/loaders.py
def get_data_hash(df: pl.DataFrame) -> str:
    row_hashes = df.hash_rows()
    combined = str(sorted(row_hashes.to_list())).encode()
    return "sha256:" + hashlib.sha256(combined).hexdigest()[:16]
```

### 3.8 Model Registry

Tracks the best model artifact per `(dataset_name, task)` pair. Updated automatically after every evaluation if the new model beats the current champion. File-locked for thread/process safety via `RegistryService`.

### 3.9 Hamilton DAG Execution

All pipelines execute through `PipelineExecutor`, which builds Hamilton drivers from node modules. Pipeline modes:

| Mode | Node Modules | Final Variable |
|---|---|---|
| `TRAINING` | preprocessing → data_preparation → model_selection → baselines → feature_engineering → model_training → state_generation | `training_state` |
| `DRIFT` | preprocessing → drift_detection | `drift_report` |
| `EXPORT` | preprocessing | `processed_dataframe` |
| `HPO` | preprocessing | `processed_dataframe` |
| `INFERENCE` | preprocessing | `processed_dataframe` |

---

## 4. Experiment Lifecycle

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
  │  2. Preprocessing    │  ← Polars expressions (Hamilton DAG)
  │     & Feature Eng.   │     Output: pl.DataFrame
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
  │  4. DataAdapter      │  ← Converts Polars → NumPy per model
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  5. Baselines        │  ← Naive + Linear baseline with lift metrics
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  6. AFE / Embedding  │  ← Optional: interaction discovery, entity embeddings
  │     (optional)       │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  7. Model Training   │  ← CV loop for each model, JSONL event per completion
  │     & Evaluation     │     Calibration applied post-training
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  8. State Generation │  ← Leaderboard, registry update, current_state.md
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  9. HPO (optional)   │  ← Optuna study on selected models
  │                      │     Warm-started from historical JSONL data
  └─────────────────────┘
```

---

## 5. Architecture Decision Log

### Active Decisions

| ID | Decision | Rationale | Alternatives Rejected |
|---|---|---|---|
| ADR-001 | Polars as sole DataFrame engine | Speed, lazy API, Arrow-native. No Pandas. | Pandas (slow), DuckDB (less ergonomic for feature eng.) |
| ADR-002 | `AbstractModel` as Protocol, not ABC | Structural subtyping — no inheritance tax. Third-party models can conform without wrapping. | `abc.ABC` (requires inheritance), duck typing (no IDE support) |
| ADR-003 | JSONL as primary event store | Zero infra dependency. LLM-readable. Queryable via Polars directly. W&B/MLflow are additive mirrors. | MLflow as primary (heavy), W&B as primary (requires account) |
| ADR-004 | Optuna for HPO | Backend-agnostic, supports pruning, integrates with every model in stack. | Ray Tune (overkill for single-node), Hyperopt (less maintained) |
| ADR-005 | Hamilton as sole execution path | DAG-native execution for all pipelines. Eliminates dual-path duplication. | Imperative fallback (duplication), optional Hamilton (two codepaths to maintain) |
| ADR-006 | Pydantic configs, not YAML | IDE completion, runtime validation, `ruff`-friendly, diffable. | Hydra (complex), YAML (no validation), argparse (no structure) |
| ADR-007 | LLM layer via MCP | Atomic tools for agentic automation. JSONL is the primary interface the LLM reads. | LLM baked into trainer (tight coupling) |
| ADR-008 | TabPFN row-count warning (soft guardrail) | Warning at 50k rows. User can override. Silent degradation is worse than a clear signal. | Hard error at 10k (too restrictive), no guardrail (silent degradation) |
| ADR-009 | Pluggable `Tracker` protocol | JSONL always works with zero config. Teams can opt into W&B without any trainer refactor. | Baking W&B directly into trainer (breaks offline use) |
| ADR-010 | Data hash in JSONL, not DVC | SHA-256 of `hash_rows()` gives lineage for free with no infra. | DVC (overkill for single-user), no lineage (silent mutations) |
| ADR-011 | File-locked `registry.json` | Zero-dependency champion tracking with thread/process safety via `filelock`. | MLflow Model Registry (needs server), no registry (can't promote models) |
| ADR-012 | `typer` CLI over argparse / click | typer generates `--help` docs from type hints automatically. Shell completion built-in. | argparse (verbose, no types), click (more boilerplate) |
| ADR-013 | ZenML kept out of `core/`, example only | Plain functions are already ZenML-compatible as steps. No core dep needed. | ZenML as core dep (too heavy) |
| ADR-014 | LightEx deferred — migration trigger documented | Worth evaluating only if JSONL query logic exceeds 2 days to extend. | Adopting in Phase 1 (unnecessary dep) |
| ADR-015 | `Cleanlab` audit skippable via config flag | On large datasets (>500k rows), Cleanlab's CV step can add minutes. Default-on, bypassable. | Always mandatory (breaks large-dataset iteration speed) |

### New Decisions (v2.0 Roadmap)

| ID | Decision | Rationale | Alternatives Rejected |
|---|---|---|---|
| ADR-016 | Hamilton as sole execution path — no imperative fallback | Dual-path (imperative + Hamilton) caused ~800 lines of duplicated logic. Every operation exists in both `engine/` and `pipelines/nodes/`. Eliminating the imperative path removes duplication, reduces call depth from 10 to 6, and makes the DAG the single source of truth. | Keeping imperative fallback (duplication), making Hamilton optional (two paths to test) |
| ADR-017 | Benchmark-validated defaults via OpenML suite | Default hyperparameters are currently generic one-size-fits-all (e.g., `afe_lift_threshold=0.01` keeps noise features, `CatBoostConfig.task_type="CPU"` ignores GPU). A curated OpenML benchmark suite provides empirical evidence for every default. | Guessed defaults (current state), user-surveyed defaults (biased) |
| ADR-018 | Dependency tier model: core / [base] / [deep] / [tracking] / [agent] / [audit] | The monolithic `[opinion]` group bundles 12 packages. Users who want wandb don't need torch. Splitting into focused tiers gives users minimal-install paths and clarifies what each tier enables. | Single `[opinion]` group (all-or-nothing), everything in core (bloated) |
| ADR-019 | Flat config with section grouping | 40+ fields in `ExperimentConfig` but only ~15 used on a typical run. Keeping flat structure for simplicity while adding docstring grouping and `model_overrides` escape hatch avoids the complexity of nested sub-configs while maintaining discoverability. | Nested sub-configs (more types, more indirection), reducing fields (limits extensibility) |

---

# Part II — Current State

> Factual snapshot of what's implemented, what's broken, and what's bloated.
> Updated after each audit cycle.

---

## 6. Implemented Features

### Data Layer
- [x] Polars-native loaders (CSV, Parquet, SQLite) with `get_data_hash()`
- [x] `DataAdapter`: Polars → NumPy conversion at model boundary
- [x] Cleanlab quality audit with configurable `noise_quality_threshold`
- [x] `auto_clean_noise` flag for automatic noise handling
- [x] Leakage detection audit (permutation importance on naive baseline)
- [x] Preprocessing cache (hash-based disk cache)
- [x] High-cardinality categorical embedding engine
- [ ] **Visual Lineage Surface**: Surface the Mermaid graph in `state` command and experiment reports
    - *Implementation*: Add section to `StateObserver` to render `pipeline_lineage` from `experiments.jsonl`

### Models
- [x] `AbstractModel` Protocol (structural subtyping)
- [x] GBDT wrappers: CatBoost, LightGBM, XGBoost with shared base class
- [x] TabPFN v2 with hardware-aware routing and row-count warnings
- [x] Deep models: FT-Transformer (PyTorch), TabNet (pytorch-tabular)
- [x] `ModelSelector` with hardware + data-size-aware routing
- [x] String-keyed model factory with lazy imports
- [x] Naive + Linear baselines with "Lift over Baseline" metrics

### Engine & Training
- [x] Hamilton DAG execution via `PipelineExecutor` (5 pipeline modes)
- [x] Cross-validation evaluator (KFold, StratifiedKFold, TimeSeriesSplit)
- [x] Metrics registry: classification (ROC-AUC, F1, log-loss) + regression (RMSE, MAE, R²)
- [x] `Tracker` protocol with `JSONLTracker` (log rotation), `WandbTracker`, `MLflowTracker`
- [x] `TrackingHook` adapter: Hamilton `NodeExecutionHook` → `Tracker` protocol
- [x] Calibration: Platt scaling + Isotonic regression via `CalibratedModel`
- [x] Automated Feature Engineering (AFE): targeted interaction discovery, target transformation
- [x] Entity embedding training (MLP + autoencoder modes)

### HPO
- [x] Optuna study factory with per-model default search spaces
- [x] Pre-warmed HPO: inject historical trials from JSONL log
- [x] PedAnova parameter importance to refine search spaces
- [ ] **Optuna Dashboard Integration**: `--view` flag to launch local `optuna-dashboard`

### Observability
- [x] `StateObserver`: generates `current_state.md` from logs + registry
- [x] LLM commentary via `litellm` (gated behind `llm_enabled=False`)
- [x] Drift detection: PSI (univariate) + Domain Classifier (multivariate) + KS/Chi²
- [x] SHAP explainability (gated behind `shap_enabled=False`)
- [x] Leaderboard auto-generation from JSONL
- [x] Configuration diffing: `tabblueprint diff <id1> <id2>`

### MCP & LLM
- [x] FastMCP server with 8 atomic tools for LLM agents
- [x] `TabularAgent` module for natural-language explanations

### Services & CLI
- [x] File-locked `RegistryService` with atomic saves
- [x] `ReportService` with metric direction logic
- [x] `ExportService`: packages champion model + preprocessing nodes + predictor script
- [x] CLI with 10 commands: `init`, `run`, `leaderboard`, `registry`, `hardware`, `drift`, `state`, `hpo`, `diff`, `export`

### Type Safety & Error Handling
- [x] Custom exception hierarchy: `DataLoadError`, `ModelFitError`, `RegistryError`
- [x] `track_errors()` decorator for typed error logging
- [x] `RestrictedUnpickler` for safe deserialization
- [ ] **Strict Typing**: Finalize remaining mypy errors, transition to `strict = true`

### Not Yet Implemented
- [ ] AFE pruning (RFE or null-importance checks)
- [ ] ONNX/TorchScript export
- [ ] Remote data loaders (S3, GCS, Snowflake)
- [ ] Uncertainty quantification (prediction intervals)
- [ ] ZenML example pipeline

---

## 7. Audit Findings (2026-05-05)

> Phase A, B, and C completed. Dual-path duplication eliminated, dead code removed, benchmark infrastructure validated.
> Remaining items are tracked in Phase D–F below.

### Resolved (Phase A)

| Issue | Resolution |
|---|---|
| Dual-path duplication (~800 lines) | Imperative path deleted; DAG-only execution |
| `PipelineExecutor` 36-param signature | Accepts `ExperimentConfig` directly; `_config_to_inputs()` helper |
| Dead service instantiation in `Trainer` | Removed; `Trainer` is 117-line thin wrapper |
| `engine/model_trainer.py` dead code | Deleted (318 lines) |

### Resolved (Phase B)

| Issue | Resolution |
|---|---|
| `engine/drift_checker.py` | Deleted |
| `engine/explainability_service.py` | Deleted |
| `models/deep/text_encoder.py` | Deleted |
| `_to_tensor()` + `_to_dataset()` | Deleted from `data/adapter.py` |
| `_MODE_MODULES` empty dict | Deleted from `pipelines/executor.py` |
| `pipelines/hamilton_executor.py` | Deleted |
| Dead constants functions | Deleted from `constants.py` |
| `accelerate` / `datasets` deps | Removed from `[opinion]` |

### Resolved (Phase C)

| Issue | Resolution |
|---|---|
| `CatBoostConfig.task_type` hardcoded `"CPU"` | Changed to `"auto"` with `model_validator` that resolves to `"GPU"`/`"CPU"` at instantiation |
| `CatBoostModel` GPU detection | Added `_detect_gpu()` using `catboost.utils.get_gpu_count` |
| No benchmark suite | `benchmarks/openml_benchmark.py` with 9 datasets, sweep support, regression checking |
| No baseline storage | `benchmarks/results/baseline_summary.json` generated via `--save-baseline` |
| No benchmark CI | `.github/workflows/benchmarks.yml` runs on `v*` tags with artifact upload |
| No benchmark docs | `benchmarks/README.md` with quickstart, sweep, and regression instructions |
| Missing `ricci` dataset | Added to `configs/default_benchmark.yaml` |

### Remaining Concerns

| Setting | Current Default | Issue |
|---|---|---|
| `afe_lift_threshold` | 0.01 | Keeps noise features (1% lift is too permissive). Sweep infra ready; needs full-DAG benchmark run. |
| `noise_quality_threshold` | 0.5 | Drops 50% of flagged rows — aggressive. Sweep infra ready; needs full-DAG benchmark run. |
| `llm_model` | `"claude-sonnet-4-20250514"` | Hardcoded model name in config |
| `TabPFNConfig.n_estimators` | 4 | Sweep infra ready; needs benchmark run with TabPFN installed. |
| `afe_pruning` | `False` | AFE runs without pruning — keeps all interactions |

---

## 8. Codebase Metrics

### File Size Distribution (Top 20)

| Lines | File |
|-------|------|
| 477 | `cli.py` |
| 359 | `data/embedding_engine.py` |
| 318 | `data/feature_engine.py` |
| 280 | `pipelines/executor.py` |
| 262 | `engine/hpo.py` |
| 250 | `engine/state_observer.py` |
| 229 | `services/registry_service.py` |
| 222 | `services/export_service.py` |
| 215 | `pipelines/nodes/feature_engineering.py` |
| 207 | `engine/hpo_warmstart.py` |
| 186 | `config.py` |
| 184 | `services/report_service.py` |
| 182 | `models/deep/sparse_embedder.py` |
| 173 | `pipelines/nodes/model_training.py` |
| 171 | `models/deep/ft_transformer.py` |
| 153 | `engine/tracker.py` |
| 149 | `engine/hpo_importance.py` |
| 149 | `engine/evaluator.py` |
| 145 | `pipelines/nodes/data_preparation.py` |
| 135 | `monitoring/explainability.py` |

### Summary

| Metric | Value |
|---|---|
| Total `.py` files in `src/` | 68 |
| Total lines in `src/` | 7,161 |
| Test files | 47 |
| Test lines | 5,618 |
| Test:source ratio | 0.79:1 |
| CLI → model.fit() call depth (Hamilton path) | 6 layers |
| `ExperimentConfig` fields | 40+ |
| Dead/orphaned files | 0 |
| Dead service instances in Trainer | 0 |

---

# Part III — Forward Plan

> Priority-ordered phases for reducing bloat and improving defaults.
> Each phase is independently executable. Phases A-C are highest priority.

---

## Phase A — Consolidate to DAG-Only Execution

> **Priority:** Critical | **Estimated impact:** ~800 lines removed, call depth 10 → 6

### A.1: Refactor `PipelineExecutor` to accept `ExperimentConfig` directly

Replace `run_training()`'s 36 individual parameters with `(config: ExperimentConfig, df: pl.DataFrame)`. Build the Hamilton `inputs` dict from config in one place inside the executor.

**Before:**
```python
executor.run_training(
    df=df, target_col=..., task=..., config_models=...,
    experiment_name=..., run_id=..., workspace_dir=...,
    vram_gb=..., cv_folds=..., cv_strategy=..., metrics=...,
    # ... 24 more params
)
```

**After:**
```python
executor.run_training(config=config, df=df, run_id=run_id)
```

- [x] Refactor `PipelineExecutor.run_training()` signature
- [x] Build Hamilton `inputs` dict from `ExperimentConfig` in executor
- [x] Update `Trainer._try_hamilton_training()` to pass config directly

### A.2: Delete imperative service layer

All logic lives in Hamilton nodes only. The following files are deleted entirely:

- [x] Delete `engine/data_preparation.py` (153 lines)
- [x] Delete `engine/feature_engineer.py` (113 lines)
- [x] Delete `engine/embedding_trainer.py` (267 lines)
- [x] Move any unique logic from these files into the corresponding Hamilton nodes

### A.3: Remove imperative fallback from `Trainer`

- [x] Delete `Trainer._run_imperative()`
- [x] `Trainer.run()` calls `PipelineExecutor.run_training()` as sole path
- [x] Remove dead service instantiation (`DriftChecker`, `ExplainabilityService`)
- [x] `Trainer` becomes a thin wrapper: load data → call executor → update state

### A.4: Merge sequential/concurrent training in `ModelTrainer`

- [x] Merge `_train_sequential` + `_train_concurrent` into single `_train_models(max_workers)`
- [x] Use `ThreadPoolExecutor` conditionally (max_workers=1 → sequential)

### A.5: Flatten call depth

**Target: 6 layers**
```
CLI → Trainer.run() → PipelineExecutor.run(config, df) → Hamilton DAG → node → model.fit()
```

- [x] Nodes call models/evaluator/services directly, no intermediate wrapping
- [x] Remove re-construction of `ExperimentConfig` inside nodes (pass as input)

### Files Modified

| File | Action |
|---|---|
| `engine/trainer.py` | Remove imperative path, remove dead services |
| `pipelines/executor.py` | Accept `ExperimentConfig`, `_config_to_inputs()` helper |
| `engine/model_trainer.py` | **DELETE** |
| `engine/data_preparation.py` | **DELETE** |
| `engine/feature_engineer.py` | **DELETE** |
| `engine/embedding_trainer.py` | **DELETE** (moved to `data/embedding_engine.py`) |
| `pipelines/nodes/*.py` | Accept `ExperimentConfig` as input, remove re-construction |

---

## Phase B — Dead Code & Orphan Removal

> **Priority:** High | **Estimated impact:** ~300 lines removed, 3 files deleted

- [x] Delete `engine/drift_checker.py` (59 lines) — instantiated but never called
- [x] Delete `engine/explainability_service.py` (53 lines) — instantiated but never called
- [x] Delete `models/deep/text_encoder.py` (78 lines) — orphaned, never imported
- [x] Delete `_to_tensor()` + `_to_dataset()` from `data/adapter.py`
- [x] Delete `_MODE_MODULES` empty dict from `pipelines/executor.py`
- [x] Delete `pipelines/hamilton_executor.py` (deprecated wrapper)
- [x] Remove export of deprecated module from `pipelines/__init__.py`
- [x] Delete `from_cv_strategy`, `from_model_name`, `from_tracker_type` from `constants.py`
- [x] Remove `accelerate` from `[opinion]` extras (never imported)
- [x] Remove `datasets` from `[opinion]` extras (only used by dead adapter code)
- [x] Remove tests for deleted modules

---

## Phase C — Benchmark-Validated Defaults

> **Priority:** High | **Estimated effort:** 1-2 weeks

### C.1: OpenML Benchmark Suite

Curate 8-10 datasets covering task × size combinations:

| Dataset | Task | Size | OpenML ID |
|---|---|---|---|
| credit-g | Binary classification | 1k rows | 31 |
| adult | Binary classification | 48k rows | 1590 |
| ricci | Binary classification | 6k rows | — |
| covertype | Multiclass classification | 581k rows | 1596 |
| shuttle | Multiclass classification | 58k rows | 40685 |
| iris | Multiclass classification | 150 rows | 61 |
| house_16H | Regression | 22k rows | 572 |
| quake | Regression | 2k rows | 772 |
| diabetes | Regression | 442 rows | sklearn |
| breast_cancer | Binary classification | 569 rows | sklearn |

- [x] Add `benchmarks/openml_benchmark.py` script
- [x] Each dataset: run all default models, record metrics + timing
- [x] Store results in `benchmarks/results/` as JSON
- [x] Add `benchmarks/README.md` with reproduction instructions

### C.2: Validate & Fix Defaults

- [x] `CatBoostConfig.task_type`: auto-detect from `HardwareProfile.has_gpu` (not hardcoded `"CPU"`)
- [ ] `afe_lift_threshold`: test 0.01 vs 0.03 vs 0.05 vs 0.10 on benchmark suite — *sweep infra ready; needs full-DAG benchmark*
- [ ] `noise_quality_threshold`: test 0.3 vs 0.5 vs 0.7 on datasets with known noise — *sweep infra ready; needs full-DAG benchmark*
- [ ] GBDT `iterations`/`learning_rate`: grid search (500/0.1, 1000/0.05, 2000/0.01) — *sweep infra ready*
- [ ] `TabPFNConfig.n_estimators`: test 2 vs 4 vs 8 on small datasets — *sweep infra ready; needs TabPFN installed*
- [ ] `LightGBMConfig.num_leaves`: test 15 vs 31 vs 63 vs 127 — *sweep infra ready*
- [x] Document validated defaults in `benchmarks/DEFAULTS.md`

### C.3: CI Integration

- [x] GitHub Action: run benchmark suite on every release tag
- [x] Assert: no default model score regression > 2% on any dataset
- [x] Store results as CI artifacts for comparison

---

## Phase D — Config Hygiene

> **Priority:** Medium

- [ ] Add section docstrings to `ExperimentConfig` for field grouping (Core / HPO / Data Quality / Feature Engineering / Embedding / Tracking / Advanced / LLM)
- [ ] Change `llm_model` default from hardcoded `"claude-sonnet-4-20250514"` to env var `TABBLUEPRINT_LLM_MODEL` or `None`
- [ ] Add `model_overrides: dict[str, dict] | None = None` for per-model param overrides without touching `ModelConfigs`
- [ ] Add deprecation warnings for fields that should become env vars
- [ ] Ensure `model_validator` correctly handles all task × strategy combos

---

## Phase E — Dependency Audit

> **Priority:** Medium

### E.1: Split `[opinion]` into focused tiers

```toml
[project.optional-dependencies]
base = ["catboost>=1.2", "lightgbm>=4.0", "xgboost>=2.0", "optuna>=3.6", "sf-hamilton>=1.70"]
deep = ["torch>=2.3", "transformers>=4.40", "tabpfn>=2.0", "pytorch-tabular>=1.0"]
tracking = ["wandb>=0.17", "mlflow>=2.13"]
agent = ["mcp>=0.9", "litellm>=1.40"]
audit = ["shap>=0.44", "cleanlab>=2.6"]
docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.25", "mike>=2.0", "pymdown-extensions>=10.0"]
full = ["tabular-blueprint[base,deep,tracking,agent,audit]"]
```

### E.2: Lazy import enforcement

- [ ] `mcp/tools.py`: defer FastMCP import to server startup, not module level
- [ ] All `[deep]` imports: behind try/except with clear error message
- [ ] Add test: verify `import tabular_blueprint` completes in <1s with only core deps

### E.3: Remove unused

- [ ] Remove `accelerate` from extras (already done in Phase B)
- [ ] Remove `datasets` from extras (already done in Phase B)

---

## Phase F — Test Coverage Gaps

> **Priority:** Medium

### New Tests

- [ ] `tests/test_model_configs.py`: validate all config defaults, HPO search spaces, invalid param rejection
- [ ] `tests/test_data_cache.py`: preprocessing cache hit/miss/expiry
- [ ] `tests/test_tracking_hook.py`: Hamilton node → Tracker event emission
- [ ] Integration test: **full Hamilton DAG path** end-to-end (not just imperative — which will be removed in Phase A)
- [ ] Benchmark regression test: assert default scores don't degrade between releases

### Updated Tests

- [ ] Remove tests for modules deleted in Phases A and B
- [ ] Update integration tests to use `ExperimentConfig`-based executor API (from Phase A.1)

---

## Success Metrics

| Metric | Current | Target |
|---|---|---|
| Total lines in `src/` | 7,172 | ~7,200 (Phase A+B+C complete) |
| CLI → model.fit() call depth | 6 layers | 6 layers |
| Import time (core deps only) | Untested | <1s |
| Default benchmark coverage | 10 datasets | 8-10 OpenML datasets |
| Config fields | 40+ (flat, ungrouped) | 40+ (flat, grouped with docstrings) |
| Test:source ratio | 0.79:1 | 0.85:1 |
| Dead code lines | 0 | 0 |
| Files with duplicated logic | 0 | 0 |
| Dependency tiers | 2 (core + opinion) | 6 (core + base + deep + tracking + agent + audit) |

---

# Part IV — Appendices

> Reference material updated to reflect actual codebase state.

---

## 15. Repository Structure

```
src/tabular_blueprint/
├── __init__.py
├── cli.py                          # Typer CLI (477 lines, 10 commands)
├── config.py                       # ExperimentConfig + HardwareProfile (186 lines)
├── constants.py                    # Enums: TaskType, CVStrategy, ModelName, etc. (75 lines)
├── exceptions.py                   # Exception hierarchy + track_errors decorator (74 lines)
├── py.typed
│
├── data/
│   ├── __init__.py
│   ├── loaders.py                  # CSV, Parquet, SQLite via Polars (106 lines)
│   ├── adapter.py                  # DataAdapter: Polars → NumPy (88 lines)
│   ├── feature_engine.py           # AFE interaction discovery (318 lines)
│   ├── leakage.py                  # LeakageReport / detect_leakage() (88 lines)
│   ├── quality.py                  # Cleanlab quality audit + noise cleaning (109 lines)
│   ├── cache.py                    # Hash-based preprocessing cache (74 lines)
│   └── embedding_engine.py         # High-cardinality categorical embeddings (359 lines)
│
├── models/
│   ├── __init__.py
│   ├── base.py                     # AbstractModel Protocol (16 lines)
│   ├── factory.py                  # String-keyed model registry with lazy imports (42 lines)
│   ├── selector.py                 # ModelSelector: hardware + size routing (66 lines)
│   ├── baselines.py                # NaiveBaseline + LinearBaseline (103 lines)
│   ├── gbdt_base.py                # Shared GBDT base class (72 lines)
│   ├── model_configs.py            # Per-model Pydantic configs + HPO search spaces (116 lines)
│   ├── conventional/
│   │   ├── catboost_model.py       # CatBoost wrapper (68 lines)
│   │   ├── lightgbm_model.py       # LightGBM wrapper (53 lines)
│   │   └── xgboost_model.py        # XGBoost wrapper (49 lines)
│   ├── tabular_foundation/
│   │   └── tabpfn_model.py         # TabPFN v2 wrapper (97 lines)
│   └── deep/
│       ├── ft_transformer.py       # FT-Transformer (PyTorch) (171 lines)
│       ├── tabnet_model.py         # TabNet (pytorch-tabular) (113 lines)
│       └── sparse_embedder.py      # Sparse autoencoder (182 lines)
│
├── engine/
│   ├── __init__.py
│   ├── trainer.py                  # Main orchestrator (117 lines)
│   ├── evaluator.py                # CV strategies, metrics registry (149 lines)
│   ├── tracker.py                  # Tracker protocol + JSONL/W&B/MLflow impls (153 lines)
│   ├── hpo.py                      # Optuna study factory (262 lines)
│   ├── hpo_warmstart.py            # Historical trial injection (207 lines)
│   ├── hpo_importance.py           # PedAnova parameter importance (149 lines)
│   ├── calibration.py              # Platt/Isotonic calibration (99 lines)
│   └── state_observer.py           # Generates current_state.md (250 lines)
│
├── pipelines/
│   ├── __init__.py
│   ├── executor.py                 # PipelineExecutor: builds Hamilton drivers (197 lines)
│   ├── preprocessing.py            # Standalone preprocessing DAG (25 lines)
│   ├── hooks/
│   │   └── tracking_hook.py        # Hamilton NodeExecutionHook → Tracker (61 lines)
│   └── nodes/
│       ├── preprocessing.py        # 9 nodes: null fill, date decomp, encoding (94 lines)
│       ├── data_preparation.py     # 7 nodes: quality, adapter, leakage, target transform (145 lines)
│       ├── model_selection.py      # 1 node: auto or explicit model list (26 lines)
│       ├── baselines.py            # 2 nodes: naive + linear baseline (48 lines)
│       ├── feature_engineering.py  # 6 nodes: AFE, embedding (with @config variants) (215 lines)
│       ├── model_training.py       # 4 nodes: train + evaluate each model (173 lines)
│       ├── state_generation.py     # 1 node: aggregate results + update registry (90 lines)
│       └── drift_detection.py      # 9 nodes: PSI, domain classifier (with @config variants) (125 lines)
│
├── monitoring/
│   ├── drift.py                    # KS + Chi² drift detection (97 lines)
│   ├── psi_drift.py                # PSI drift detection (94 lines)
│   ├── domain_classifier.py        # Domain classifier drift (77 lines)
│   └── explainability.py           # SHAP-based explainer (135 lines)
│
├── services/
│   ├── __init__.py
│   ├── registry_service.py         # File-locked JSON champion registry (229 lines)
│   ├── report_service.py           # Leaderboard + metric direction logic (184 lines)
│   └── export_service.py           # Champion export to portable directory (222 lines)
│
├── mcp/
│   └── tools.py                    # FastMCP server with 8 atomic tools (152 lines)
│
├── llm/
│   └── __init__.py                 # TabularAgent via litellm (147 lines)
│
└── utils/
    ├── jsonl.py                    # JSONL log reading (53 lines)
    └── safe_pickle.py              # RestrictedUnpickler allowlist (71 lines)
```

---

## 16. Tech Stack & Dependency Manifest

### Environment & Packaging

| Tool | Role |
|---|---|
| `uv` | Env + package management |
| `ruff` | Linting + formatting |
| `pyproject.toml` | Single source of truth |
| `mypy` | Type checking (targeting `strict = true`) |

### Core Dependencies (always installed)

| Package | Version Floor | Role |
|---|---|---|
| `polars` | ≥ 1.0 | Core DataFrame engine |
| `pydantic` | ≥ 2.0 | Config validation |
| `scikit-learn` | ≥ 1.4 | Metrics, CV splitters, baselines |
| `numpy` | ≥ 1.26 | Array format at model boundary |
| `typer` | ≥ 0.12 | CLI with `--help` from type hints |
| `rich` | ≥ 13.0 | Terminal formatting |
| `psutil` | ≥ 5.9 | HardwareProfile detection |
| `pyyaml` | ≥ 6.0 | Config file loading |
| `filelock` | ≥ 3.12 | Registry file locking |

### Optional Dependency Tiers (after Phase E)

| Tier | Extras Group | Packages | Enables |
|---|---|---|---|
| **Base ML** | `[base]` | catboost, lightgbm, xgboost, optuna, sf-hamilton | GBDT models, HPO, DAG execution |
| **Deep Learning** | `[deep]` | torch, transformers, tabpfn, pytorch-tabular | TabPFN, FT-Transformer, TabNet |
| **Experiment Tracking** | `[tracking]` | wandb, mlflow | W&B/MLflow tracker backends |
| **LLM Agent** | `[agent]` | mcp, litellm | MCP server, LLM commentary |
| **Data Audit** | `[audit]` | shap, cleanlab | SHAP explainability, noise detection |
| **Documentation** | `[docs]` | mkdocs-material, mkdocstrings, mike, pymdown-extensions | Docs site |
| **Everything** | `[full]` | meta-package installing all above | Complete install |

### Removed Dependencies

| Package | Reason |
|---|---|
| `accelerate` | Never imported anywhere (Phase B) |
| `datasets` (HuggingFace) | Only used by dead `_to_dataset()` code (Phase B) |
| `pydantic-settings` | Not currently used |
| `skrub` | Not currently imported |

---

## 17. Testing & Validation Strategy

### Unit Tests (`tests/`)

- `DataAdapter` round-trip: Polars → NumPy → assert no precision loss
- `ModelSelector` routing: assert correct model list for each size/hardware combo
- `Cleanlab` wrapper: synthetic noisy labels, assert flagged rows match expected
- `ExperimentConfig` validation: assert Pydantic catches invalid field combos
- `get_data_hash()`: identical DataFrames → identical hashes; mutated frame → different hash
- `Tracker` fan-out: `JSONLTracker` + `WandbTracker` receive identical metric dicts
- `ModelConfigs`: all configs validate, HPO search spaces produce valid params
- `PreprocessingCache`: hit/miss/expiry behavior
- `TrackingHook`: Hamilton node completion → correct Tracker events

### Integration Tests

- Full Hamilton DAG pipeline on `sklearn.datasets.make_classification` (1k rows, 20 features)
- Assert JSONL event written with correct schema (including `data_hash`) after each model run
- Assert `leaderboard.md` updates after run
- Assert `registry.json` updates when new champion detected
- TabPFN guardrail: warning emitted when n_rows > 50k
- CLI: `tabblueprint leaderboard` exits 0 and prints expected table headers
- CLI: `tabblueprint run --config ...` executes full Hamilton DAG path

### Benchmark Regression Tests (after Phase C)

- Run default models on OpenML benchmark suite
- Assert: no score regression > 2% on any dataset vs. baseline
- Run on every release tag via GitHub Actions

---

## 18. Open-Source Readiness Checklist

To be completed before public release:

**Documentation**
- [ ] `README.md` with 60-second quickstart
- [ ] `CONTRIBUTING.md` with PR guidelines and code style rules
- [ ] `LICENSE` (MIT)
- [ ] `CHANGELOG.md` initialized and maintained
- [ ] All Pydantic models have docstrings and field descriptions
- [ ] All public functions have type hints and docstrings
- [ ] Example configs for 3 dataset types (classification, regression, text)

**Code Quality**
- [ ] Remove all hardcoded local paths (use `pathlib.Path` + config)
- [ ] `workspace/` fully gitignored (only `.gitkeep` committed)
- [ ] `ruff` passes with zero warnings on full repo
- [ ] No `import pandas` anywhere in `src/`
- [ ] `mypy --strict` passes with zero errors
- [ ] `import tabular_blueprint` completes in <1s with only core deps

**CI/CD**
- [ ] GitHub Actions: `ruff check` + `mypy` + `pytest tests/unit/` on every PR
- [ ] GitHub Actions: benchmark suite on every release tag
- [ ] Dependabot config for weekly dep updates
- [ ] Pre-commit hooks: `ruff format`, `ruff check`, `pytest tests/unit/`

**Environment**
- [ ] `Dockerfile` with CUDA 12.4 base + `uv` installed
- [ ] `.devcontainer/devcontainer.json` for VS Code / Codespaces
- [ ] `docker-compose.yml` with optional MLflow tracking server service

---

*Last updated: 2026-05-05 | Phase A+B+C complete — Next: Phase D*
