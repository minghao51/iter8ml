# iter8ml — Technical Roadmap

> **Status:** Active | **Last updated:** 2026-05-15
> **Audience:** Contributors and users

---

## What iter8ml Is

A **high-velocity iteration framework for tabular ML**:

- **30-second baseline** for any tabular task (CatBoost / TabPFN)
- **Progressive depth**: quick baseline → tuned GBDT → deep model
- **Polars-native** throughout, Hamilton DAG execution
- **LLM-ready**: MCP server exposes atomic tools for agent automation
- **Programmatic-first**: `ExperimentSession` API, CLI is thin wrappers

---

## Architecture

```
ExperimentSession          ← primary API (session.py)
  ├── Workspace            ← filesystem paths dataclass (workspace.py)
  ├── Trainer              ← orchestrator (engine/trainer.py)
  │     └── PipelineExecutor  ← Hamilton DAG builder (engine/pipelines/executor.py)
  │           ├── nodes/prep.py
  │           ├── nodes/features.py
  │           ├── nodes/train.py
  │           └── nodes/drift_detection.py
  ├── ReportService        ← leaderboard (services/reporting.py)
  ├── ExportService        ← champion packaging (services/export.py)
  ├── RegistryService      ← thread-safe model registry (services/registry.py)
  └── StateObserver        ← markdown state summaries (engine/state_observer.py)
```

**Call depth:** `Session.run() → Trainer → PipelineExecutor → Hamilton DAG → node → model.fit()` (5 layers)

---

## Current Module Layout

```
src/iter8ml/
├── session.py, workspace.py, config.py, constants.py, exceptions.py
├── analysis/        # drift detection, SHAP explainability
├── cli/             # typer CLI (thin wrappers around Session)
├── data/            # loading, adapter, features, embedding, quality, leakage, cache
├── engine/          # trainer, evaluator, tracker, HPO, calibration, state_observer
│   ├── models/      # baselines, GBDTs, TabPFN, FT-Transformer, TabNet, factory, selector
│   └── pipelines/   # executor, nodes, hooks
├── services/        # registry, export, reporting, LLM agent, MCP server
└── utils/           # JSONL I/O, safe pickle
```

---

## What's Implemented

| Layer | Features |
|-------|----------|
| **Data** | Polars loaders (CSV/Parquet/SQLite), DataAdapter, Cleanlab quality audit, leakage detection, preprocessing cache, categorical embedding engine, automated feature engineering |
| **Models** | CatBoost, LightGBM, XGBoost, TabPFN v2, FT-Transformer, TabNet, naive/linear baselines. Hardware-aware `ModelSelector`. Plugin-ready factory via entry points. |
| **Training** | Hamilton DAG pipeline, cross-validation (KFold/Stratified/TimeSeries), calibration (Platt/Isotonic), Optuna HPO with warmstarting and parameter importance |
| **Tracking** | JSONLTracker (default, always active), WandbTracker, MLflowTracker. TrackingHook for Hamilton node observability. |
| **Analysis** | Drift detection (KS/Chi², PSI, Domain Classifier), SHAP explainability (TreeExplainer/KernelExplainer) |
| **Services** | File-locked registry, champion export, leaderboard reports, state generation, LLM commentary via litellm, MCP server with 8 tools |
| **CLI** | `iter8 init/run/hpo/drift/leaderboard/export/state/diff/registry/hardware` — all delegate to `ExperimentSession` |
| **Quality** | Custom exception hierarchy, `RestrictedUnpickler`, mypy strict, ruff, pip-audit, pre-commit hooks |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Polars-only (no Pandas) | Speed, lazy API, Arrow-native |
| `AbstractModel` Protocol | Structural subtyping — third-party models conform without wrapping |
| JSONL as primary event store | Zero infra, LLM-readable, queryable via Polars |
| Hamilton DAG-only execution | Single codepath, no imperative fallback |
| `Workspace` dataclass | Separates runtime filesystem from config |
| `ExperimentSession` as primary API | Programmatic-first; CLI is thin wrapper |
| Plugin model factory | Entry points for third-party models |
| Pydantic configs | IDE completion, runtime validation, diffable |

---

## Dependency Tiers

| Tier | Extras | Key Packages |
|------|--------|-------------|
| Core | (default) | polars, pydantic, scikit-learn, typer, numpy |
| Base ML | `[train]` | catboost, lightgbm, xgboost, optuna, sf-hamilton |
| Deep Learning | `[deep]` | torch, transformers, tabpfn, pytorch-tabular |
| Tracking | `[tracking]` | wandb, mlflow |
| Agent | `[agent]` | mcp, litellm |
| Audit | `[audit]` | shap, cleanlab |

---

## Next Steps

| Priority | Item | Status |
|----------|------|--------|
| High | Benchmark-validated defaults sweep | Infra ready, needs full-DAG run |
| High | AFE pruning (RFE or null-importance) | Not started |
| Medium | ONNX/TorchScript export | Not started |
| Medium | Remote data loaders (S3, GCS) | Not started |
| Low | Uncertainty quantification | Not started |
| Low | Optuna Dashboard integration | Not started |

---

*Legacy roadmap (979 lines, archived 2026-05-15) → [plans/20260513-legacy-roadmap.md](plans/20260513-legacy-roadmap.md)*
