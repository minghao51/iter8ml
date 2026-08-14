# iter8ml

A high-velocity iteration framework for tabular ML.

---

## Getting Started

```python
import iter8ml as iml

session = iml.ExperimentSession()

config = iml.ExperimentConfig(
    name="demo",
    task=iml.TaskType.CLASSIFICATION,
    target_col="y",
    data_path="data.csv",
    models=["catboost", "lightgbm"],
)

df = iml.load_data("data.csv")
results = session.run(config, df)

lb = session.leaderboard()
print(lb)

session.export("demo:classification")
```

---

## Public API

| Symbol | Source | Purpose |
|--------|--------|---------|
| `ExperimentSession` | `session.py` | Primary high-level interface |
| `Workspace` | `workspace.py` | Filesystem paths dataclass |
| `ExperimentConfig` | `config.py` | Pydantic experiment config |
| `HardwareProfile` | `config.py` | GPU/RAM/CPU detection |
| `Trainer` | `engine/trainer.py` | Orchestrates experiment runs |
| `Evaluator` | `engine/evaluator.py` | Cross-validation + metrics |
| `ModelSelector` | `engine/models/selector.py` | Hardware-aware model routing |
| `Tracker` / `JSONLTracker` | `engine/tracker.py` | Tracking protocol + default impl |
| `ExportService` | `services/export.py` | Package champion models |
| `RegistryService` | `services/registry.py` | Thread-safe model registry |
| `PromotionResult` | `services/registry.py` | Pydantic promotion result |
| `ReportService` | `services/reporting.py` | Leaderboard reports |
| `load_data` | `data/loader.py` | CSV/Parquet → Polars |
| `available_model_names` | `engine/models/factory.py` | Registered model names |
| `get_model_class` | `engine/models/factory.py` | Resolve model class by name |
| Enums: `TaskType`, `CVStrategy`, `FeatureStrategy`, `EmbeddingMethod`, `TrackerType` | `constants.py` | Type-safe config |
| Exceptions: `TabularBlueprintError`, `DataLoadError`, `ModelFitError`, `RegistryError` | `exceptions.py` | Error hierarchy |

---

## Session API

`ExperimentSession` is the recommended entry point:

```python
session = ExperimentSession()                    # auto-creates Workspace

# Run experiment
results = session.run(config, df)

# Inspect results
lb = session.leaderboard(metric="roc_auc")       # → pl.DataFrame

# Promote and export
session.promote("run_id", "key")                  # → PromotionResult
session.export("key")                             # → Path

# Drift detection
session.drift_check(ref_df, live_df, method="psi")

# State summary (LLM-readable markdown)
md = session.state()
```

### Workspace

```python
ws = Workspace(root="./workspace")    # or set ITER8ML_WORKSPACE env var
ws.init()                             # create dirs + touch files

ws.experiments_path   # workspace/experiments.jsonl
ws.registry_path      # workspace/registry.json
ws.artifacts_dir      # workspace/artifacts/
ws.exports_dir        # workspace/exports/
ws.state_path         # workspace/current_state.md
ws.leaderboard_path   # workspace/leaderboard.md
```

---

## Guides

- [Pipeline Architecture](pipeline-architecture.md)
- [Data Loading](data-loading.md)
- [Preprocessing](preprocessing.md)
- [Feature Engineering](feature-engineering.md)
- [Models](models.md)
- [Evaluation & Metrics](evaluation.md)
- [Hyperparameter Optimization](hpo.md)
- [Drift Detection](drift-detection.md)
- [Explainability](explainability.md)
- [Design Decisions](design-decisions.md)

## Notebooks

- [Notebook Tutorials](notebooks/index.md)

## CLI

```
iter8 init                  Initialize workspace
iter8 run config.yaml       Run experiment
iter8 leaderboard           Show leaderboard
iter8 hpo config.yaml       Run HPO
iter8 drift ref.csv live.csv  Drift detection
iter8 export key            Export champion
iter8 state                 Show experiment state
```
