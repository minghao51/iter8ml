# Tabular Blueprint

A high-velocity iteration framework for tabular machine learning. Built for single-node efficiency with Polars-native speed.

## Core Philosophy & Design Goals

- **Fast feedback loops** – from data change to model metric in minutes (not hours/days).
- **Low boilerplate** – common operations (splits, encoding, scaling, evaluation) are one-liners or config-driven.
- **Reproducibility** – every run is tracked (code, data hash, hyperparameters, metrics).
- **Config over code** – hyperparameters, feature lists, model types, and even pipeline steps are defined in YAML/TOML.
- **Extensible** – easy to drop in custom transformers, metrics, or models.

## Quick Start

**Option A: Project install (recommended for development)**

```bash
uv sync --extra base      # ML models + HPO + Hamilton DAG
uv sync --extra opinion   # Deep learning, SHAP, experiment tracking, LLM/MCP, data quality
uv sync --extra docs      # Documentation tooling

# Run an experiment
uv run tabblueprint run --data path/to/data.csv --target target_column
```

**Option B: Ephemeral run (no install, for quick experiments)**

```bash
# Run directly with uvx (uses the CLI entry point from git)
uvx --from git+https://github.com/your-org/iter8ml tabblueprint run --data data.csv --target label

# Or after publishing to PyPI:
uvx tabular-blueprint run --data data.csv --target label
```

**Option C: Permanent install on PATH**

```bash
uv tool install git+https://github.com/your-org/iter8ml
tabblueprint run --data data.csv --target label
```

## CLI Commands

All commands below use the `uv run` prefix. Replace with `uvx tabular-blueprint` or just `tabblueprint` if using Option B or C from Quick Start.

```bash
# Initialize workspace
uv run tabblueprint init --data path/to/data.csv

# Run experiments
uv run tabblueprint run --data data.csv --target label
uv run tabblueprint run --config examples/credit_risk.yaml --models catboost lightgbm
uv run tabblueprint run --config examples/credit_risk.toml
uv run tabblueprint run --config examples/credit_risk.json

# Quick iteration mode (2 folds, 20% data, skip SHAP/AFE/calibration)
uv run tabblueprint run --data data.csv --target label --quick

# Resume a previous run (skip already-completed models)
uv run tabblueprint run --data data.csv --target label --resume

# Compare runs (Side-by-side config & metric diff)
uv run tabblueprint diff exp_id_1 exp_id_2

# Inspect results
uv run tabblueprint leaderboard
uv run tabblueprint leaderboard --top 5 --metric roc_auc

# Manage model registry
uv run tabblueprint registry show

# Hyperparameter optimization (with warm-start from history)
uv run tabblueprint hpo --data data.csv --target label --model catboost --trials 100

# Detect data drift (KS-test, PSI, or Domain Classifier)
uv run tabblueprint drift --reference train.parquet --new batch.parquet

# View experiment state & pipeline lineage
uv run tabblueprint state

# Check hardware (Auto-detected CUDA/VRAM/CPU)
uv run tabblueprint hardware
```

## Key Features

- **Hamilton DAG Pipelines** — Function-based DAG with visual lineage, config variants, and lifecycle hooks. See [pipeline-architecture.md](docs/pipeline-architecture.md).
- **Smart Model Routing** — Hardware-aware selection across 7 model families. See [models.md](docs/models.md).
- **Automated Feature Engineering** — Permutation importance top-K, pairwise interaction discovery, and pruning. See [feature-engineering.md](docs/feature-engineering.md).
- **Leakage Audit** — Pre-train permutation test flags "too-good-to-be-true" features. See [preprocessing.md](docs/preprocessing.md).
- **Warm-start HPO** — Historical trial injection + PedAnova importance + search space refinement via Optuna. See [hpo.md](docs/hpo.md).
- **3 Drift Detectors** — KS/Chi-squared, PSI, and domain classifier AUC. See [drift-detection.md](docs/drift-detection.md).
- **SHAP Explainability** — TreeExplainer (GBDTs) and KernelExplainer (others) with beeswarm + dependence plots. See [explainability.md](docs/explainability.md).
- **Probability Calibration** — Platt scaling and isotonic regression post-training. See [evaluation.md](docs/evaluation.md).

## Pipeline Modes

The Hamilton DAG executor supports 5 pipeline modes:

| Mode | CLI Command | Terminal Node | Purpose |
|------|------------|---------------|---------|
| `TRAINING` | `run` | `training_state` | Full experiment (7 modules) |
| `DRIFT` | `drift` | `drift_report` | Compare reference vs live data |
| `EXPORT` | `export` | `processed_dataframe` | Package champion model |
| `HPO` | `hpo` | `processed_dataframe` | Hyperparameter optimization |
| `INFERENCE` | — | `processed_dataframe` | Batch prediction |

## Experiment Configuration

All pipeline behavior is controlled by `ExperimentConfig`. Key knobs:

```python
from tabular_blueprint.config import ExperimentConfig

config = ExperimentConfig(
    name="credit_risk_v2",
    task="classification",
    target_col="default",
    data_path="data/credit.csv",

    # Models: "auto" for hardware-aware routing, or a list
    models="auto",

    # Automated feature engineering
    afe_enabled=True,
    afe_top_k=15,
    afe_pruning=True,

    # Target transformation (log1p, yeo-johnson, box-cox, auto)
    target_transform="auto",

    # Probability calibration (platt, isotonic)
    calibration="platt",

    # Drift detection (psi, domain_classifier, both)
    drift_detection="both",

    # SHAP explainability
    shap_enabled=True,

    # Concurrency (auto-reduced for low-VRAM GPUs)
    max_workers=2,
)
```

See [pipeline-architecture.md](docs/pipeline-architecture.md) for the full config reference.

## MCP Server (Agentic ML)

Connect Claude Desktop or any MCP client to run experiments conversationally:

```bash
# Install LLM/MCP dependencies
uv sync --extra opinion

# Start the MCP server
uv run tabblueprint mcp
```

Available MCP tools: `get_experiment_state`, `get_column_stats`, `run_baseline`, `run_hpo`, `get_event_log`, `registry_show`, `registry_promote`, `detect_drift`, `export_champion`.

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tabular-blueprint": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/iter8ml", "tabblueprint", "mcp"]
    }
  }
}
```

Or with uvx (no local clone needed):

```json
{
  "mcpServers": {
    "tabular-blueprint": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/your-org/iter8ml", "--with", "tabular-blueprint[opinion]", "tabblueprint", "mcp"]
    }
  }
}
```

## Architecture

```
src/tabular_blueprint/
├── cli.py                # CLI entry points
├── config.py             # ExperimentConfig, HardwareProfile
├── constants.py          # Enums (TaskType, CVStrategy, ModelName, TrackerType)
├── data/                 # Polars-native data layer
│   ├── loaders.py        #   CSV, Parquet, SQLite ingestion
│   ├── adapter.py        #   Format conversion (numpy, tensor, HF Dataset)
│   ├── quality.py        #   Cleanlab label noise detection
│   ├── leakage.py        #   Permutation-based leakage audit
│   └── feature_engine.py #   Target transforms, interaction discovery, pruning
├── models/               # Model wrappers + selection
│   ├── baselines.py      #   Naive (mean/mode) + Linear (Logistic/Ridge)
│   ├── conventional/     #   CatBoost, LightGBM, XGBoost
│   ├── deep/             #   FT-Transformer, TabNet, DeBERTa text encoder
│   ├── tabular_foundation/  # TabPFN v2
│   ├── selector.py       #   Hardware-aware model routing
│   └── factory.py        #   Lazy-import model registry
├── engine/               # Orchestration & evaluation
│   ├── trainer.py        #   Top-level experiment orchestrator
│   ├── evaluator.py      #   K-fold CV + metrics registry
│   ├── model_trainer.py  #   Sequential/concurrent training loop
│   ├── hpo.py            #   Optuna study factory + optimization loop
│   ├── calibration.py    #   Platt scaling + isotonic regression
│   └── tracker.py        #   JSONL, W&B, MLflow trackers
├── pipelines/            # Hamilton DAG orchestration
│   ├── executor.py       #   PipelineExecutor (5 modes)
│   ├── nodes/            #   8 node modules (preprocessing, data_prep, etc.)
│   └── hooks/            #   TrackingHook (lifecycle events)
├── monitoring/           # Drift detection & explainability
│   ├── drift.py          #   KS-test + Chi-squared
│   ├── psi_drift.py      #   Population Stability Index
│   ├── domain_classifier.py  # Multivariate drift via classifier AUC
│   └── explainability.py #   SHAP TreeExplainer + KernelExplainer
└── services/             # Reporting, registry, export
    ├── registry_service.py   # Thread-safe model registry
    ├── report_service.py     # Leaderboard + markdown reports
    └── export_service.py     # Portable champion model packaging
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design document.

## Documentation

| Document | Content |
|----------|---------|
| [models.md](docs/models.md) | All 7 model implementations, selector logic, factory registry, hyperparameter defaults |
| [evaluation.md](docs/evaluation.md) | CV strategies, metric formulas (RMSE, ROC AUC, R², etc.), lift, calibration |
| [preprocessing.md](docs/preprocessing.md) | Null imputation, date decomposition, encoding, target transforms, quality audit, leakage |
| [feature-engineering.md](docs/feature-engineering.md) | Permutation importance, pairwise interactions, pruning |
| [hpo.md](docs/hpo.md) | Optuna pruners, warmstarting, PedAnova importance, search space refinement |
| [drift-detection.md](docs/drift-detection.md) | KS/Chi-squared, PSI formula, domain classifier AUC |
| [explainability.md](docs/explainability.md) | SHAP TreeExplainer/KernelExplainer, beeswarm + dependence plots |
| [data-loading.md](docs/data-loading.md) | CSV/Parquet/SQLite loading, security measures, data hashing |
| [pipeline-architecture.md](docs/pipeline-architecture.md) | Hamilton DAG composition, config variants, hooks, extension guide |

## Optional Integrations

**With `uv sync` (project install):**

```bash
uv sync --extra base        # ML models (catboost, lightgbm, xgboost) + HPO + Hamilton DAG
uv sync --extra opinion     # Everything optional: DL, SHAP, wandb, MLflow, MCP/LLM, cleanlab
uv sync --extra docs        # Documentation tooling (mkdocs, mkdocstrings, mike)
```

| Extra | Packages |
|-------|----------|
| `base` | catboost, lightgbm, xgboost, optuna, sf-hamilton |
| `opinion` | shap, cleanlab, torch, accelerate, transformers, tabpfn, pytorch-tabular, datasets, wandb, mlflow, mcp, litellm |
| `docs` | mkdocs-material, mkdocstrings, mike, pymdown-extensions |

**With `uvx` (ephemeral):**

```bash
uvx --from git+https://github.com/your-org/iter8ml --with wandb tabblueprint run --data data.csv --target label
```

## Running in Docker

```bash
docker build -t tabular-blueprint .
docker run -v $(pwd):/workspace tabular-blueprint tabblueprint run --data data.csv --target label
```

## Development

```bash
# Install development/test extras
uv sync --group dev --extra base --extra opinion

# Run tests
uv run --group dev --extra base --extra opinion pytest tests/unit -v

# Lint
uv run --group dev ruff check .

# Format
uv run --group dev ruff format .
```

## License

MIT
