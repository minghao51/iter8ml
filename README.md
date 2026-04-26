# Tabular Blueprint

A high-velocity iteration framework for tabular machine learning. Built for single-node efficiency with Polars-native speed.

## Quick Start

```bash
# Install core dependencies
uv sync

# Install optional deep-learning dependencies (torch, transformers, tabpfn)
uv sync --extra deep

# Run an experiment
uv run tabblueprint run --data path/to/data.csv --target target_column

# Or use a config file
uv run tabblueprint run --config examples/credit_risk.py
```

## CLI Commands

```bash
# Initialize workspace
uv run tabblueprint init --data path/to/data.csv

# Run experiments
uv run tabblueprint run --data data.csv --target label
uv run tabblueprint run --config examples/credit_risk.py --models catboost lightgbm

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

- **Hamilton-Powered Data Layer**: Visual lineage and DAG-based preprocessing.
- **Smart Routing**: Hardware-aware model selection (TabPFN, GBDTs, Transformers).
- **Leakage Audit**: Pre-train checks for "too-good-to-be-true" features.
- **Warm-start HPO**: Injects historical experiment results into Optuna searches.
- **Explainability**: Automated SHAP beeswarm and importance plots.

## Architecture

```
src/tabular_blueprint/
├── data/          # Polars-native data layer (loaders, adapter, leakage)
├── models/        # Model wrappers (GBDT, TabPFN, FT-Transformer, TabNet)
├── engine/        # Trainer, evaluator, HPO, state observation
└── monitoring/    # Drift detection & explainability
```

## Optional Integrations

```bash
# Weights & Biases tracking
uv sync --extra wandb

# MLflow tracking
uv sync --extra mlflow

# LLM/MCP server (Claude Desktop / Agentic loop)
uv sync --extra llm
```

## Running in Docker

```bash
docker build -t tabular-blueprint .
docker run -v $(pwd):/workspace tabular-blueprint tabblueprint run --data data.csv --target label
```

## Development

```bash
# Install development/test extras
uv sync --extra dev --extra llm

# Run tests
uv run --extra dev --extra llm pytest tests/unit -v

# Lint
uv run --extra dev ruff check .

# Format
uv run --extra dev ruff format .
```

## License

MIT
