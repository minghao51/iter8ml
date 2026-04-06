# Tabular Blueprint

A high-velocity iteration framework for tabular machine learning.

## Quick Start

```bash
# Install dependencies
uv sync

# Run an experiment
uv run tabblueprint run --data path/to/data.csv --target target_column

# Or use a config file
uv run tabblueprint run --config configs/my_experiment.py
```

## CLI Commands

```bash
# Initialize workspace
uv run tabblueprint init --data path/to/data.csv

# Run experiments
uv run tabblueprint run --data data.csv --target label
uv run tabblueprint run --config configs/credit_risk.py --models catboost lightgbm

# Inspect results
uv run tabblueprint leaderboard
uv run tabblueprint leaderboard --top 5 --metric roc_auc

# Manage model registry
uv run tabblueprint registry show

# Hyperparameter optimization
uv run tabblueprint hpo --data data.csv --target label --model catboost --trials 100

# Detect data drift
uv run tabblueprint drift --reference train.parquet --new batch.parquet

# View experiment state
uv run tabblueprint state

# Check hardware
uv run tabblueprint hardware
```

## Architecture

```
core/
├── data/          # Polars-native data layer
├── models/        # Model wrappers (GBDT, TabPFN, Transformers)
├── engine/        # Trainer, evaluator, HPO, tracking
└── monitoring/    # Drift detection (Phase 3)
```

## Optional Integrations

```bash
# Weights & Biases tracking
uv sync --extra wandb

# MLflow tracking
uv sync --extra mlflow

# Hamilton pipeline DAGs
uv sync --extra hamilton

# LLM/MCP server (Phase 2)
uv sync --extra llm
```

## Running in Docker

```bash
docker build -t tabular-blueprint .
docker run -v $(pwd):/workspace tabular-blueprint tabblueprint run --data data.csv --target label
```

## Development

```bash
# Run tests
uv run pytest tests/unit -v

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## License

MIT
