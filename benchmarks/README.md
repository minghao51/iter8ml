# Benchmark Suite

Validate default hyperparameters against a curated OpenML + sklearn dataset collection.

## Quick Start

```bash
# Run on quick mode (samples large datasets)
uv run python benchmarks/run_openml_benchmark.py --quick

# Run full suite
uv run python benchmarks/run_openml_benchmark.py
```

Results are saved to `benchmarks/results/` as JSON.

## Parameter Sweeps

Define variants in a YAML file:

```yaml
# benchmarks/sweeps/catboost_task_type.yaml
variants:
  - name: default
    config: {}
  - name: gpu
    config:
      catboost:
        task_type: GPU
```

Run the sweep:

```bash
uv run python benchmarks/run_openml_benchmark.py --sweep-config benchmarks/sweeps/catboost_task_type.yaml --quick
```

## Regression Checking

1. Save a baseline after a known-good run:
   ```bash
   uv run python benchmarks/run_openml_benchmark.py --quick --save-baseline
   ```

2. Compare future runs against it:
   ```bash
   uv run python benchmarks/run_openml_benchmark.py --quick --check-regression benchmarks/results/baseline_summary.json
   ```

The check exits with code 1 if any model regresses by more than the threshold (default 2%).

## Datasets

See `configs/default_benchmark.yaml` for the full list. Covers:

| Dataset | Task | Size | Source |
|---------|------|------|--------|
| credit-g | classification | 1k rows | OpenML 31 |
| adult | classification | 48k rows | OpenML 1590 |
| covertype | classification | 581k rows | OpenML 1596 |
| shuttle | classification | 58k rows | OpenML 40685 |
| iris | classification | 150 rows | OpenML 61 |
| house_16H | regression | 22k rows | OpenML 572 |
| quake | regression | 2k rows | OpenML 772 |
| diabetes | regression | 442 rows | sklearn |
| breast_cancer | classification | 569 rows | sklearn |

## CI Integration

The benchmark suite runs automatically on every release tag (`v*`) via `.github/workflows/benchmarks.yml`.

It runs in `--quick` mode, uploads results as artifacts, and checks for errors.
