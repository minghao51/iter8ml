# Validated Defaults

Default hyperparameters are validated against the OpenML benchmark suite
defined in `configs/default_benchmark.yaml`. Every default change must
include benchmark evidence.

## Current Defaults (pre-benchmark)

| Setting | Current Default | Status |
|---------|----------------|--------|
| `CatBoostConfig.task_type` | `"auto"` | Resolved at config time via `model_validator` |
| `afe_lift_threshold` | `0.01` | Unvalidated — sweep infra ready, needs full-DAG runs |
| `noise_quality_threshold` | `0.5` | Unvalidated — sweep infra ready, needs full-DAG runs |
| `TabPFNConfig.n_estimators` | `4` | Unvalidated — sweep infra ready, needs TabPFN install |

## Baseline Results (Quick Mode)

Run: `2026-05-05` via `run_openml_benchmark.py --quick`

### Classification (ROC-AUC)

| Dataset | CatBoost | LightGBM | XGBoost | TabPFN |
|---------|----------|----------|---------|--------|
| breast_cancer | 0.9947 | 0.9920 | 0.9937 | N/A |

### Regression (R²)

| Dataset | CatBoost | LightGBM | XGBoost |
|---------|----------|----------|---------|
| diabetes | 0.4573 | 0.4315 | 0.3910 |
| house_16H | 0.9371 | 0.8671 | 0.8783 |

> Note: Quick mode samples large datasets to 20%. Full baseline pending.

## Parameter Sweeps

### CatBoost GPU Auto-Detect

- **Change:** `CatBoostConfig.task_type` defaults to `"auto"` — resolved to `"GPU"` if `catboost.utils.get_gpu_count() > 0`, else `"CPU"`
- **Evidence:** Baseline established. GPU vs CPU comparison pending on GPU-enabled runner.
- **Decision:** Implemented in `model_configs.py` + `catboost_model.py`

### AFE Lift Threshold

Candidates: `0.01`, `0.03`, `0.05`, `0.10`

| Threshold | Mean Score | vs Baseline | Decision |
|-----------|-----------|-------------|----------|
| 0.01      | TBD       | TBD         | Pending — needs full-DAG benchmark (current infra tests model-level params only) |
| 0.03      | TBD       | TBD         | Pending |
| 0.05      | TBD       | TBD         | Pending |
| 0.10      | TBD       | TBD         | Pending |

### Noise Quality Threshold

Candidates: `0.3`, `0.5`, `0.7`

| Threshold | Mean Score | vs Default | Decision |
|-----------|-----------|------------|----------|
| 0.3       | TBD       | TBD        | Pending — needs full-DAG benchmark |
| 0.5       | TBD       | TBD        | Current  |
| 0.7       | TBD       | TBD        | Pending |

### TabPFN n_estimators

Candidates: `2`, `4`, `8`

| n_estimators | Mean AUC | vs Default | Decision |
|-------------|----------|------------|----------|
| 2           | TBD      | TBD        | Pending — needs TabPFN installed |
| 4           | TBD      | TBD        | Current  |
| 8           | TBD      | TBD        | Pending |

### LightGBM num_leaves

Candidates: `15`, `31`, `63`, `127`

| num_leaves | Mean Score | vs Default | Decision |
|-----------|-----------|------------|----------|
| 15        | TBD       | TBD        | Pending |
| 31        | TBD       | TBD        | Current  |
| 63        | TBD       | TBD        | Pending |
| 127       | TBD       | TBD        | Pending |

## How to Run Benchmarks

```bash
# Full benchmark suite (may take hours)
uv run python benchmarks/run_openml_benchmark.py

# Quick mode (samples large datasets)
uv run python benchmarks/run_openml_benchmark.py --quick

# Parameter sweep
uv run python benchmarks/run_openml_benchmark.py --sweep-config benchmarks/sweeps/catboost_task_type.yaml --quick

# Save baseline
uv run python benchmarks/run_openml_benchmark.py --quick --save-baseline

# Check regression against baseline
uv run python benchmarks/run_openml_benchmark.py --quick --check-regression benchmarks/results/baseline_summary.json
```

Results are saved to `benchmarks/results/` as JSON files.
