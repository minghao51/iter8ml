# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.1.0] - 2026-08-06

First public release. A high-velocity iteration framework for tabular ML:
Polars-native data layer, Hamilton DAG pipeline, hardware-aware model routing
across 7 families (CatBoost/LightGBM/XGBoost, TabPFN, FT-Transformer, TabNet,
naive/linear), Optuna HPO with warmstarting, 3 drift detectors, SHAP
explainability, probability calibration, and an MCP server for agentic ML.

### Added
- CPU benchmark suite (CatBoost/LightGBM/XGBoost, 5-fold CV) with committed,
  reproducible results across 9 OpenML/scikit-learn datasets.
- `iter8ml[gbdt]` lean extra for the core `iter8 run` path; `[train]`/`[full]`
  add deep models, tracking, LLM/MCP, SHAP.
- GitHub Actions CI workflow (ruff lint + format check, pytest unit tests on push/PR)
- Dependabot configuration for weekly dependency updates
- Pre-commit hooks (ruff format, ruff check, pytest)
- Regression integration test (`test_full_pipeline_catboost_regression`)
- `datasets>=2.14` as `transformers` optional dependency
- `CHANGELOG.md` initialized
- Example experiment config (`configs/examples/credit_risk.py`)

### Fixed
- OpenMP deadlock on Linux hybrid-core (P+E) CPUs that hung lightgbm/xgboost;
  threads now capped at 8 on Linux (override via `OMP_NUM_THREADS`).
- PEP 639 license metadata conflict (license expression + classifier removed).
- **Critical:** `TextEncoder.model_name` property was shadowing the `__init__` `model_name` parameter — renamed property to `encoder_name`
- **Critical:** `Evaluator.evaluate()` was reusing the same model instance across CV folds, causing state leakage — now accepts `model_cls` and instantiates fresh per fold
- Hardcoded workspace paths in `Trainer` — replaced with configurable `workspace_dir` parameter
- `datasets` package used in `DataAdapter._to_dataset()` but not declared in dependencies
- FT-Transformer now uses `accelerate.Accelerator` for multi-GPU / mixed-precision support

### Changed
- `Trainer` accepts optional `workspace_dir` parameter for isolation during testing
- `optimize_model()` now requires explicit `task` argument for correct metric routing
