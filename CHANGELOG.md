# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow (ruff lint + format check, pytest unit tests on push/PR)
- Dependabot configuration for weekly dependency updates
- Pre-commit hooks (ruff format, ruff check, pytest)
- Regression integration test (`test_full_pipeline_catboost_regression`)
- `datasets>=2.14` as `transformers` optional dependency
- `CHANGELOG.md` initialized
- Example experiment config (`configs/examples/credit_risk.py`)

### Fixed
- **Critical:** `TextEncoder.model_name` property was shadowing the `__init__` `model_name` parameter — renamed property to `encoder_name`
- **Critical:** `Evaluator.evaluate()` was reusing the same model instance across CV folds, causing state leakage — now accepts `model_cls` and instantiates fresh per fold
- Hardcoded workspace paths in `Trainer` — replaced with configurable `workspace_dir` parameter
- `datasets` package used in `DataAdapter._to_dataset()` but not declared in dependencies
- FT-Transformer now uses `accelerate.Accelerator` for multi-GPU / mixed-precision support

### Changed
- `Trainer` accepts optional `workspace_dir` parameter for isolation during testing
- `optimize_model()` now requires explicit `task` argument for correct metric routing
