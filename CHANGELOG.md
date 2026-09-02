# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Behavior-breaking config surface changes (2026-09-01 pipeline/config audit fix
pass + W6 report-trust workstream). Read **Changed/Removed** before upgrading
existing configs.

### Added
- `iter8 run --check`: side-effect-free config↔data preflight
  (`verification/preflight.py`) — target presence/nulls/constancy, task
  misdeclaration warnings, CV feasibility, timeseries-without-dates, unknown
  `ignore_cols`, ID-like leakage hints. Exit 1 on blocking issues.
- `ignore_cols` config field: drop ID/leaky columns before feature engineering
  (unknown columns fail loudly; applied after `row_id` computation so medallion
  split alignment is preserved).
- `positive_class` config field (classification, binary): explicitly orients
  the target so the positive class encodes to 1 — `roc_auc` no longer depends
  on value appearance order. Unknown values fail in preflight and prep; the
  export bundle records it in `metadata.json` (`iter8 export --positive-class`).
- `iter8ml.metrics` entry-point group for custom metrics (`module:func`,
  optional `func.task` scoping and `func.lower_is_better`).
- `primary_metric` config field: one ranking rule for lift, leaderboard, and
  registry promotion (default `metrics[0]`; must be a member of `metrics`).
- `iter8 hpo --config <file>`: drive HPO from an ExperimentConfig (task,
  target, data, folds, metrics, seed, `ignore_cols`, `positive_class`,
  per-model `model_overrides` as fixed params); explicit flags override.
- Fold-level reporting: `cv_std` captured end-to-end (`mean ±std` in
  events/state/CLI); `experiment_started` records `data_digest`,
  `library_versions`, row counts on the flat path.
- Regression example config (`examples/house_prices_regression.yaml`).

### Changed
- **Unknown config keys now fail at parse time** (`extra="forbid"`): typos
  like `cv_fold: 10` exit 1 instead of being silently ignored.
- **Metrics, `cv_strategy`, and `primary_metric` are validated at parse time**
  against the task's registry; `stratified` CV is rejected for regression.
- `config.random_seed` now reaches CV splitters, model constructors,
  calibration, and data sampling (previously only the medallion split).
- `--quick`'s `data_sample` is now actually applied (was validated-but-inert).
- `tracker` config field selects the tracking backend (JSONL/W&B/MLflow).
- Event log (`experiments.jsonl`) reading is torn-write tolerant and includes
  rotated backups with deduplication; a truncated final line no longer bricks
  `current_state.md` generation or resume.
- Leaderboards sort task-isolated (classification and regression never
  interleave) and `latest_run` is the max-timestamp entry, not file order.
- `compute_lift` returns `None` for missing metrics or a 0 baseline instead of
  fabricating a 0.0 lift; unevaluable baselines are skipped with a warning.
- All-models-fail now fails the run (`ModelFitError` → exit 1) instead of
  exiting 0 with an empty leaderboard.
- HPO routes the raw frame through the same preprocessing chain as training
  (string categoricals no longer crash LightGBM/XGBoost) and raises when fewer
  than `min(n_trials, max(3, n_trials // 10))` trials complete.

### Removed
- `shap_enabled` and `drift_detection` config keys (loud deprecation errors;
  use `iter8 drift` / `iter8 hpo` respectively).

### Fixed
- Legacy flat-key configs silently replaced the default 8-step pipeline with a
  fragment (no FEATURE_ENGINEERING step → DAG crash); legacy keys now seed the
  full default step list.
- Calibration: requested-but-unapplied calibration logs a warning instead of
  silently downgrading; pre-calibration CV scores are marked with an asterisk
  legend in events/state/leaderboards.
- `--models` CLI values are validated before overwriting config models.

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
