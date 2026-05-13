# Tabular Blueprint — Current State

Last updated: 2026-05-13

## What's Implemented

| Feature | Backend | Frontend |
|---------|---------|----------|
| CLI (`init`, `run`, `leaderboard`, `registry`, `hardware`, `drift`, `state`, `hpo`, `diff`, `export`) | Full | N/A (CLI-only) |
| Data loading (CSV, Parquet, SQLite) | Full | N/A |
| Preprocessing (null fill, date decomposition, categorical encoding) | Full | N/A |
| Model suite (CatBoost, LightGBM, XGBoost, TabPFN, FT-Transformer, TabNet, Naive/Linear baselines) | Full | N/A |
| Cross-validation (KFold, Stratified, TimeSeries) | Full | N/A |
| Evaluation metrics (roc_auc, f1_macro, accuracy, log_loss, rmse, mae, r2) | Full | N/A |
| Hardware-aware model selection (ModelSelector) | Full | N/A |
| Hyperparameter optimization (Optuna + warmstart + importance) | Full | N/A |
| Automated feature engineering (interaction discovery + pruning) | Full | N/A |
| Target transformation (log1p, box-cox, yeo-johnson, auto) | Full | N/A |
| Probability calibration (Platt, Isotonic) | Full | N/A |
| Drift detection (KS/Chi2, PSI, Domain Classifier) | Full | N/A |
| SHAP explainability (TreeExplainer, KernelExplainer + plots) | Full | N/A |
| Data quality audit (Cleanlab label noise) | Partial | N/A |
| Leakage detection (permutation-based audit) | Full | N/A |
| Embedding engine (Entity, Autoencoder for high-cardinality features) | Full | N/A |
| Model registry (file-locked, atomic writes, auto-promote) | Full | N/A |
| Export service (portable prediction packages) | Full | N/A |
| Experiment tracking (JSONL + W&B + MLflow) | Full | N/A |
| Hamilton DAG pipeline orchestration | Full | N/A |
| LLM commentary (litellm integration) | Full | N/A |
| MCP server (10 tools for LLM agents) | Full | N/A |
| Preprocessing cache (NumPy .npy) | Full | N/A |
| Experiment resume (skip completed models) | Full | N/A |

## Stubbed / Unimplemented

All files below raise `NotImplementedError` or return `501`:

- None found. All registered modules contain working implementations.

## Known Bugs

| Severity | Issue | Location |
|----------|-------|----------|
| High | `WandbTracker` and `MLflowTracker` missing `current_run_id` attribute, violating the `Tracker` Protocol. Will raise `AttributeError` if accessed. | `src/tabular_blueprint/engine/tracker.py:91` (WandbTracker), `src/tabular_blueprint/engine/tracker.py:121` (MLflowTracker) |
| High | `XGBoostModel._build_params` calls `self.params.pop("random_seed", 42)` which mutates `self.params` in place. Subsequent calls lose the `random_seed` key. | `src/tabular_blueprint/models/conventional/xgboost_model.py:26` |
| Medium | `LightGBMModel.fit` calls `self._build_params()` via `self._build_params()` inside `_train_model`, duplicating params and potentially overriding `_model` state set by `_create_model`. | `src/tabular_blueprint/models/conventional/lightgbm_model.py:40-44` |
| Medium | `feature_engineering.py` Hamilton `@config` variants (`training_features__afe_enabled`, `training_features__embedding_enabled`) are never registered when Hamilton is not installed. Falls through silently with `MagicMock` — no fallback path exists. | `src/tabular_blueprint/pipelines/nodes/feature_engineering.py:148-211` |
| Medium | `DriftReport` dataclass in `drift_detection.py` shadows the `DriftReport` Pydantic model from `monitoring/drift.py`. Importing both in the same namespace is ambiguous. | `src/tabular_blueprint/pipelines/nodes/drift_detection.py:22` |
| Low | `PipelineExecutor.run_training` always returns `None` when Hamilton is not installed, with no warning or fallback. | `src/tabular_blueprint/pipelines/executor.py:181-182` |
| Low | `_fit_importance_model` in `feature_engineering.py` calls `cls(task=task).fit(X, y) or cls(task=task)`. The `or` branch is dead code — `fit()` returns `None` for all models, so the second `cls(task=task)` always executes, creating an unfitted model. | `src/tabular_blueprint/pipelines/nodes/feature_engineering.py:36` |
| Low | `TabPFNModel` always selects CPU when no CUDA GPU detected, even when MPS (Apple Silicon) is available. | `src/tabular_blueprint/models/tabular_foundation/tabpfn_model.py:28-37` |

## Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| High | `safe_pickle.py` uses `pickle.dump` without restriction — `safe_dump` serializes arbitrary objects. Only deserialization is restricted via whitelist. A malicious actor with write access could craft a valid `.pkl` file that bypasses the whitelisted prefixes (e.g., any `sklearn.*` class is allowed). | `src/tabular_blueprint/utils/safe_pickle.py:61-66` |
| Medium | `load_sqlite` uses keyword-blocklist approach for SQL injection prevention. This is fragile — edge cases with comments (`--`), hex-encoded keywords, or Unicode bypasses could evade detection. Parameterized queries would be safer. | `src/tabular_blueprint/data/loaders.py:66-88` |
| Medium | `config.py` allows executing arbitrary Python files as config when `--allow-unsafe-config` is passed. The flag is user-facing with no additional sandboxing. | `src/tabular_blueprint/config.py:171-185` |
| Low | `NaiveBaseline.load` calls `np.load(path + ".npz", allow_pickle=False)` which is safe, but the `.npz` extension is appended automatically — if a user passes a path ending in `.npz`, it becomes `.npz.npz`. | `src/tabular_blueprint/models/baselines.py:52` |
| Low | Export service embeds all model class paths in `allowlisted_model_classes` metadata. This leaks internal module structure to anyone with access to the export package. | `src/tabular_blueprint/services/export_service.py:208` |

## Performance Issues

| Issue | Location |
|-------|----------|
| `discover_interactions` runs `cross_val_score` for every candidate pair × operation, making it O(top_k² × n_ops × cv_folds) model fits. With `afe_top_k=10`, this is ~90 CV evaluations — very slow on large datasets. | `src/tabular_blueprint/data/feature_engine.py:164-254` |
| `detect_leakage` runs one `cross_val_score` per feature (n_features × cv_folds fits). No parallelism. | `src/tabular_blueprint/data/leakage.py:56-84` |
| `DomainClassifierDriftDetector` runs `cross_val_score` with `LogisticRegression` synchronously. No GPU or parallelism support. | `src/tabular_blueprint/monitoring/domain_classifier.py:62-69` |
| `quality.py` runs `cross_val_predict` + `find_label_issues` on the full dataset in memory. No chunking for large datasets. | `src/tabular_blueprint/data/quality.py:42-46` |
| `EmbeddingEngine` always trains on CPU (`device = torch.device("cpu")`), ignoring available GPU. | `src/tabular_blueprint/data/embedding_engine.py:228,283` |
| `PreprocessingCache` has no size-based eviction. `clear()` is the only way to free space. | `src/tabular_blueprint/data/cache.py:67-74` |
| `MLflowTracker.log_event` iterates all dict keys and calls `mlflow.log_param` per key — O(n) individual API calls per event. | `src/tabular_blueprint/engine/tracker.py:145-151` |
| `JSONLTracker.log_event` opens and closes the file on every single event. No buffering. | `src/tabular_blueprint/engine/tracker.py:74-83` |

## Maintenance Issues

| Issue | Detail |
|-------|--------|
| Inconsistent model base classes | `CatBoostModel` (`src/tabular_blueprint/models/conventional/catboost_model.py:10`) does not use `BaseGBDTModel`, while `LightGBMModel` and `XGBoostModel` do. Duplication of `apply_overrides`, `save`, and `predict_proba` logic. |
| 14 bare `except Exception` clauses | Spread across 10 files. Most silently swallow errors with no logging: `model_training.py:134`, `explainability.py:136`, `baselines.py:46`, `model_configs.py:29`, `domain_classifier.py:70`, `catboost_model.py:30`, `exceptions.py:60`, `mcp/tools.py:172`, `llm/__init__.py:100`, `hpo.py:221,253`, `hpo_warmstart.py:156,207`, `feature_engine.py:220` |
| DriftReport name collision | `DriftReport` is defined as both a Pydantic model (`monitoring/drift.py:17`) and a dataclass (`pipelines/nodes/drift_detection.py:22`). Same name, different types. |
| `Tracker` Protocol not fully implemented | `WandbTracker` and `MLflowTracker` lack `current_run_id` attribute required by the `Tracker` Protocol at `engine/tracker.py:10`. |
| Hamilton conditional import with MagicMock fallback | `feature_engineering.py:14-17` and `drift_detection.py:15-17` use `unittest.mock.MagicMock` as a stand-in for Hamilton's `@config` decorator when Hamilton is not installed. This silently produces broken DAG nodes instead of raising an error. |
| No type checking for `data_prep_result` | Pipeline nodes (`model_selection.py`, `model_training.py`, `state_generation.py`, `baselines.py`) accept `data_prep_result: Any` instead of `DataPrepResult`, losing type safety across the DAG. |
| Dead code in `_fit_importance_model` | The `or cls(task=task)` fallback at `feature_engineering.py:36` always executes because `fit()` returns `None`, creating an unfitted model that gets used for `permutation_importance`. |
| Export template uses f-string `{{` escaping | `export_service.py:67` uses `{{` in the template for Python dict literal, which is correct but fragile — any changes to the template risk breaking the double-brace escaping. |
| CatBoost `classes_count` parameter | `catboost_model.py:39-40` sets `classes_count` for multiclass but this parameter was renamed to `class_count` in newer CatBoost versions, potentially causing runtime errors. |
