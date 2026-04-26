# Concerns

> Last updated: 2025-04-23

## Security

### HIGH: Unsafe `pickle.load()` on untrusted files
Multiple models deserialize with `pickle.load()` without integrity checks. Loading a tampered `.artifact` file executes arbitrary code.
- `src/tabular_blueprint/models/baselines.py:105` — `pickle.load(f)` on arbitrary file path
- `src/tabular_blueprint/models/tabular_foundation/tabpfn_model.py:93` — `pickle.load(f)`
- `src/tabular_blueprint/engine/calibration.py:93` — `pickle.load(f)`
- `src/tabular_blueprint/data/loaders.py:56` — `np.load(path + ".npz", allow_pickle=True)` explicitly enables pickle in NumPy

### HIGH: `pickle` imported in generated export template
- `src/tabular_blueprint/services/export_service.py:20` — The `PREDICTOR_TEMPLATE` string includes `import pickle` in generated predictor scripts. Consumers of exported models will inherit this insecure deserialization path.

### MEDIUM: `fcntl` file locking is Unix-only
- `src/tabular_blueprint/services/registry_service.py:3,48,70` — Uses `fcntl.flock()` which is Linux/macOS only. Will crash on Windows with `ModuleNotFoundError`. The class docstring claims "thread-safe" but the locking only works on Unix.

### LOW: SQL injection validation in `load_sqlite` is minimal
- `src/tabular_blueprint/data/loaders.py:64-83` — Only checks that query starts with "SELECT" and lacks semicolons. Does not prevent subqueries like `SELECT * FROM users; DROP TABLE users--` (via comment injection). The code itself acknowledges this: "This is a basic validation."

### LOW: LLM API key handling
- `src/tabular_blueprint/llm/__init__.py:77` — API key fetched from env var with empty string fallback. If `api_key_env` is set but the env var doesn't exist, returns `"[LLM disabled: API key not configured]"` rather than raising. Silent failure mode may mask misconfiguration.

## Type Safety

### 34 uses of `Any` across the codebase
Heavy reliance on `typing.Any` weakens type checking. Key locations:
- `src/tabular_blueprint/engine/state_observer.py:4` — `from typing import Any`; used in 5+ method signatures
- `src/tabular_blueprint/engine/trainer.py:7` — `from typing import Any, ClassVar`; many `Any` params
- `src/tabular_blueprint/engine/hpo.py:5` — `from typing import Any`; function params like `model_cls: Any`
- `src/tabular_blueprint/engine/evaluator.py:1` — `from typing import Any`; `model_cls: Any` in evaluate()
- `src/tabular_blueprint/models/base.py:9` — Protocol uses `**kwargs: object` not `**kwargs: Any` but return types are untyped
- `src/tabular_blueprint/models/deep/text_encoder.py:24-25` — `self.tokenizer: Any = None`, `self.model: Any = None`
- `src/tabular_blueprint/engine/calibration.py:27` — `self._calibrated: Any = None`
- `src/tabular_blueprint/services/export_service.py:6` — `from typing import Any`

### Mixed `dataclass` and `Pydantic` models
- `src/tabular_blueprint/data/leakage.py:10` — Uses `@dataclass(frozen=True)` for `LeakageReport` while most other models use Pydantic `BaseModel`. Inconsistent; dataclasses lack validation.
- `src/tabular_blueprint/engine/hpo_warmstart.py:17` — `@dataclass(frozen=True)` for `WarmstartInjection`
- `src/tabular_blueprint/engine/hpo_importance.py:10-18` — `@dataclass(frozen=True)` for `ParamImportance`, `ImportanceReport`
- `src/tabular_blueprint/services/registry_service.py:14` — `@dataclass(frozen=True)` for `PromotionResult`
- `src/tabular_blueprint/services/report_service.py:13-31` — `@dataclass(frozen=True)` for `LeaderboardEntry`, `ExperimentReport` (large dataclass with 12 fields, some `Any` typed)

## Performance

### HIGH: O(n_features * cv_folds) leakage detection is expensive
- `src/tabular_blueprint/data/leakage.py:58-63` — For each feature, copies the entire matrix, shuffles one column, and runs `cross_val_score()`. With 100 features and 3 folds, this is 300 full CV runs. No parallelism, no caching.

### MEDIUM: AFE interaction discovery has quadratic complexity
- `src/tabular_blueprint/data/feature_engine.py:186-233` — `discover_interactions()` iterates all pairs of top-k features, running `cross_val_score()` for each pair×operation. With `top_k=10`, that's 45 pairs × 2 ops = 90 CV runs. With `top_k=50`, it's 1225 × 2 = 2450 CV runs.

### MEDIUM: `_fit_quick_gbdt` called twice during AFE with pruning
- `src/tabular_blueprint/engine/trainer.py:204,244` — When `afe_pruning=True`, a GBDT is fit at line 204 for importance, then again at line 244 on the augmented feature set. The second fit could reuse the first model.

### MEDIUM: SHAP KernelExplainer can be very slow
- `src/tabular_blueprint/monitoring/explainability.py:91` — Falls back to `KernelExplainer` for non-tree models, which is O(n_samples²) and prohibitively slow on large datasets. The background sample is capped at 100 which helps but may not be enough for large data.

### LOW: `DataAdapter._to_numpy()` does not handle categorical/string columns
- `src/tabular_blueprint/data/adapter.py:37-41` — Calls `df.to_numpy()` directly without encoding categoricals. Will fail or produce garbage if string columns remain after preprocessing.

### LOW: `load_events()` reads entire JSONL into memory
- `src/tabular_blueprint/utils/jsonl.py:7` — Returns the full list of events. With large experiment logs (100MB+ before rotation), this could be problematic. The `StateObserver` loads all events multiple times (`_render_state` calls `_load_all_events()`).

### LOW: `get_data_hash()` uses sum instead of XOR
- `src/tabular_blueprint/data/loaders.py:98` — Comment says "Use XOR aggregation instead of sorting" but actually uses `.sum()`, not XOR. Sum is more collision-prone than XOR for hash combining.

## Bugs & Edge Cases

### HIGH: Model evaluated twice in `_train_single_model`
- `src/tabular_blueprint/engine/trainer.py:505,533` — `evaluator.evaluate()` at line 505 trains a model per fold for CV scoring, then at line 533 `model.fit(X, y)` trains again on full data. This means each model is trained `cv_folds + 1` times. The CV model and the final model are different objects. The final model's artifact is saved but CV scores come from the CV-only models. This is standard practice but the double training cost should be noted.

### MEDIUM: `ExperimentConfig.models` type mismatch
- `src/tabular_blueprint/config.py:23` — `models: list[str] | Literal["auto"] = "auto"` but the CLI at `src/tabular_blueprint/cli.py:77` sets `experiment_config.models = models` where `models` is `list[str] | None`. If `models` is `None`, it sets `models = None` on the config, bypassing Pydantic validation.

### MEDIUM: `Evaluator.evaluate()` creates model with `task` keyword but some models expect different kwargs
- `src/tabular_blueprint/engine/evaluator.py:98` — `model = model_cls(task=model_task, **model_kwargs)` but `FTTransformerModel.__init__` requires `n_features` and `n_classes`. These aren't passed during evaluation, so evaluating `ft_transformer` through `Evaluator` will fail. The `Trainer` works around this at line 507-513 by creating the model separately, but the evaluator path is broken.

### MEDIUM: `state_observer.py` uses `assert latest is not None`
- `src/tabular_blueprint/engine/state_observer.py:60` — `assert latest is not None` right after checking `if not report.latest_run` (line 43) returns early. The assert is fine in normal flow but using `assert` for runtime checks is bad practice — it can be disabled with `python -O`.

### MEDIUM: `preprocessing.py` categorical columns only detects `pl.Categorical` dtype
- `src/tabular_blueprint/pipelines/preprocessing.py:16` — `cs.categorical()` only matches `pl.Categorical` type. String columns (`pl.Utf8`) are not detected as categorical, so `fill_nulls_categorical` and `encoded_df` skip them entirely. The `export_service.py` template handles both `pl.Utf8` and `pl.Categorical`.

### LOW: `StateObserver` creates `ReportService` twice per `generate()` call
- `src/tabular_blueprint/engine/state_observer.py:38-41,200-203` — `ReportService` is instantiated and `build_report()` called at line 41, then `_render_leaderboard()` creates another `ReportService` and calls `format_leaderboard_markdown()` which calls `build_report()` again. Three full JSONL parses per `generate()`.

### LOW: Drift detection in `trainer.py` uses 80/20 split of the same dataset
- `src/tabular_blueprint/engine/trainer.py:289-292` — `_run_drift_check()` splits the training data 80/20 as "reference" and "live". This is comparing the first 80% of rows vs last 20% of the same dataset, not actual production drift. It will produce misleading results if data is sorted or has temporal patterns.

### LOW: `quality.py` drops rows by index using a set membership test
- `src/tabular_blueprint/data/quality.py:89` — Creates a boolean mask via `[i in set(flagged_indices) for i in range(len(df))]`. This is O(n*m) where m is the flagged count. Should use `np.isin` or Polars `is_in()` for O(n) performance.

### LOW: `LeakageReport.score_drop` direction is unconditionally `baseline - permuted`
- `src/tabular_blueprint/data/leakage.py:65` — `drop = baseline_score - permuted_score`. For metrics where lower is better (e.g. RMSE), a permuted score *higher* than baseline is actually the expected direction, and the drop would be negative, potentially missing leaky features.

## Platform Compatibility

### `fcntl` module is Unix-only
- `src/tabular_blueprint/services/registry_service.py:3` — Will crash on Windows. No fallback or conditional import.

### `OMP_NUM_THREADS` set globally via `os.environ`
- `src/tabular_blueprint/config.py:111-113` — `HardwareProfile.configure_omp_threads()` is called at module-level in `trainer.py:14`. This mutates global state and may interfere with parent processes or other libraries.

### MPS fallback hardcoded in test
- `tests/unit/test_ft_transformer.py:6` — `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` set at module level, which modifies global state for all tests in that file.

## Missing Tests / Coverage Gaps

### No tests found for:
- `src/tabular_blueprint/monitoring/explainability.py` — SHAP wrapper is untested
- `src/tabular_blueprint/pipelines/hamilton_executor.py` — Hamilton DAG executor untested
- `src/tabular_blueprint/pipelines/preprocessing.py` — Preprocessing functions untested in isolation
- `src/tabular_blueprint/data/adapter.py` — `_to_tensor()` and `_to_dataset()` conversion paths untested
- `src/tabular_blueprint/models/conventional/xgboost_model.py` — XGBoost model wrapper (integration tests cover it but no unit test)
- `src/tabular_blueprint/models/deep/tabnet_model.py` — TabNet model wrapper untested
- `src/tabular_blueprint/models/deep/text_encoder.py` — TextEncoder completely untested
- `src/tabular_blueprint/engine/calibration.py` — Only one test file exists but edge cases (no `predict_proba`, regression task) may be uncovered
- `src/tabular_blueprint/mcp/tools.py` — MCP tools untested (test_mcp_tools.py exists but likely mocked)
- `src/tabular_blueprint/utils/jsonl.py` — Error path for malformed JSON may be undertested
- `src/tabular_blueprint/services/report_service.py` — Edge cases like empty events, missing scores

### Empty `tests/e2e/` directory
- `tests/e2e/.gitkeep` — No end-to-end tests exist despite the marker being defined in `pyproject.toml`.

## Complexity

### `trainer.py` — 647 lines, high cyclomatic complexity
- `src/tabular_blueprint/engine/trainer.py` — Single class with 14 methods, mixing orchestration, training, drift detection, SHAP explainability, state observation, and registry management. The `run()` method alone is 133 lines with deeply nested conditionals. Should be decomposed.

### `cli.py` — 386 lines, growing command surface
- `src/tabular_blueprint/cli.py` — 10 CLI commands in one file. The `diff` command contains a nested function `_extract_run_summary()` with complex dict manipulation.

### `state_observer.py` — 246 lines, string concatenation-heavy
- `src/tabular_blueprint/engine/state_observer.py` — Builds markdown via list-of-strings concatenation across 5 event-type branches. Fragile and hard to test. Lacks template engine or structured rendering.

## Dependencies

### Heavy core dependencies
- `pyproject.toml:13-34` — All model libraries (catboost, lightgbm, xgboost, tabpfn, torch, transformers, cleanlab, shap) are in core dependencies. This means even basic CLI usage requires installing PyTorch (~2GB) and all ML libraries. They should be optional extras.

### Duplicated optional dependency entries
- `pyproject.toml:36-44` — `shap`, `hamilton`, and `transformers` are listed in both core `dependencies` and `[project.optional-dependencies]`. The optional entries are redundant since they're already required.

### `pydantic-settings` imported but unused in core
- `pyproject.toml:16` — `pydantic-settings>=2.0` is a dependency but no file imports from `pydantic_settings`. It may be used for future config loading from env/files but is currently dead weight.

### `skrub` imported but not found in source
- `pyproject.toml:21` — `skrub>=0.3` is a dependency but no source file imports `skrub`. Dead dependency.

### `accelerate` as core dependency
- `pyproject.toml:25` — `accelerate>=0.30` is required even when only running CPU models. Should be an optional `[dl]` extra.

## Documentation

### `ARCHITECTURE.md` does not exist
- No architecture documentation found. Per `AGENTS.md`, this should exist and be updated when structure changes.

### Export service template has dead code
- `src/tabular_blueprint/services/export_service.py:36-42` — The `PREDICTOR_TEMPLATE` `__init__` imports `fill_nulls_numeric`, etc., from a local `pipelines` directory, but `_preprocess()` at line 63-73 imports them again from `tabular_blueprint.pipelines.preprocessing`. The local import at line 35 is dead code.

### `state_observer.py` renders fields that may not exist
- `src/tabular_blueprint/engine/state_observer.py:107` — `latest_tt.get("original_skewness", "N/A"):.4f` — If the value is `"N/A"` (a string), `.4f` formatting will raise `TypeError`. Only triggered when target transform event exists but keys are missing.

### `ModelName` enum in `constants.py` includes `NODE` but factory doesn't
- `src/tabular_blueprint/constants.py:30` — `NODE = "node"` exists but `src/tabular_blueprint/models/factory.py:5-14` has no entry for `node`. Selecting "node" via the enum would pass validation but fail at model lookup.

## Thread Safety

### `JSONLTracker` uses `threading.Lock` but not process-safe
- `src/tabular_blueprint/engine/tracker.py:37,75` — Uses `threading.Lock()` for file writes. When `Trainer._train_concurrent()` uses `ThreadPoolExecutor` (line 446), multiple threads share the same tracker. The lock protects the JSONL file but the `_should_rotate()` + `_rotate_log()` sequence is not atomic — a race could cause log loss during rotation.

### `RegistryService` uses file locking but `load()` doesn't
- `src/tabular_blueprint/services/registry_service.py:30-35` — `load()` reads the registry file without acquiring the lock. If `update_if_better()` is called concurrently, `load()` may read partially-written data.
