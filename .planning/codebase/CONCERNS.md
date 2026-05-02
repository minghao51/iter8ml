# Codebase Concerns & Tech Debt

## Tech Debt Areas

### 1. Large Files Needing Refactoring

| File | Lines | Issue |
|------|-------|-------|
| `src/tabular_blueprint/cli.py` | 477 | The `drift` command has 5+ duplicate code paths for PSI/KS/Domain methods; the `diff` command embeds a rich table builder inline; the `hpo` command has extensive result-formatting logic |
| `src/tabular_blueprint/engine/model_trainer.py` | 367 | `_train_sequential` and `_train_concurrent` share ~80% duplicated champion-update logic (lines 144-155 vs 200-213) |
| `src/tabular_blueprint/engine/trainer.py` | 334 | `__init__` accepts 5+ configuration flags for `DataPreparationService`/`FeatureEngineer`/`EmbeddingEngine`/`DriftChecker`/`ExplainabilityService` — these could be composed externally |
| `src/tabular_blueprint/data/feature_engine.py` | 318 | `discover_interactions` (line 156) has O(n²) complexity over `top_k_indices`; the function mixes candidate generation, evaluation, and result formatting |
| `tests/unit/test_cli.py` | 342 | CLI tests are integration-level (mock file I/O, real data loading) — slow and fragile |

### 2. TODOs / FIXMEs / NOTES

Only one found:
- **NOTE**: `src/tabular_blueprint/services/registry_service.py:192` — Caller MUST hold file lock before calling `update_if_better`. This is an unchecked precondition; a future refactor should make lock acquisition internal.

### 3. Duplicated Logic

- **`model_trainer.py:109-157` vs `159-219`**: `_train_sequential` and `_train_concurrent` duplicate the champion-registry update loop (`best_score`, `primary_metric`, `metric_value_is_better`, `_update_champion`). Extract into a shared helper.
- **`cli.py:236-250` vs `196-206`**: PSI drift report printing is duplicated across `method="both"` and `method="psi"` paths.
- **`hpo_warmstart.py:143-155` vs `194-205`**: Identical trial-injection try/except blocks. Extract into `_inject_trial(study, params, score) -> bool`.

### 4. Hamilton Config Branching Conflict

`src/tabular_blueprint/pipelines/nodes/feature_engineering.py:157-201`:
- `training_features__default` fires when `afe_enabled != True`
- `training_features__afe_enabled` fires when `afe_enabled == True`
- `training_features__embedding_enabled` fires when `embedding_enabled == True`

If a config sets `afe_enabled=False, embedding_enabled=True`, both `__default` and `__embedding_enabled` match, causing a Hamilton runtime resolution error. The two features are effectively mutually exclusive but nothing enforces this.

### 5. Deprecated `HamiltonExecutor`

`src/tabular_blueprint/pipelines/hamilton_executor.py:13` — Emits `DeprecationWarning` but is still importable and used. Should be removed in next major version. No test coverage.

### 6. Thread-Based Concurrent Training

`src/tabular_blueprint/engine/model_trainer.py:177`: Uses `ThreadPoolExecutor` for CPU-bound model training (GBDTs, neural nets). The GIL prevents true parallelism. Should use `ProcessPoolExecutor` or `joblib.Parallel` for CPU-bound model fitting.

---

## Security Considerations

### 1. Safe Unpickler — Adequate but Permissive

`src/tabular_blueprint/utils/safe_pickle.py:36-48` — `RestrictedUnpickler` uses **prefix-based** whitelisting (e.g., `"sklearn."` allows any class under the sklearn package). This is reasonable for an ML library but is broader than class-level allowlisting. Consider:
- Adding subprocess-level classes (e.g., `sklearn.externals.joblib`) to watch for supply-chain attacks
- Adding a deny-list override for known-dangerous modules like `os`, `subprocess`, `builtins.exec`

### 2. Dynamic Import in Factory — Safe by Registry

`src/tabular_blueprint/models/factory.py:39` — `importlib.import_module` is safe because it resolves from a hardcoded `_MODEL_REGISTRY` dict. This is a good pattern.

### 3. Dynamic Import in Export Service — Allowlisted

`src/tabular_blueprint/services/export_service.py:62` — Uses class-level allowlisting (line 51-56) from metadata. Good pattern.

### 4. Python Config Loading — Gated

`src/tabular_blueprint/config.py:105-119` — `.py` config loading requires `--allow-unsafe-config` flag. Appropriate.

### 5. No Secrets in Source Code

All API keys come from environment variables (`llm/__init__.py:77`). No hardcoded secrets found. Good.

### 6. No `shell=True`, `subprocess`, `eval`, `exec`

None found. Clean.

---

## Error Handling Issues

### 1. Bare/Overly Broad `except Exception` (11 instances)

| File | Line | Risk |
|------|------|------|
| `services/registry_service.py` | 226 | `except BaseException` on atomic write — this is intentional (cleanup on any failure), but should narrow to `OSError` |
| `engine/hpo_warmstart.py` | 153, 204 | `except Exception: continue` — silently swallows all errors during trial injection. A malformed trial fails silently |
| `data/feature_engine.py` | 212 | `except Exception: continue` — silently skips interaction candidates that fail cross-validation |
| `monitoring/domain_classifier.py` | 68 | `except Exception` — silently falls back to AUC=0.5 on any failure |
| `monitoring/explainability.py` | 132 | `except Exception: pass` — silently skips SHAP dependence plots |
| `pipelines/nodes/baselines.py` | 46 | `except Exception: continue` — silent failure on baseline evaluation |
| `llm/__init__.py` | 91 | `except Exception` — returns error string, which is acceptable for optional LLM feature |
| `engine/model_trainer.py` | 216, 322 | `except Exception` — catches all errors; 322 re-raises as `ModelFitError` (better but broad) |
| `engine/hpo_importance.py` | — | (not checked for this pattern specifically) |

### 2. Optional Import Silent Fallbacks (12+ files)

Many modules use `try: ... except ImportError:` to handle missing optional deps (torch, hamilton, cleanlab, shap, etc.). This is the intended pattern for optional dependencies, but it means:
- Failures are deferred to runtime
- No warning is emitted when an optional dep is missing and a feature is requested
- Example: `config.py:94` catches `ModuleNotFoundError` (should also catch `ImportError` for robustness)

---

## Mypy Gaps

`pyproject.toml:91-101` — Several modules have `ignore_errors = true`:

| Module | Lines |
|--------|-------|
| `tabular_blueprint.engine.hpo` | 262 |
| `tabular_blueprint.engine.hpo_warmstart` | 207 |
| `tabular_blueprint.engine.hpo_importance` | 149 |
| `tabular_blueprint.pipelines.nodes.*` | ~500+ |
| `tabular_blueprint.engine.trainer` | 334 |

~1,450 lines of type-checked source are excluded from mypy verification.

---

## Test Coverage Gaps

### Source modules without dedicated test files:
| Source Module | Notes |
|---------------|-------|
| `data/loaders.py` | Has `test_loaders.py` ✅ |
| `data/leakage.py` | Has `test_leakage.py` ✅ |
| `data/cache.py` | No direct test file |
| `data/embedding_engine.py` | No direct test file (tested indirectly via `embedding_trainer`) |
| `engine/drift_checker.py` | No direct test file |
| `engine/feature_engineer.py` | No direct test file |
| `engine/data_preparation.py` | Tested via `test_data_prep_nodes.py` ✅ |
| `engine/explainability_service.py` | No direct test file |
| `engine/tracker.py` | Has `test_tracker_rotation.py` ✅ |
| `services/report_service.py` | Has `test_report_service.py` ✅ |
| `services/__init__.py`, `pipelines/__init__.py`, etc. | Init files, minimal logic |
| `pipelines/hooks/tracking_hook.py` | 61 lines, no tests |
| `pipelines/hamilton_executor.py` | 23 lines, deprecated — no tests |

### Thin test coverage (<50 lines):

| Test File | Lines |
|-----------|-------|
| `tests/unit/test_safe_pickle.py` | 19 |
| `tests/unit/test_model_factory.py` | 25 |
| `tests/unit/test_tabpfn_guardrails.py` | 26 |
| `tests/unit/test_jsonl.py` | 33 |
| `tests/unit/test_adapter.py` | 42 |
| `tests/unit/test_tabpfn.py` | 45 |
| `tests/unit/test_domain_classifier.py` | 46 |

---

## Performance Concerns

1. **ThreadPoolExecutor for CPU-bound work** (`model_trainer.py:177`): GIL serializes Python threads. GBDT fitting releases the GIL in native code (LightGBM/XGBoost), but PyTorch/transformers work does not. `ProcessPoolExecutor` or `joblib.Parallel` would be better for mixed workloads.

2. **O(n²) feature interaction search** (`feature_engine.py:186-197`): `discover_interactions` evaluates all pairs of top-k features with cross-validation. For k=10, that's 45 candidates × 2 operations × 3-fold CV. For k=50, it's 2,450 candidates — each running 3-fold CV.

3. **Full dataset in-memory** (`data/quality.py:36-43`): `audit_data_quality` converts the full Polars DataFrame to NumPy and runs cross-validated LogisticRegression. For datasets >100k rows, this can be OOM.

4. **Large JSONL loading** (`utils/jsonl.py:9-34`, `engine/hpo_warmstart.py`): `load_events` loads the entire JSONL file into memory. `iter_events` (line 37) is available but not used in warmstart paths. Warmstart could OOM on large experiment logs.

---

## Dependency Risks

1. **Optional dep fragility**: 7+ optional dependency groups (`deep`, `shap`, `cleanlab`, `llm`, `wandb`, `mlflow`, `all`) with 12+ `except ImportError` guards. If a dep is installed but broken, the silent fallback gives no warning.

2. **TabPFN row limit**: `trainer.py:208` hardcodes a row limit warning but doesn't prevent usage. Users may hit OOM with no guardrail.

3. **Torch dependency by version only**: `pyproject.toml:32` pins `torch>=2.3` — no CUDA version constraint. Users on CPU-only or mismatched CUDA will get runtime errors.

---

## Documentation Gaps

1. `ARCHITECTURE.md` — needs update if structure changed (per AGENTS.md instructions)
2. Several pipeline nodes lack module-level docstrings (e.g., `pipelines/nodes/feature_engineering.py`)
3. The Hamilton config branching rules (`@config.when` / `@config.when_not`) are implicit and undocumented — especially the `afe_enabled` ↔ `embedding_enabled` conflict
4. No CONTRIBUTING.md or development setup guide beyond AGENTS.md
