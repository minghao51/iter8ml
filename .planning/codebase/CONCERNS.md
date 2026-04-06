# CONCERNS — Technical Debt & Code Quality Analysis

**Project:** iter8ml (tabular-blueprint)
**Date:** 2026-04-06
**Scope:** All source Python files (excluding `.venv/`, `notebooks/`)
**Total source LOC:** ~3,431 across 50 files

---

## 1. BUGS / LOGIC ISSUES

### 1.1 Dead Code — unreachable duplicate blocks in TabPFNModel._build_model
- **File:** `core/models/tabular_foundation/tabpfn_model.py:37-50`
- **Severity:** CRITICAL
- **Issue:** Lines 37–50 are unreachable dead code. The function returns at line 36 (or line 33), making lines 37–50 completely dead. The same if/else block is duplicated twice — once at lines 28-36 and again at lines 42-50, with an orphaned `return TabPFNRegressor(...)` at line 37. This appears to be a botched merge or refactor.
- **Impact:** Confusing to readers; if the first return block is ever removed, the second block would activate with potentially different behavior.

### 1.2 CatBoostModel._build_model() mutates caller's params dict
- **File:** `core/models/conventional/catboost_model.py:22-23`
- **Severity:** HIGH
- **Issue:** `self.params.pop("random_seed", 42)` mutates the caller's dict in-place. On second call to `_build_model()`, `random_seed` is no longer in params and defaults to 42 silently. Line 23 then filters out `random_seed` from the already-mutated dict, making the `pop()` pointless on subsequent calls.
- **Same pattern in:** `core/models/conventional/lightgbm_model.py:20`, `core/models/conventional/xgboost_model.py:20`
- **Impact:** Non-deterministic behavior on re-fit; HPO trials lose seed consistency.

### 1.3 Trainer.run() trains every model twice
- **File:** `core/engine/trainer.py:191-197`
- **Severity:** HIGH
- **Issue:** `evaluator.evaluate()` at line 191 already trains models internally (fresh instance per fold). Then lines 194-197 create a **new** model instance and train again on the full dataset just to save artifacts. This doubles training time for every model.
- **Impact:** 2x training time, wasted compute.

### 1.4 HPO objective silently prunes all exceptions
- **File:** `core/engine/hpo.py:61-62`
- **Severity:** MEDIUM
- **Issue:** `except Exception: raise optuna.TrialPruned() from None` — the `from None` suppresses the original traceback, making debugging failed trials impossible. All non-Optuna errors are swallowed. Failed trials are indistinguishable from legitimately pruned ones.

### 1.5 Evaluator uses self.task for metrics but model_task for model construction
- **File:** `core/engine/evaluator.py:89`
- **Severity:** MEDIUM
- **Issue:** `METRICS_REGISTRY[self.task]` uses `self.task` instead of `model_task` (line 78). If a task override is passed to `evaluate()`, metrics lookup still uses the original task, potentially causing KeyError for mismatched task/metric combinations.

### 1.6 LightGBM/XGBoost binary classification threshold hardcoded at 0.5
- **Files:** `core/models/conventional/lightgbm_model.py:33`, `core/models/conventional/xgboost_model.py:35`
- **Severity:** LOW
- **Issue:** Binary classification uses a fixed 0.5 threshold. For imbalanced datasets this may produce poor results.

### 1.7 run_id collision possible with time-based generation
- **File:** `core/engine/trainer.py:140`
- **Severity:** LOW
- **Issue:** `run_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"` — if two runs start within the same second, there is a small but non-zero probability of collision with only 6 hex chars from UUID (~16M space).

---

## 2. SECURITY CONCERNS

### 2.1 Insecure pickle deserialization
- **File:** `core/models/tabular_foundation/tabpfn_model.py:80`
- **Severity:** HIGH
- **Issue:** `pickle.load(f)` can execute arbitrary code. If artifact files are sourced from untrusted locations (e.g., shared workspace), this is a remote code execution vector.
- **Recommendation:** Use `torch.save`/`torch.load(weights_only=True)` or a safe serialization format.

### 2.2 Arbitrary module execution via CLI config loading
- **File:** `main.py:43-45`
- **Severity:** MEDIUM
- **Issue:** `spec.loader.exec_module(module)` executes arbitrary Python code from a user-supplied config file path. While intentional for config loading, there is no validation, sandboxing, or warning about security implications.
- **Recommendation:** Add documentation warning, or restrict config loading to a known directory.

### 2.3 No input validation on MCP tool file paths
- **File:** `mcp_server/tools.py:25-31, 76-77, 178-179`
- **Severity:** LOW
- **Issue:** File paths passed to MCP tools (`data_path`, `reference_path`, `new_path`) are used directly without path traversal checks. An LLM agent could be tricked into reading arbitrary files.

### 2.4 SQLite connection not using context manager
- **File:** `core/data/loaders.py:34-36`
- **Severity:** LOW
- **Issue:** If `pl.read_database` raises, `conn.close()` is never called, leaking the connection.

---

## 3. PERFORMANCE ISSUES

### 3.1 Duplicate model training
- **File:** `core/engine/trainer.py:191-197`
- **Severity:** HIGH
- **Issue:** As noted in 1.3, every model is trained twice — once during CV evaluation and once for artifact saving.

### 3.2 No caching for JSONL file reads
- **Files:** `core/engine/state_observer.py:73-80`, `mcp_server/tools.py:108-114`, `core/engine/trainer.py:88-93`, `main.py:98-101`
- **Severity:** MEDIUM
- **Issue:** The JSONL events file is read and parsed from disk on every call. As the file grows, this becomes increasingly slow. No caching, rotation, or incremental read strategy exists. Unbounded file growth over time.

### 3.3 Repeated JSONL parsing in multiple locations
- **Files:** `core/engine/trainer.py:88-93`, `core/engine/state_observer.py:73-80`, `mcp_server/tools.py:108-112, 141-145`, `main.py:98-101`
- **Severity:** LOW
- **Issue:** The same pattern of reading and parsing JSONL lines appears in 5+ locations with nearly identical code. No shared utility function.

### 3.4 _MODEL_CLASS_CACHE is an unbounded module-level mutable global
- **File:** `core/engine/trainer.py:30`
- **Severity:** LOW
- **Issue:** `_MODEL_CLASS_CACHE: dict = {}` is never cleared. In long-running processes (e.g., MCP server), this holds references indefinitely. Also lacks type parameters.

---

## 4. CODE QUALITY CONCERNS

### 4.1 Duplicated JSONL parsing logic
- **Files:**
  - `core/engine/trainer.py:88-93`
  - `core/engine/state_observer.py:73-80`
  - `mcp_server/tools.py:108-114`
  - `mcp_server/tools.py:141-145`
  - `main.py:97-101`
- **Severity:** MEDIUM
- **Issue:** The pattern `for line in f: if line.strip(): events.append(json.loads(line))` is duplicated in 5+ locations. Should be a shared utility function like `load_jsonl_events(path)`.

### 4.2 Duplicated data loading logic
- **Files:**
  - `mcp_server/tools.py:44-45`
  - `mcp_server/tools.py:76-77`
  - `main.py:66-73`
  - `main.py:217`
- **Severity:** MEDIUM
- **Issue:** The `if path.suffix == ".parquet" else load_csv(path)` pattern is repeated 4 times. Should be a single `load_data(path)` utility.

### 4.3 Duplicated HPO orchestration
- **Files:**
  - `main.py:210-241` (hpo command)
  - `mcp_server/tools.py:60-98` (run_hpo tool)
- **Severity:** MEDIUM
- **Issue:** The HPO workflow (load data → adapter → evaluator → search_space → optimize_model) is nearly identical in both files (~30 lines each). Should be extracted to a shared service function.

### 4.4 Registry promotion lacks file locking
- **File:** `mcp_server/tools.py:129-168`
- **Severity:** MEDIUM
- **Issue:** `registry_promote` reads and writes `registry.json` without the `fcntl.flock` used in `trainer.py:63-83`. Concurrent MCP tool calls could corrupt the registry.

### 4.5 AbstractModel Protocol is unused
- **File:** `core/models/base.py:8-16`
- **Severity:** LOW
- **Issue:** The `AbstractModel` Protocol is defined but never imported or used as a type annotation anywhere. All model wrappers implement the interface implicitly but are not type-checked against it.

### 4.6 TextEncoder is not integrated into any pipeline
- **File:** `core/models/deep/text_encoder.py`
- **Severity:** LOW
- **Issue:** The `TextEncoder` class exists but is not referenced by the trainer, adapter, or any model. It is dead code until integrated.

### 4.7 Inconsistent error handling in MCP tools
- **File:** `mcp_server/tools.py`
- **Severity:** MEDIUM
- **Issue:** MCP tools return error strings (e.g., `"Unsupported format: {path.suffix}"` at line 31) rather than raising exceptions. This makes it hard for calling code to distinguish errors from valid output.

### 4.8 Hardcoded workspace path in multiple locations
- **Files:** `main.py:20`, `core/engine/trainer.py:118`, `mcp_server/tools.py:104`, `core/engine/tracker.py:20`, `core/engine/state_observer.py:12-14`
- **Severity:** LOW
- **Issue:** `"workspace"` is hardcoded in multiple places. The `WORKSPACE_DIR` constant in `trainer.py:118` is only used by `Trainer`, not by CLI commands or MCP tools.

---

## 5. ERROR HANDLING GAPS

### 5.1 Trainer.run catches Exception but continues silently
- **File:** `core/engine/trainer.py:232-240`
- **Severity:** MEDIUM
- **Issue:** When a model fails, the exception is logged but the experiment continues. No retry logic, no circuit breaker, and `results[model_name] = {"error": str(e)}` makes it easy for callers to miss failures if they only check for numeric scores. No traceback is preserved.

### 5.2 No validation of model output shapes
- **File:** `core/engine/evaluator.py:86-95`
- **Severity:** MEDIUM
- **Issue:** `model.predict(X_val)` and `model.predict_proba(X_val)` are called without validating output shapes match `y_val`. A misbehaving model could produce silently wrong metric values.

### 5.3 HPO search space parsing is fragile
- **File:** `core/engine/hpo.py:39-51`
- **Severity:** LOW
- **Issue:** The search space parsing uses `len(param_range)` heuristics (2-tuple vs 3-tuple) with no validation. An incorrectly specified range will silently produce wrong behavior or raise opaque errors.

### 5.4 load_sqlite has no error handling
- **File:** `core/data/loaders.py:30-37`
- **Severity:** LOW
- **Issue:** No try/except around `sqlite3.connect` or `pl.read_database`. Invalid DB path or malformed SQL will raise unhandled exceptions.

### 5.5 No validation on model name in HPO CLI command
- **File:** `main.py:224`
- **Severity:** LOW
- **Issue:** `getattr(model_configs, model)` will raise `AttributeError` if an invalid model name is passed, with no user-friendly error message.

---

## 6. TYPE SAFETY

### 6.1 Untyped df parameter in Trainer.run()
- **File:** `core/engine/trainer.py:138`
- **Severity:** LOW
- **Issue:** `def run(self, df)` lacks a type annotation. Should be `df: pl.DataFrame`.

### 6.2 Missing type parameters on _MODEL_CLASS_CACHE
- **File:** `core/engine/trainer.py:30`
- **Severity:** LOW
- **Issue:** `_MODEL_CLASS_CACHE: dict = {}` should be `_MODEL_CLASS_CACHE: dict[str, type] = {}`.

---

## 7. TEST COVERAGE GAPS

### 7.1 No unit tests for XGBoostModel / LightGBMModel
- **Files:** `core/models/conventional/xgboost_model.py`, `core/models/conventional/lightgbm_model.py`
- **Issue:** These models are only tested via integration tests. No unit tests exist for save/load, predict_proba edge cases, or parameter mutation bugs.

### 7.2 No tests for DataAdapter._to_dataset()
- **File:** `core/data/adapter.py:52-66`
- **Issue:** The HuggingFace Dataset conversion path is untested.

### 7.3 No tests for TextEncoder
- **File:** `core/models/deep/text_encoder.py`
- **Issue:** The entire text encoder module has no test coverage.

### 7.4 No tests for WandbTracker / MLflowTracker
- **File:** `core/engine/tracker.py:45-107`
- **Issue:** Optional tracking integrations are untested.

### 7.5 No test for cleanlab ImportError path in quality.py
- **File:** `core/data/quality.py`
- **Issue:** No test for the `ImportError` path when cleanlab is not installed.

---

## 8. COMPLEXITY HOTSPOTS

| File | Lines | Concern |
|------|-------|---------|
| `core/engine/trainer.py` | 260 | Orchestrates everything; does too much (training, registry, leaderboard, state). `run()` method is 60+ lines with multiple responsibilities. |
| `main.py` | 245 | CLI with 7 commands, duplicated loading logic, business logic overlaps with MCP tools |
| `mcp_server/tools.py` | 187 | 8 tools with duplicated data loading and config setup |
| `core/engine/hpo.py` | 70 | Search space parsing logic is fragile (relies on tuple length) |

---

## 9. DESIGN CONCERNS

### 9.1 Single Responsibility Violation in Trainer
- **File:** `core/engine/trainer.py`
- **Issue:** `Trainer.run()` handles: experiment tracking, data hashing, model selection, data adaptation, CV evaluation, model training, artifact saving, registry updates, leaderboard generation, and state observation. Should be decomposed.
- **Recommendation:** Extract `_update_registry` and `_generate_leaderboard` to a separate `registry.py` module.

### 9.2 HPO Search Space Format is Ad Hoc
- **File:** `core/engine/hpo.py:38-51`
- **Issue:** Search space is a dict where values are lists of length 2 or 3, with the third element being a string hint (`"log"`). This is fragile and undocumented. Should use a typed dataclass or Optuna-native distribution objects.

### 9.3 Model Selector Hardcodes Model Names
- **File:** `core/models/selector.py:29-37`
- **Issue:** Model names are hardcoded strings that must match `_MODEL_REGISTRY` keys in `trainer.py`. No compile-time or runtime validation ensures consistency.

### 9.4 Drift Detector Has No Multiple Testing Correction
- **File:** `core/monitoring/drift.py:49-56`
- **Issue:** Each column is tested independently at `alpha=0.05`. With many columns, the family-wise error rate inflates. No Bonferroni or FDR correction is applied.

---

## 10. CLEAN ITEMS (No Issues Found)

### 10.1 No circular dependencies detected
- **Status:** CLEAN — Import graph is a clean DAG. `core/engine/` imports from `core/data/` and `core/models/`, but not vice versa.

### 10.2 No TODO/FIXME/HACK/XXX comments in project source
- **Status:** CLEAN — Zero technical debt markers found in project code (excluding `.venv/`).

### 10.3 No hardcoded secrets or credentials
- **Status:** CLEAN — No API keys, passwords, or tokens found in project source code.

### 10.4 No deprecated imports
- **Status:** CLEAN — All imports use current API patterns. `importlib` usage is appropriate for dynamic model loading.

---

## SUMMARY

| Category | Count | Highest Severity |
|---|---|---|
| Bugs / Logic Issues | 7 | CRITICAL (dead code in tabpfn_model.py) |
| Security | 4 | HIGH (pickle deserialization) |
| Performance | 4 | HIGH (duplicate training) |
| Code Quality | 8 | MEDIUM (duplicated logic) |
| Error Handling | 5 | MEDIUM (silent failure continuation) |
| Type Safety | 2 | LOW |
| Test Gaps | 5 | MEDIUM |
| Complexity Hotspots | 4 | — |
| Design Concerns | 4 | MEDIUM |

**Priority fixes:**
1. Remove dead code in `tabpfn_model.py:37-50` (CRITICAL)
2. Fix params mutation bug in CatBoostModel/LightGBMModel/XGBoostModel (HIGH)
3. Eliminate duplicate model training in `trainer.py:191-197` (HIGH)
4. Extract shared JSONL parsing and data loading utilities (MEDIUM)
5. Add file locking to `registry_promote` in MCP tools (MEDIUM)
6. Preserve exception chain in HPO (`from None` → `from e`) (MEDIUM)
7. Fix metrics lookup to use `model_task` not `self.task` in evaluator (MEDIUM)
