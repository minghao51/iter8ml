# Handoff: iter8ml Codebase Concern Fixes

**Date:** 2026-04-06
**Project:** `/Users/minghao/Desktop/personal/iter8ml` (iter8ml / tabular-blueprint)

## Context
Working through identified code quality concerns from the codemap analysis (`CONCERNS.md`). Following systematic debugging workflow — each fix should be validated before moving to the next.

## Completed Fixes (3/11)

| # | Fix | Status | Files Modified |
|---|-----|--------|----------------|
| 1 | Remove dead code in tabpfn_model.py:37-50 | ✅ Done | `core/models/tabular_foundation/tabpfn_model.py` |
| 2 | Fix params mutation bug (`.pop()` → `.get()`) | ✅ Done | `catboost_model.py:22`, `lightgbm_model.py:20`, `xgboost_model.py:20` |
| 3 | Eliminate duplicate model training | ⏸️ Reverted | — |

**Note on Fix 3:** The "duplicate training" (CV folds + full-data artifact) is standard ML practice — CV trains fold models for scoring, then full-data training produces the deployable model. This is intentional, not a bug. Remove from concern list or reclassify as "by design."

## Remaining Fixes (8 items)

### HIGH Priority

#### Fix 4: Replace pickle with torch serialization
- **File:** `core/models/tabular_foundation/tabpfn_model.py` (save/load methods, lines 69-80)
- **What:** Replace `pickle.dump/load` with `torch.save/torch.load(weights_only=True)`
- **Why:** Pickle is a remote code execution vector for untrusted artifacts
- **Approach:** TabPFN uses PyTorch internally, so `model.state_dict()` and `model.load_state_dict()` should work
- **Caveat:** Verify TabPFN model has `.state_dict()` and `.load_state_dict()` methods before implementing

### MEDIUM Priority

#### Fix 5: Preserve exception chain in HPO
- **File:** `core/engine/hpo.py:61-62`
- **What:** Change `raise optuna.TrialPruned() from None` → `from e`
- **Why:** `from None` suppresses original traceback, making debugging impossible

#### Fix 6: Fix metrics lookup in evaluator
- **File:** `core/engine/evaluator.py:89`
- **What:** Change `METRICS_REGISTRY[self.task]` → `METRICS_REGISTRY[model_task]`
- **Why:** Task override parameter is ignored for metrics lookup, causing potential KeyError

#### Fix 7: Extract shared JSONL parsing utility - COMPLETED
- **File:** `core/utils/jsonl.py` already exists with proper implementation
- **What:** Replaced duplicate JSONL parsing with `load_events()` from `core/utils/jsonl.py`
- **Updated usages:**
  - `core/engine/trainer.py` → `from core.utils.jsonl import load_events`
  - `core/engine/state_observer.py` → `from core.utils.jsonl import load_events`
  - `mcp_server/tools.py` → `from core.utils.jsonl import load_events`
  - `main.py` → `from core.utils.jsonl import load_events`
- **Why:** DRY violation resolved, maintenance burden eliminated

#### Fix 8: Extract shared data loading utility - COMPLETED
- **File:** Added `load_data(path: Path) → pl.DataFrame` to `core/data/loaders.py`
- **What:** Consolidated duplicate `if .parquet else .csv` patterns from:
  - `mcp_server/tools.py:44-45` (in `run_baseline`)
  - `mcp_server/tools.py:76-77` (in `run_hpo`)
  - `main.py:66-73` (in `train` command)
  - `main.py:217` (in `hpo` command)
- **Why:** DRY violation eliminated

### LOW Priority

#### Fix 9: Add file locking to registry_promote
- **File:** `mcp_server/tools.py:129-168` (in `registry_promote` tool)
- **What:** Add `fcntl.flock` like `_update_registry` does in `trainer.py:63-83`
- **Why:** Concurrent MCP calls could corrupt registry.json
- **Reference pattern:** Copy from `trainer.py:_update_registry` exactly, including lock file creation

#### Fix 10: Fix SQLite connection leak
- **File:** `core/data/loaders.py:30-37` (in `load_sqlite`)
- **What:** Use context manager: `with sqlite3.connect(...) as conn:`
- **Why:** Connection leak on exception

#### Fix 11: Type annotations and minor fixes
Multiple small fixes:
- `trainer.py:138` — Add `df: pl.DataFrame` type hint
- `trainer.py:30` — Type `_MODEL_CLASS_CACHE: dict[str, type] = {}`
- `main.py:224` — Validate model name before `getattr()`, raise user-friendly error
- `hpo.py:39-51` — Validate search space tuple format, raise clear errors for malformed input
