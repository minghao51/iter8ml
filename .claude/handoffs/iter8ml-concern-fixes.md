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

#### Fix 7: Extract shared JSONL parsing utility
- **Files:** 5 locations with identical pattern:
  - `core/engine/trainer.py:88-93` (in `_generate_leaderboard`)
  - `core/engine/state_observer.py:73-80`
  - `mcp_server/tools.py:108-114` (in `get_event_log`)
  - `mcp_server/tools.py:141-145` (in `registry_promote`)
  - `main.py:97-101`
- **What:** Create `core/utils/jsonl.py` with `load_events(path)` function
- **Why:** DRY violation, maintenance burden

#### Fix 8: Extract shared data loading utility
- **Files:** 4 locations with `if .parquet else .csv` pattern:
  - `mcp_server/tools.py:44-45` (in `run_baseline`)
  - `mcp_server/tools.py:76-77` (in `run_hpo`)
  - `main.py:66-73` (in `train` command)
  - `main.py:217` (in `hpo` command)
- **What:** Add `load_data(path: Path) → pl.DataFrame` to `core/data/loaders.py`
- **Why:** DRY violation

#### Fix 9: Add file locking to registry_promote
- **File:** `mcp_server/tools.py:129-168` (in `registry_promote` tool)
- **What:** Add `fcntl.flock` like `_update_registry` does in `trainer.py:63-83`
- **Why:** Concurrent MCP calls could corrupt registry.json
- **Reference pattern:** Copy from `trainer.py:_update_registry` exactly, including lock file creation

### LOW Priority

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

## Key Files Reference

```
core/models/tabular_foundation/tabpfn_model.py  — TabPFN wrapper (already edited: dead code removed)
core/models/conventional/catboost_model.py      — CatBoost wrapper (already edited: .get() fix)
core/models/conventional/lightgbm_model.py      — LightGBM wrapper (already edited: .get() fix)
core/models/conventional/xgboost_model.py       — XGBoost wrapper (already edited: .get() fix)
core/engine/trainer.py                          — Main trainer (260 lines)
core/engine/hpo.py                              — Optuna HPO (70 lines)
core/engine/evaluator.py                        — CV evaluation (95 lines)
core/data/loaders.py                            — Data ingestion (37 lines)
mcp_server/tools.py                             — MCP tools (187 lines)
main.py                                         — CLI entry point (245 lines)
core/engine/state_observer.py                   — State generation
```

## Testing Approach
After each fix (or batch of related fixes):
```bash
cd /Users/minghao/Desktop/personal/iter8ml
uv run pytest tests/ -v
uv run ruff check core/ mcp_server/ main.py
```

## Important Notes

1. **Fix 3 was reverted** — the "duplicate training" is intentional ML practice (CV for scoring + full-data for deployment artifact). Should be reclassified as "by design."

2. **Fix 4** — TabPFN uses PyTorch internally, so `torch.save/load` with `weights_only=True` is the right approach. May need to verify TabPFN model has `.state_dict()` and `.load_state_dict()` methods.

3. **Fixes 7 & 8** are refactoring — should be done together as they both extract shared utilities. Create `core/utils/` directory if it doesn't exist.

4. **Fix 9** — Copy the `fcntl.flock` pattern from `trainer.py:_update_registry` exactly, including lock file creation.

## Current State of Modified Files

| File | Changes Made |
|------|-------------|
| `tabpfn_model.py` | Dead code removed (lines 37-50 deleted). Pickle→torch NOT yet done. |
| `catboost_model.py` | `.pop()` → `.get()` done |
| `lightgbm_model.py` | `.pop()` → `.get()` done |
| `xgboost_model.py` | `.pop()` → `.get()` done |
| All other files | Untouched |
