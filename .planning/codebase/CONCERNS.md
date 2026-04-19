# Concerns

## MEDIUM SEVERITY

### Type Stub Gaps
- Many optional dependencies (polars, optuna, catboost, etc.) lack type stubs
- Causes ~92 mypy errors, mostly import-not-found and import-untyped
- Currently mitigated with `ignore_missing_imports = true`

### Error Handling
- Bare `except` clause in `core/engine/trainer.py:241`
- Missing custom exceptions in `core/models/base.py`

### Example Config Type Safety
- `configs/examples/credit_risk.py` previously used strings instead of enum values
- Should consistently use TaskType, CVStrategy, TrackerType enums

## LOW SEVERITY

### Test Coverage Gaps
- No tests for: `mcp_server/tools.py`, `core/models/tabular_foundation/tabpfn_model.py`
- Integration tests for model selection and training pipeline exist but may need expansion

### No Explicit Tech Debt Tracking
- Zero TODO/FIXME/HACK/XXX comments found in source — debt not tracked in code

## RESOLVED

### main.py Monolith
- Previously was ~6500 lines; refactored to ~200 lines with proper CLI commands
- Each command (init, run, leaderboard, registry, hardware, drift, state, hpo) is now a discrete function

### Secret Management
- Environment variables in services layer now use pydantic-settings for validation
- No secret management system — but no obvious security holes either
