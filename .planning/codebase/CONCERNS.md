# Concerns

## HIGH SEVERITY

### main.py Monolith
- `main.py` (~6500 lines) — extremely large entry point doing too much
- Needs split into smaller modules (CLI commands, config loading, orchestration)

### Secret Management
- Environment variables in `core/services/registry_service.py` and `core/services/report_service.py` lack validation
- No secret management system — configs in `configs/experiment.py` may have insecure defaults

## MEDIUM SEVERITY

### Circular Dependencies
- Potential circular imports between `core/engine/` and `core/services/`
- `core/engine/trainer.py` imports from services layer

### Error Handling
- Bare `except` clauses in `core/engine/trainer.py`
- Missing custom exceptions in `core/models/base.py`

### Performance
- Synchronous blocking calls in `core/engine/trainer.py`
- Missing pagination in `core/services/registry_service.py`
- Potential N+1 pattern in `core/engine/hpo.py` hyperparameter sweeps

## LOW SEVERITY

### Test Coverage Gaps
- No tests for: `mcp_server/tools.py`, `core/engine/state_observer.py`, `core/models/tabular_foundation/tabpfn_model.py`
- Limited integration tests for model selection and training pipeline

### No Explicit Tech Debt Tracking
- Zero TODO/FIXME/HACK/XXX comments found in source — debt not tracked in code
