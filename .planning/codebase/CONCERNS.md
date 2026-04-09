# Codebase Concerns Analysis

## Technical Debt

### Security Concerns
- **Pickle Serialization Risk** (`core/models/tabular_foundation/tabpfn_model.py:56-65`)
  - Using `torch.load()` without `weights_only=True` in load method
  - Remote code execution vulnerability when loading untrusted artifacts
  - **Status**: Awaiting fix - needs `weights_only=True` parameter

### Performance Issues
- **Large File: trainer.py** (354 lines)
  - Complex monolithic function with multiple responsibilities
  - Should be broken into smaller, focused modules
- **Large File: ft_transformer.py** (162 lines)
  - Deep learning model with potential memory issues
  - No batch size optimization or memory management

### Code Quality Concerns
- **Duplicate JSONL parsing logic** (5 locations)
  - `core/engine/trainer.py:88-93`
  - `core/engine/state_observer.py:73-80`
  - `mcp_server/tools.py:108-114`
  - `mcp_server/tools.py:141-145`
  - `main.py:97-101`
- **Duplicate Data Loading Pattern** (4 locations)
  - `mcp_server/tools.py:44-45`
  - `mcp_server/tools.py:76-77`
  - `main.py:66-73`
  - `main.py:217`
- **SQLite Connection Leak** (`core/data/loaders.py:30-37`)
  - No context manager for database connections
  - Potential connection leaks on exceptions

## Known Bugs

### Critical
- **Exception Chain Loss** (`core/engine/hpo.py:61-62`)
  - Using `raise optuna.TrialPruned() from None`
  - Original traceback is suppressed, making debugging impossible
- **Metrics Lookup Bug** (`core/engine/evaluator.py:89`)
  - Using `METRICS_REGISTRY[self.task]` instead of `METRICS_REGISTRY[model_task]`
  - Task override parameter is ignored, causing potential KeyError
- **Params Mutation Bug** (FIXED in handoffs)
  - Using `.pop()` which mutates input config
  - Fixed in `catboost_model.py`, `lightgbm_model.py`, `xgboost_model.py`

### Medium Priority
- **Registry Race Condition** (`mcp_server/tools.py:129-168`)
  - No file locking for concurrent registry updates
  - Could corrupt `registry.json` with concurrent MCP calls
- **Missing Type Annotations**
  - `trainer.py:138` - Missing `df: pl.DataFrame` type hint
  - `trainer.py:30` - Missing `_MODEL_CLASS_CACHE: dict[str, type]` type hint
  - `main.py:224` - No validation before `getattr()` call
- **Search Space Validation** (`core/engine/hpo.py:39-51`)
  - Insufficient validation for malformed search space tuples
  - Could cause runtime errors

## Security Concerns

### Hardcoded Values
- **Max Rows Hardcoded** (`core/models/tabular_foundation/tabpfn_model.py:18`)
  - `MAX_ROWS = 10_000` should be configurable
- **OMP_NUM_THREADS** (`core/engine/trainer.py:19-20`)
  - Hardcoded thread pinning for macOS ARM64
  - Should be configurable or based on CPU count

### Secrets Management
- No hardcoded API keys or secrets found
- Environment variable access is minimal and appropriate

## Performance Issues

### Potential Bottlenecks
- **No Batch Processing** in data loading
- **No Memory Optimization** for large datasets
- **No Parallel Processing** for multi-model training
- **No Caching** for repeated operations

## Outdated Dependencies/Patterns

### Code Patterns
- **Using `importlib` for dynamic loading** (`main.py:45-48`)
  - Modern alternatives like `importlib.resources` available
- **Manual file locking** (`core/engine/trainer.py:68-90`)
  - Could use `pathlib.Path` with context managers
- **Legacy exception chaining** (`core/engine/hpo.py:61-62`)
  - Should use `from e` instead of `from None`

### Dependencies
- All dependencies appear to be current and maintained
- Using modern Python 3.11+ features
- UV for package management is current

## Recommendations

### High Priority
1. Replace pickle serialization with `torch.load(weights_only=True)`
2. Fix exception chaining in HPO
3. Fix metrics lookup bug in evaluator
4. Add file locking to registry operations
5. Extract shared utilities for JSONL parsing and data loading

### Medium Priority
1. Add comprehensive type annotations
2. Improve error handling and validation
3. Refactor large files into smaller modules
4. Add memory management for deep learning models
5. Make hardcoded values configurable

### Low Priority
1. Replace manual file locking with context managers
2. Add batch processing capabilities
3. Implement caching for repeated operations
4. Consider parallel processing for multi-model training
