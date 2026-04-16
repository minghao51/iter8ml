# Testing Conventions

Derived from pyproject.toml, conftest files, and test modules.

## Test Framework

### pytest configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "-v --tb=short"
```

- **Framework:** pytest (>= 8.0, declared in dev extras).
- **Default test path:** `tests/`
- **Output:** verbose (`-v`) with short tracebacks (`--tb=short`).
- **No coverage config** in pyproject.toml. No `[tool.coverage]` section.
- **No test markers** configured in pyproject.toml.

## Test Directory Structure

```
tests/
  __init__.py
  unit/                          # Fast, isolated unit tests
    __init__.py
    test_hpo.py
    test_loaders.py
    test_trainer.py
    test_report_service.py
    test_model_factory.py
    test_model_selector.py
    test_adapter.py
    test_config.py
    test_jsonl.py
    test_processors.py
    test_quality.py
    test_tracker_rotation.py
    test_ft_transformer.py
    test_tabpfn.py
    test_drift.py
    test_state_observer.py
    test_registry_service.py
    test_mcp_tools.py
    test_cli.py
  integration/                   # Slower, multi-component tests
    __init__.py
    conftest.py                  # Session-scoped fixtures for datasets
    test_hpo.py
    test_gdbt_models.py
    test_full_pipeline.py
    test_registry_and_drift.py
    test_model_selection.py
  fixtures/                      # Shared test fixtures
    __init__.py
    conftest.py
```

## Fixtures (conftest.py patterns)

### Integration conftest (`tests/integration/conftest.py`)

Uses `scope="session"` for expensive dataset fixtures shared across all integration tests:

```python
import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def classification_data():
    """Small classification dataset for quick integration tests."""
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture(scope="session")
def regression_data():
    """Small regression dataset for quick integration tests."""
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture(scope="session")
def large_classification_data():
    """Larger classification dataset (>10k rows) to test model routing."""
    X, y = make_classification(n_samples=15000, n_features=20, n_informative=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))


@pytest.fixture
def tmp_workspace(tmp_path):
    """Temporary workspace directory for experiment outputs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace
```

### Inline fixtures in unit tests

Unit tests typically define fixtures locally within the test file rather than in a shared conftest:

```python
# tests/unit/test_hpo.py
@pytest.fixture
def sample_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y
```

### Built-in pytest fixtures used

- `tmp_path` -- temporary directory (pytest built-in), used extensively for file-based tests.
- `monkeypatch` -- used for swapping module-level objects (e.g., replacing `RegistryService` in trainer tests).

## Test Naming Convention

- **File names:** `test_<module_name>.py` (e.g., `test_hpo.py`, `test_loaders.py`, `test_report_service.py`).
- **Function names:** `test_<behavior_description>` using snake_case.
- **Docstrings:** Many test functions have a one-line docstring describing the expected behavior (especially for edge cases and integration tests).

Examples:

```python
def test_get_data_hash_consistency():
def test_load_sqlite_invalid_path():
def test_optimize_model_preserves_exception_context():
def test_hpo_returns_best_params(hpo_classification_data, tmp_path):
def test_build_report_classification_orders_by_primary_score(tmp_path):
```

## Test Structure Patterns

### Arrange-Act-Assert

Tests follow a flat AAA style without classes (most tests are plain functions):

```python
def test_validate_model_name_returns_name():
    assert validate_model_name("catboost") == "catboost"
```

### Class-based test helpers (rare)

Only seen for dummy models used as test doubles:

```python
class DummyModel:
    """A deterministic dummy model for HPO testing."""
    model_name = "Dummy"

    def __init__(self, task="classification", lr=0.01, n_estimators=100):
        ...

    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
```

## Mocking Patterns

### unittest.mock.Mock

Used sparingly for isolating external dependencies:

```python
# tests/unit/test_hpo.py
from unittest.mock import Mock

evaluator = Mock()
call_count = [0]

def side_effect(*args, **kwargs):
    call_count[0] += 1
    if call_count[0] == 1:
        raise ValueError("Invalid data shape")
    return {"roc_auc": 0.8}

evaluator.evaluate.side_effect = side_effect
```

### monkeypatch for dependency injection

Used to replace module-level imports in tests:

```python
# tests/unit/test_trainer.py
def test_trainer_uses_registry_service(tmp_path, monkeypatch):
    class MockRegistryService:
        def __init__(self, registry_path): ...
        def update_if_better(self, key, model_name, run_id, score, artifact_path): ...

    import core.engine.trainer
    monkeypatch.setattr(core.engine.trainer, "RegistryService", MockRegistryService)
```

### Dummy models instead of mocks

For model-dependent tests, the project uses concrete dummy classes that satisfy the `AbstractModel` protocol rather than Mock objects:

```python
class DummyModel:
    """A deterministic dummy model for HPO testing."""
    model_name = "Dummy"

    def __init__(self, task="classification", lr=0.01, n_estimators=100): ...
    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
```

## pytest.raises Patterns

### Matching error messages

Tests use `pytest.raises` with `match=` for regex-based error message validation:

```python
with pytest.raises(ValueError, match="Unknown model"):
    get_model_class("not_a_model")

with pytest.raises(FileNotFoundError, match="Database file not found"):
    load_sqlite("/nonexistent/db.sqlite", "SELECT 1")

with pytest.raises(ValueError, match="Only SELECT queries are supported"):
    load_sqlite(db_file, "INVALID SQL QUERY")

with pytest.raises(ValueError, match="Query cannot be empty"):
    load_sqlite(db_file, "")
```

### Matching multiple exception types

For integration tests where the exact exception type may vary:

```python
with pytest.raises((ValueError, TypeError)):
    optimize_model(..., search_space=invalid_search_space, ...)
```

## File-Based Test Patterns

### tmp_path for temporary files

Tests that need temporary files use pytest's `tmp_path` fixture. This is the standard approach for tests involving file I/O:

```python
def test_build_report_empty(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    log_path.touch()
    report = ReportService(log_path=log_path, registry_path=tmp_path / "registry.json").build_report()
```

### JSONL event construction

For report service tests, events are constructed as dicts and written as JSONL:

```python
events = [
    {"event": "model_completed", "run_id": "run_low", "model": "ModelLow", ...},
    {"event": "model_completed", "run_id": "run_high", "model": "ModelHigh", ...},
]
log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
```

### tempfile for CSV tests

Older tests use `tempfile.NamedTemporaryFile` for CSV loading tests:

```python
def test_load_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n4,5,6\n")
        f.flush()
        df = load_csv(f.name)
```

## Unit vs Integration Test Boundaries

### Unit tests (`tests/unit/`)

- No external dependencies (no real model training, no file I/O where avoidable).
- Use dummy models, mocks, and monkeypatch.
- Test individual functions and classes in isolation.
- Typically fast (< 1 second each).

### Integration tests (`tests/integration/`)

- Use real model classes (e.g., `get_model_class("catboost")`).
- Use real `ExperimentConfig` and `Evaluator` instances.
- Use `session`-scoped fixtures for dataset generation.
- May have longer runtimes (HPO tests assert elapsed time < 120 seconds).

## Coverage

No coverage configuration exists in pyproject.toml. No `.coveragerc` or `setup.cfg` coverage section found. Coverage is not enforced in CI test runs by default.
