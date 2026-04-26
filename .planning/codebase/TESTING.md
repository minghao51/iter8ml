# Testing

> Last updated: 2026-04-23

## Test Framework

- **Framework**: [pytest](https://docs.pytest.org/) >= 8.0
- **Config**: `pyproject.toml` → `[tool.pytest.ini_options]`
- **Runner**: `uv run pytest` (always via `uv run`, never bare `python` or `pytest`)
- **Dev dependency group**: `[dependency-groups] dev = ["pytest>=8.0", "ruff>=0.4", "pre-commit>=3.6"]`

## Test Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = ["--strict-markers", "-ra", "--durations=10", "--import-mode=importlib"]
```

Key settings:
- `--strict-markers`: Unregistered markers cause errors — all markers must be declared
- `-ra`: Show summary of all test outcomes (not just failures)
- `--durations=10`: Show the 10 slowest tests
- `--import-mode=importlib`: Use importlib mode for test imports (avoids `__init__.py` issues)
- `pythonpath = ["src"]`: Makes `tabular_blueprint` importable without installation

## Test Structure

### Directory Layout

```
tests/
├── conftest.py                      # Root fixtures + auto-marking hook
├── unit/                            # Fast isolated unit tests (~30 files)
│   ├── test_adapter.py
│   ├── test_baselines.py
│   ├── test_calibration.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_domain_classifier.py
│   ├── test_drift.py
│   ├── test_exceptions.py
│   ├── test_export_service.py
│   ├── test_feature_engine.py
│   ├── test_ft_transformer.py
│   ├── test_hpo_importance.py
│   ├── test_hpo_unit.py
│   ├── test_hpo_warmstart.py
│   ├── test_jsonl.py
│   ├── test_leakage.py
│   ├── test_llm_agent.py
│   ├── test_loaders.py
│   ├── test_mcp_tools.py
│   ├── test_model_factory.py
│   ├── test_model_selector.py
│   ├── test_processors.py
│   ├── test_psi_drift.py
│   ├── test_quality.py
│   ├── test_registry_service.py
│   ├── test_report_service.py
│   ├── test_state_observer.py
│   ├── test_tabpfn.py
│   ├── test_tabpfn_guardrails.py
│   ├── test_tracker_rotation.py
│   └── test_trainer.py
├── integration/                     # Cross-component integration tests (~5 files)
│   ├── conftest.py                  # Larger datasets (15k rows)
│   ├── test_full_pipeline.py
│   ├── test_gdbt_models.py
│   ├── test_hpo.py
│   ├── test_model_selection.py
│   └── test_registry_and_drift.py
└── e2e/                             # End-to-end tests (placeholder, has .gitkeep)
    └── .gitkeep
```

### File Naming

- Test files: `test_{module_name}.py` — matches source file names
- Convention: one test file per source module (e.g., `test_drift.py` tests `monitoring/drift.py`)

### Auto-Marking

`tests/conftest.py` automatically applies markers based on file path:

```python
def pytest_collection_modifyitems(items):
    for item in items:
        if "unit/" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "e2e/" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
```

## Test Types

### Unit Tests (`tests/unit/`)

- **Scope**: Single function/class in isolation
- **Speed**: Fast (no real model training where possible)
- **Data**: Small synthetic datasets (200-500 rows via `sklearn.datasets`)
- **Count**: ~30 test files
- **Marker**: `@pytest.mark.unit` (auto-applied by path)

### Integration Tests (`tests/integration/`)

- **Scope**: Multiple components working together (e.g., Trainer + Config + Tracker + Models)
- **Speed**: Slow (actual model training with cross-validation)
- **Data**: Larger synthetic datasets (500-15,000 rows)
- **Count**: ~5 test files
- **Markers**: `@pytest.mark.integration` + `@pytest.mark.slow` (auto-applied by path)

### E2E Tests (`tests/e2e/`)

- **Status**: Placeholder (`.gitkeep` only, no tests yet)
- **Marker**: `@pytest.mark.e2e` + `@pytest.mark.slow`

## Registered Markers

All markers declared in `pyproject.toml`:

| Marker | Description |
|--------|-------------|
| `slow` | Tests taking >1s |
| `integration` | External services or slow tests |
| `unit` | Fast isolated tests |
| `e2e` | Full workflow tests |
| `serial` | Cannot run in parallel |
| `network` | Needs internet/auth |
| `smoke` | Critical path tests |

## Test Commands

### Run All Tests
```bash
uv run pytest                          # All tests
uv run pytest -v                       # Verbose
uv run pytest --tb=short               # Shorter tracebacks
```

### Run by Type
```bash
uv run pytest tests/unit/              # Unit tests only
uv run pytest tests/integration/       # Integration tests only
uv run pytest -m unit                  # By marker
uv run pytest -m "not slow"            # Exclude slow tests
uv run pytest -m "not network"         # Exclude network tests
```

### Run Specific Tests
```bash
uv run pytest tests/unit/test_drift.py                     # Single file
uv run pytest tests/unit/test_drift.py::test_no_drift      # Single test
uv run pytest -k "drift"                                   # By keyword
```

### Other Commands
```bash
uv run pytest --durations=0            # Show all test durations
uv run pytest -x                       # Stop on first failure
uv run pytest --lf                      # Run last-failed tests
```

## Fixtures

### Root Fixtures (`tests/conftest.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `classification_data` | `session` | 500-row Polars DataFrame with 10 features, binary target (sklearn `make_classification`) |
| `regression_data` | `session` | 500-row Polars DataFrame with 10 features, continuous target (sklearn `make_regression`) |
| `tmp_workspace` | `function` | Temporary workspace directory under `tmp_path` |

**Data fixture pattern**:
```python
@pytest.fixture(scope="session")
def classification_data():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(target=pl.Series(y))
```

### Integration Fixtures (`tests/integration/conftest.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `large_classification_data` | `session` | 15,000-row Polars DataFrame with 20 features |

### Module-Level Fixtures

Many test files define their own fixtures for module-specific data:
- `tests/unit/test_baselines.py`: `classification_data` (200 rows), `regression_data` (200 rows)
- `tests/unit/test_cli.py`: `sample_csv`, `sample_parquet` (100-row CSV/Parquet files)
- `tests/integration/test_gdbt_models.py`: `classification_data`, `regression_data` (500 rows each)

## Mocking and Stubbing

### Primary Approach: Monkeypatch

- **`monkeypatch`** (pytest builtin) is the primary mocking tool
- Used for replacing module-level imports and class references
- No external mocking library (no `unittest.mock`, no `pytest-mock`)

### Patterns

**1. Monkeypatching module attributes** (`tests/unit/test_trainer.py`):
```python
def test_trainer_uses_registry_service(tmp_path, monkeypatch):
    class MockRegistryService:
        def __init__(self, registry_path):
            self.registry_path = registry_path
        def update_if_better(self, key, model_name, run_id, score, artifact_path, metric_name=None):
            registry_calls.append({...})
            return True

    monkeypatch.setattr(
        tabular_blueprint.engine.trainer, "RegistryService", MockRegistryService
    )
```

**2. Fake/stub classes for testing decorators** (`tests/unit/test_exceptions.py`):
```python
class DummyTracker:
    def __init__(self):
        self.events = []
    def log_event(self, event):
        self.events.append(event)

class DummyTrainer:
    def __init__(self):
        self.tracker = DummyTracker()
    @track_errors()
    def good_method(self):
        return 42
```

**3. `pytest.raises` for exception testing** (used throughout):
```python
with pytest.raises(ValueError, match="Unknown model"):
    get_model_class("not_a_model")
```

### Not Used

- No `unittest.mock.patch`
- No `pytest-mock` plugin
- No mock libraries

## Test Writing Patterns

### Test Function Style

- **Flat functions preferred** over classes for simple test files:
  ```python
  def test_get_data_hash_consistency():
      df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
      hash1 = get_data_hash(df)
      hash2 = get_data_hash(df)
      assert hash1 == hash2
  ```

- **Class-based grouping** used when testing a class with multiple methods:
  ```python
  class TestNaiveBaseline:
      def test_classification_fit_predict(self, classification_data):
          ...
      def test_regression_fit_predict(self, regression_data):
          ...
  ```

### Common Test Data Creation

Synthetic data via sklearn, wrapped in Polars:
```python
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
df = df.with_columns(target=pl.Series(y))
```

### Temporary File Handling

- `tmp_path` (pytest builtin) for temporary directories
- `tempfile.TemporaryDirectory()` for CLI tests that need `os.chdir()`
- Always use context manager pattern with `os.chdir()` to restore original directory

### Integration Test Pattern

Integration tests follow a consistent pattern:
1. Create synthetic data (Polars DataFrame)
2. Create `ExperimentConfig` with `workspace_dir=tmp_path`
3. Create `JSONLTracker` pointing to temp workspace
4. Create `Trainer` with config + tracker
5. Call `trainer.run(df)`
6. Assert on results dict and/or read back JSONL events

```python
def test_full_pipeline_catboost_classification():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    df = df.with_columns(target=pl.Series(y))

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        config = ExperimentConfig(
            name="integration_test", task="classification",
            target_col="target", data_path="", workspace_dir=ws,
            cv_folds=3, metrics=["roc_auc", "f1_macro"],
        )
        tracker = JSONLTracker(str(ws / "experiments.jsonl"))
        trainer = Trainer(config, tracker=tracker)
        results = trainer.run(df)

        assert "catboost" in results
        assert "roc_auc" in results["catboost"]["cv_scores"]
```

### CLI Test Pattern

Uses `typer.testing.CliRunner`:
```python
from typer.testing import CliRunner
from tabular_blueprint.cli import app

runner = CliRunner()

def test_init_command():
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Workspace initialized" in result.stdout
```

## Coverage

- **No coverage tool configured** — no `.coveragerc`, no `pytest-cov` in dependencies, no coverage flags in CI
- Tests focus on correctness, not coverage metrics

## CI Testing

### GitHub Actions (`tests/.github/workflows/ci.yml`)

**Jobs**:

| Job | Runner | What it runs |
|-----|--------|-------------|
| `lint` | ubuntu-latest | `uv run ruff check .` |
| `format` | ubuntu-latest | `uv run ruff format --check .` |
| `test` | ubuntu-latest | Unit + integration tests on Python 3.11, 3.12, 3.13 |
| `pre-commit` | ubuntu-latest | `uvx pre-commit run --all-files` |

**Test job detail**:
```yaml
test:
  strategy:
    matrix:
      python-version: ['3.11', '3.12', '3.13']
  steps:
    - uv sync --frozen --group dev --extra llm
    - uv run pytest tests/unit/ -v --tb=short
    - uv run pytest tests/integration/ -v --tb=short
```

**Dependency install command**: `uv sync --frozen --group dev --extra llm`

**Pre-commit hook** (local): Runs `uv run --group dev --extra llm pytest tests/unit -v` on every commit (all Python files trigger it, `pass_filenames: false`)

### Running Tests Locally

```bash
# Install dev dependencies
uv sync --group dev --extra llm

# Run the same checks as CI
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit/ -v --tb=short
uv run pytest tests/integration/ -v --tb=short
```
