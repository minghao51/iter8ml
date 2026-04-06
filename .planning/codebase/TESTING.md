# Testing Conventions

## Test Framework and Version

- **Framework**: pytest >= 8.0
- **Config**: `pyproject.toml:36` (dev dependency), `pyproject.toml:53-55` (pytest.ini_options)
- **Pytest options**:
  - `testpaths = ["tests"]`
  - `addopts = "-v --tb=short"`
- **No conftest.py** in project tests directory (no shared fixtures at root level)

## Test Directory Structure

```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_adapter.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_drift.py
│   ├── test_ft_transformer.py
│   ├── test_hpo.py
│   ├── test_loaders.py
│   ├── test_mcp_tools.py
│   ├── test_model_selector.py
│   ├── test_processors.py
│   ├── test_quality.py
│   ├── test_state_observer.py
│   └── test_tabpfn.py
└── integration/
    ├── __init__.py
    ├── test_full_pipeline.py
    └── test_gdbt_models.py
```

- **Unit tests**: `tests/unit/` — 13 test files testing individual modules
- **Integration tests**: `tests/integration/` — 2 test files testing full pipelines
- **Mapping**: Test files mirror source modules (e.g., `test_processors.py` → `core/data/processors.py`)

## Test Naming Conventions

### File Naming
- `test_<module_name>.py` (e.g., `test_config.py`, `test_loaders.py`, `test_drift.py`)

### Function Naming
- `test_<function_or_feature>_<scenario>()` pattern
- Examples:
  - `test_default_config()` — `tests/unit/test_config.py:9`
  - `test_fill_nulls_median()` — `tests/unit/test_processors.py:14`
  - `test_fill_nulls_categorical_unknown()` — `tests/unit/test_processors.py:40`
  - `test_no_drift_same_distribution()` — `tests/unit/test_drift.py:8`
  - `test_optimize_model_log_space()` — `tests/unit/test_hpo.py:107`
  - `test_full_pipeline_catboost_classification()` — `tests/integration/test_full_pipeline.py:15`

### Class Naming
- Dummy/mock classes use descriptive names: `DummyModel` at `tests/unit/test_hpo.py:10`
- Helper functions prefixed with `_`: `_make_classification_df()` at `tests/unit/test_quality.py:13`

## Test Patterns

### Simple Function Tests (No Fixtures)
- Most unit tests are simple functions with no fixtures
- Direct instantiation and assertion
- Example: `tests/unit/test_config.py:9-20`

```python
def test_default_config():
    config = ExperimentConfig(name="test", task="classification", ...)
    assert config.cv_folds == 5
```

### Pytest Fixtures
- **Module-level fixtures** using `@pytest.fixture` decorator
- **`tmp_path` fixture** (pytest built-in) used extensively for temp files
- **Custom fixtures**:
  - `sample_data` — `tests/unit/test_hpo.py:34-39` — returns sklearn classification data
  - `sample_csv` — `tests/unit/test_cli.py:17-24` — creates temp CSV with classification data
  - `sample_parquet` — `tests/unit/test_cli.py:27-34` — creates temp Parquet file
- **Fixture scope**: Default (function-scoped), no explicit scope parameter used

### Temp File Patterns
- `tempfile.NamedTemporaryFile()` for file-based tests: `tests/unit/test_loaders.py:24-29`
- `tempfile.TemporaryDirectory()` as context manager: `tests/integration/test_full_pipeline.py:20`
- `tmp_path` pytest fixture for path-based temp dirs: `tests/unit/test_cli.py:17-24`

### CLI Testing
- Uses `typer.testing.CliRunner`: `tests/unit/test_cli.py:10,14`
- `runner.invoke(app, ["command", "--flag", "value"])` pattern
- Asserts on `result.exit_code` and `result.stdout`
- Directory switching pattern with try/finally for workspace tests:
  ```python
  orig = os.getcwd()
  os.chdir(tmpdir)
  try:
      result = runner.invoke(app, ["init"])
      assert result.exit_code == 0
  finally:
      os.chdir(orig)
  ```
- File: `tests/unit/test_cli.py:38-50`

### Exception Testing
- Uses `pytest.raises(ValidationError)` context manager
- Example: `tests/unit/test_config.py:23-30`

```python
def test_invalid_task():
    with pytest.raises(ValidationError):
        ExperimentConfig(name="test", task="invalid_task", ...)
```

### Dummy/Mock Objects
- **DummyModel class** — `tests/unit/test_hpo.py:10-31`
  - Implements `fit()`, `predict()`, `predict_proba()` interface
  - Uses `np.random.seed(42)` for deterministic output
  - No mocking library used; hand-crafted dummy objects preferred
- No `unittest.mock`, `MagicMock`, or `patch` usage found in project tests

### Data Generation
- `sklearn.datasets.make_classification` and `make_regression` for synthetic data
- `polars.DataFrame` constructed inline for small test cases
- Helper function `_make_classification_df()` at `tests/unit/test_quality.py:13-23`

### Integration Test Pattern
- Full pipeline tests use `tempfile.TemporaryDirectory()` for isolated workspace
- Creates `ExperimentConfig`, `JSONLTracker`, `Trainer` and runs end-to-end
- Validates output files exist and contain expected content
- Files: `tests/integration/test_full_pipeline.py`, `tests/integration/test_gdbt_models.py`

## Coverage Configuration

- **No coverage tool configured** — no `pytest-cov` dependency
- **No coverage config** in `pyproject.toml`
- **No `.coveragerc`** file
- **No coverage badges or reports**

## CI Test Configuration

### GitHub Actions
- **File**: `.github/workflows/ci.yml:1-29`
- **Triggers**: push to `main`, pull requests to `main`
- **Jobs**:
  1. **`lint`** — runs `ruff check .` and `ruff format --check .`
  2. **`test`** — runs `uv run pytest tests/unit -v --tb=short`
- **Note**: Only `tests/unit` runs in CI, not `tests/integration`
- **Setup**: Uses `astral-sh/setup-uv@v5` with cache enabled

### Pre-commit Hooks
- **File**: `.pre-commit-config.yaml:1-19`
- **Hooks**:
  1. `ruff format` — formats Python files
  2. `ruff check --fix` — lints and auto-fixes
  3. `pytest unit tests` — runs `uv run pytest tests/unit -v`
- All hooks run via `uv run` (not direct executable)
- pytest hook has `pass_filenames: false` (runs all unit tests, not per-file)

## Test File Details

| File | Lines | Tests | Key Patterns |
|------|-------|-------|--------------|
| `tests/unit/test_config.py` | 41 | 3 | Pydantic validation, `pytest.raises` |
| `tests/unit/test_loaders.py` | 38 | 4 | `tempfile.NamedTemporaryFile`, hash consistency |
| `tests/unit/test_processors.py` | 127 | 14 | Pure function testing, edge cases |
| `tests/unit/test_hpo.py` | 124 | 7 | `@pytest.fixture`, DummyModel, sklearn data |
| `tests/unit/test_cli.py` | 187 | 13 | `CliRunner`, `tmp_path`, directory switching |
| `tests/unit/test_quality.py` | 77 | 5 | Helper function, noise injection |
| `tests/unit/test_drift.py` | 48 | 4 | Pydantic result validation |
| `tests/unit/test_adapter.py` | 28 | 2 | Type checking (`isinstance`) |
| `tests/unit/test_state_observer.py` | 47 | 2 | JSON file manipulation, `tmp_path` |
| `tests/unit/test_mcp_tools.py` | - | - | MCP tool testing |
| `tests/unit/test_model_selector.py` | - | - | Model selection logic |
| `tests/unit/test_ft_transformer.py` | - | - | FT-Transformer model |
| `tests/unit/test_tabpfn.py` | - | - | TabPFN model |
| `tests/integration/test_full_pipeline.py` | 85 | 2 | End-to-end, `TemporaryDirectory`, file validation |
| `tests/integration/test_gdbt_models.py` | - | - | GDBT model integration |
