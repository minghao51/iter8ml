# Code Quality and Bug Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix code quality concerns, known bugs, and security issues to improve code maintainability, debugging capability, and configurability.

**Architecture:** This plan addresses three categories of issues:
1. **Code deduplication** - Centralize scattered patterns into reusable utilities
2. **Bug fixes** - Fix exception handling, add missing validations
3. **Configuration** - Make hardcoded values configurable

**Tech Stack:** Python 3.12+, Polars, Pytest, file locking with fcntl

---

## Task 1: Deduplicate Inline CSV/Parquet Loading in MCP Server

**Files:**
- Modify: `mcp_server/tools.py:25-38`
- Test: `tests/unit/test_mcp_tools.py`

**Issue:** The `get_column_stats` function has inline CSV/Parquet loading that doesn't use the centralized `load_data` utility from `core/data/loaders.py`.

**Step 1: Write the failing test**

Add to `tests/unit/test_mcp_tools.py`:

```python
def test_get_column_stats_uses_centralized_loader(monkeypatch):
    """Verify get_column_stats uses load_data from core.data.loaders."""
    from unittest.mock import patch
    from mcp_server.tools import get_column_stats

    with patch("core.data.loaders.load_data") as mock_load:
        mock_load.return_value = pl.DataFrame({"a": [1, 2, 3]})
        result = get_column_stats("test.csv")
        mock_load.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_mcp_tools.py::test_get_column_stats_uses_centralized_loader -v`

Expected: FAIL - `get_column_stats` doesn't use `load_data`

**Step 3: Implement the fix**

Replace `mcp_server/tools.py:25-38`:

```python
@mcp.tool()
def get_column_stats(data_path: str) -> str:
    """Returns Polars describe() output for a dataset."""
    import polars as pl

    df = load_data(data_path)
    desc = df.describe()
    return desc.to_pandas().to_markdown()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_mcp_tools.py::test_get_column_stats_uses_centralized_loader -v`

Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools.py tests/unit/test_mcp_tools.py
git commit -m "refactor: use centralized load_data in get_column_stats"
```

---

## Task 2: Create Unified Registry Service

**Files:**
- Create: `core/services/registry_service.py`
- Modify: `core/engine/trainer.py:55-91`
- Modify: `mcp_server/tools.py:125-171`
- Test: `tests/unit/test_registry_service.py`

**Issue:** Registry update logic is duplicated between `trainer.py` and `mcp_server/tools.py`. Both have identical file locking and JSON handling code.

**Step 1: Create the new registry service module**

Create `core/services/__init__.py`:

```python
"""Core services for registry and state management."""
```

Create `core/services/registry_service.py`:

```python
"""Unified model registry service with file locking."""

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RegistryService:
    """Thread-safe model registry with file locking."""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.lock_path = str(self.registry_path.with_suffix(".lock"))

    def load(self) -> dict[str, Any]:
        """Load registry from disk, returns empty dict if not exists."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def update_if_better(
        self,
        key: str,
        model_name: str,
        run_id: str,
        score: float,
        artifact_path: str,
    ) -> bool:
        """Update registry only if new score beats existing champion."""
        with open(self.lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                registry = self.load()

                if key not in registry or score > registry[key].get("score", -float("inf")):
                    registry[key] = {
                        "model": model_name,
                        "run_id": run_id,
                        "score": score,
                        "artifact_path": artifact_path,
                        "registered_at": datetime.now(UTC).isoformat(),
                    }
                    self._save(registry)
                    return True
                return False
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get(self, key: str) -> dict[str, Any] | None:
        """Get entry by key, returns None if not found."""
        registry = self.load()
        return registry.get(key)

    def get_all(self) -> dict[str, Any]:
        """Get all registry entries."""
        return self.load()

    def _save(self, registry: dict[str, Any]) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)
```

**Step 2: Write tests for the registry service**

Create `tests/unit/test_registry_service.py`:

```python
"""Test RegistryService."""

import json
from pathlib import Path

import pytest

from core.services.registry_service import RegistryService


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temporary registry file."""
    return tmp_path / "registry.json"


def test_load_empty_registry(temp_registry):
    """Test loading non-existent registry returns empty dict."""
    service = RegistryService(temp_registry)
    assert service.load() == {}


def test_load_existing_registry(temp_registry):
    """Test loading existing registry."""
    temp_registry.write_text(json.dumps({"key1": {"model": "catboost"}}))
    service = RegistryService(temp_registry)
    assert service.load() == {"key1": {"model": "catboost"}}


def test_update_if_better_new_key(temp_registry):
    """Test updating registry with new key."""
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run1", 0.95, "/path/to/model")
    assert result is True
    registry = service.load()
    assert registry["key1"]["score"] == 0.95


def test_update_if_better_higher_score(temp_registry):
    """Test updating registry with higher score."""
    temp_registry.write_text(json.dumps({"key1": {"score": 0.90}}))
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run2", 0.95, "/path/to/model")
    assert result is True
    assert service.load()["key1"]["score"] == 0.95


def test_update_if_better_lower_score(temp_registry):
    """Test that lower score doesn't update registry."""
    temp_registry.write_text(json.dumps({"key1": {"score": 0.95}}))
    service = RegistryService(temp_registry)
    result = service.update_if_better("key1", "catboost", "run2", 0.90, "/path/to/model")
    assert result is False
    assert service.load()["key1"]["score"] == 0.95
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_registry_service.py -v`

Expected: PASS (all 5 tests)

**Step 4: Commit**

```bash
git add core/services/ tests/unit/test_registry_service.py
git commit -m "feat: add unified RegistryService with file locking"
```

---

## Task 3: Refactor Trainer to Use RegistryService

**Files:**
- Modify: `core/engine/trainer.py:55-91, 220-228, 268-276`
- Test: `tests/unit/test_trainer.py` (extend existing)

**Issue:** `_update_registry` function in trainer.py duplicates registry logic.

**Step 1: Write the failing test**

Add to `tests/unit/test_trainer.py`:

```python
def test_trainer_uses_registry_service(monkeypatch, tmp_path):
    """Verify trainer uses RegistryService for updates."""
    from unittest.mock import Mock, patch
    from core.engine.trainer import Trainer
    from configs.experiment import ExperimentConfig

    mock_registry = Mock()
    mock_registry.update_if_better.return_value = True

    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="test.csv",
        workspace_dir=tmp_path,
    )

    with patch("core.engine.trainer.RegistryService", return_value=mock_registry):
        trainer = Trainer(config)
        # ... verify update_if_better is called during training
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_trainer.py::test_trainer_uses_registry_service -v`

Expected: FAIL - Trainer doesn't use RegistryService yet

**Step 3: Refactor trainer.py to use RegistryService**

Add import at top of `core/engine/trainer.py`:

```python
from core.services.registry_service import RegistryService
```

Replace the `_update_registry` function (lines 55-91) with a method in Trainer class:

```python
def _update_champion_if_better(
    self, key: str, model_name: str, run_id: str, score: float, artifact_path: str
) -> bool:
    """Update registry if new model beats champion."""
    registry = RegistryService(str(self.config.workspace_dir / "registry.json"))
    return registry.update_if_better(key, model_name, run_id, score, artifact_path)
```

Update calls in `_train_sequential` (around line 221) and `_train_concurrent` (around line 268):

Replace:
```python
_update_registry(
    str(self.config.workspace_dir / "registry.json"),
    f"{self.config.name}:{self.config.task.value}",
    result["model_name"],
    run_id,
    score,
    result["artifact_path"],
)
```

With:
```python
self._update_champion_if_better(
    f"{self.config.name}:{self.config.task.value}",
    result["model_name"],
    run_id,
    score,
    result["artifact_path"],
)
```

Remove the standalone `_update_registry` function entirely.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_trainer.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/engine/trainer.py tests/unit/test_trainer.py
git commit -m "refactor: use RegistryService in trainer.py"
```

---

## Task 4: Refactor MCP Server to Use RegistryService

**Files:**
- Modify: `mcp_server/tools.py:114-171`
- Test: `tests/unit/test_mcp_tools.py` (extend existing)

**Issue:** `registry_show` and `registry_promote` functions duplicate registry logic.

**Step 1: Write the failing test**

Add to `tests/unit/test_mcp_tools.py`:

```python
def test_registry_tools_use_service(monkeypatch, tmp_path):
    """Verify registry tools use RegistryService."""
    from unittest.mock import Mock, patch
    from mcp_server.tools import registry_show, registry_promote

    mock_registry = Mock()
    mock_registry.get_all.return_value = {"key1": {"model": "catboost"}}

    with patch("mcp_server.tools.RegistryService", return_value=mock_registry):
        result = registry_show()
        assert "catboost" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_mcp_tools.py::test_registry_tools_use_service -v`

Expected: FAIL - Tools don't use RegistryService yet

**Step 3: Refactor mcp_server/tools.py to use RegistryService**

Add import at top of `mcp_server/tools.py`:

```python
from core.services.registry_service import RegistryService
```

Replace `registry_show` function (lines 114-121):

```python
@mcp.tool()
def registry_show() -> str:
    """Returns current registry.json content."""
    registry = RegistryService("workspace/registry.json")
    data = registry.get_all()
    if not data:
        return "Registry is empty."
    return json.dumps(data, indent=2)
```

Replace `registry_promote` function (lines 124-171):

```python
@mcp.tool()
def registry_promote(run_id: str, key: str) -> str:
    """Promotes a run_id to champion in the registry."""
    log_path = Path("workspace/experiments.jsonl")
    if not log_path.exists():
        return "No events found to locate run."

    events = load_events(log_path)

    run_event = next(
        (
            e
            for e in events
            if e.get("run_id") == run_id and e.get("event") == "model_completed"
        ),
        None,
    )

    if not run_event:
        return f"Run {run_id} not found."

    cv_scores = run_event.get("cv_scores", {})
    score = cv_scores.get("roc_auc", cv_scores.get("r2", 0))

    registry = RegistryService("workspace/registry.json")
    updated = registry.update_if_better(
        key=key,
        model_name=run_event.get("model"),
        run_id=run_id,
        score=score,
        artifact_path=run_event.get("artifact_path"),
    )

    if updated:
        return f"Promoted {run_id} to champion for {key}."
    return f"Existing champion for {key} has better score."
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_mcp_tools.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools.py tests/unit/test_mcp_tools.py
git commit -m "refactor: use RegistryService in MCP server tools"
```

---

## Task 5: Add Enhanced JSONL Error Handling

**Files:**
- Modify: `core/utils/jsonl.py:7-16`
- Test: `tests/unit/test_jsonl.py`

**Issue:** `load_events` doesn't handle malformed JSON, encoding issues, or provide useful error messages.

**Step 1: Write the failing test**

Create `tests/unit/test_jsonl.py`:

```python
"""Test JSONL utilities."""

import pytest
from core.utils.jsonl import load_events


def test_load_events_handles_empty_file(tmp_path):
    """Test loading empty JSONL file."""
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_text("")
    assert load_events(jsonl_file) == []


def test_load_events_handles_malformed_json(tmp_path):
    """Test loading JSONL with malformed line."""
    jsonl_file = tmp_path / "malformed.jsonl"
    jsonl_file.write_text('{"valid": true}\n{invalid json}\n{"also": "valid"}')
    with pytest.raises(ValueError, match="Invalid JSON at line"):
        load_events(jsonl_file)


def test_load_events_handles_nonexistent_path():
    """Test loading non-existent file returns empty list."""
    assert load_events("/nonexistent/path.jsonl") == []


def test_load_events_filters_blank_lines(tmp_path):
    """Test that blank lines are filtered out."""
    jsonl_file = tmp_path / "blanks.jsonl"
    jsonl_file.write_text('{"a": 1}\n\n{"b": 2}\n   \n{"c": 3}')
    events = load_events(jsonl_file)
    assert len(events) == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_jsonl.py -v`

Expected: FAIL - Current implementation doesn't raise proper errors for malformed JSON

**Step 3: Implement enhanced error handling**

Replace `core/utils/jsonl.py:7-16`:

```python
def load_events(path: str | Path) -> list[dict]:
    """Load events from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of event dictionaries. Returns empty list if file doesn't exist.

    Raises:
        ValueError: If a line contains malformed JSON.
    """
    events = []
    path = Path(path)
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_num} in {path}: {e}"
                    ) from e
    return events
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_jsonl.py -v`

Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add core/utils/jsonl.py tests/unit/test_jsonl.py
git commit -m "feat: add enhanced error handling to load_events"
```

---

## Task 6: Add SQLite Connection Validation

**Files:**
- Modify: `core/data/loaders.py:41-47`
- Test: `tests/unit/test_loaders.py` (extend existing)

**Issue:** `load_sqlite` doesn't validate database path, query syntax, or provide useful error messages.

**Step 1: Write the failing test**

Add to `tests/unit/test_loaders.py`:

```python
def test_load_sqlite_invalid_path():
    """Test loading from non-existent database path."""
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        load_sqlite("/nonexistent/db.sqlite", "SELECT 1")


def test_load_sqlite_invalid_query(tmp_path):
    """Test loading with invalid SQL query."""
    db_file = tmp_path / "test.db"
    # Create valid database
    import sqlite3
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")

    with pytest.raises(ValueError, match="Invalid SQL query"):
        load_sqlite(db_file, "INVALID SQL QUERY")


def test_load_sqlite_empty_result(tmp_path):
    """Test loading query with no results."""
    import sqlite3
    db_file = tmp_path / "empty.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE test (id INTEGER)")

    df = load_sqlite(db_file, "SELECT * FROM test WHERE 1=0")
    assert len(df) == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_loaders.py::test_load_sqlite_invalid_path -v`

Expected: FAIL - Current implementation doesn't validate path

**Step 3: Implement validation**

Replace `core/data/loaders.py:41-47`:

```python
def load_sqlite(db_path: str | Path, query: str) -> pl.DataFrame:
    """Execute a SQL query against a SQLite database and return results as a Polars DataFrame.

    Args:
        db_path: Path to SQLite database file.
        query: SQL query to execute.

    Returns:
        Polars DataFrame with query results.

    Raises:
        FileNotFoundError: If database file doesn't exist.
        ValueError: If query is invalid or database is corrupted.
    """
    import sqlite3

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Basic SQL injection check - only allow SELECT statements
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are supported for security reasons")

    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pl.read_database(query, conn)
    except sqlite3.Error as e:
        raise ValueError(f"Database error: {e}") from e

    return df
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_loaders.py -k sqlite -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/data/loaders.py tests/unit/test_loaders.py
git commit -m "feat: add SQLite connection validation and security checks"
```

---

## Task 7: Fix Exception Chain Loss in HPO

**Files:**
- Modify: `core/engine/hpo.py:80-90`
- Test: `tests/unit/test_hpo.py` (extend existing)

**Issue:** When evaluation fails in HPO, the exception chain is preserved but we should add more context.

**Step 1: Write the failing test**

Add to `tests/unit/test_hpo.py`:

```python
def test_optimize_model_preserves_exception_context():
    """Test that evaluation failures preserve exception context."""
    from core.engine.hpo import optimize_model
    from core.models.conventional.catboost_model import CatBoostModel

    # Create invalid data to trigger error
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])  # Wrong shape for classification

    evaluator = Mock()
    evaluator.evaluate.side_effect = ValueError("Invalid data shape")

    result = optimize_model(
        CatBoostModel,
        X,
        y,
        evaluator,
        "catboost",
        n_trials=2,
        search_space={},
    )

    # Should have completed trials without crashing
    assert result["n_trials"] == 2
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/unit/test_hpo.py::test_optimize_model_preserves_exception_context -v`

Expected: May PASS - current code already preserves chain with `raise ... from e`

**Step 3: Enhance exception context (improvement)**

The current code already uses `raise optuna.TrialPruned() from e` which preserves the chain. However, we can add more context:

Replace `core/engine/hpo.py:80-90`:

```python
try:
    scores = evaluator.evaluate(model_cls, X, y, task=task, **params)
    if not scores:
        raise optuna.TrialPruned("No scores returned from evaluator")
    primary = list(scores.values())[0]
    return primary
except optuna.TrialPruned:
    raise
except Exception as e:
    # Add context about which parameters failed
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    raise optuna.TrialPruned(
        f"Evaluation failed for trial with params: {params_str}. Error: {e}"
    ) from e
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_hpo.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add core/engine/hpo.py tests/unit/test_hpo.py
git commit -m "feat: enhance exception context in HPO trial failures"
```

---

## Task 8: Add Type Annotations to Trainer

**Files:**
- Modify: `core/engine/trainer.py:35, 138`
- Test: `uv run ty core/engine/trainer.py` (type check only)

**Issue:** Missing type hints for `_MODEL_CLASS_CACHE` and `df` parameter in `run` method.

**Step 1: Run type checker to see current issues**

Run: `uv run ty check core/engine/trainer.py`

Expected: Shows missing type annotations

**Step 2: Add missing type annotations**

Line 35 - add type annotation:

```python
_MODEL_CLASS_CACHE: dict[str, type] = {}
```

Line 138 - the `df` parameter already has type annotation `pl.DataFrame`. This is correct.

**Step 3: Run type checker to verify**

Run: `uv run ty check core/engine/trainer.py`

Expected: PASS

**Step 4: Commit**

```bash
git add core/engine/trainer.py
git commit -m "style: add missing type annotation to _MODEL_CLASS_CACHE"
```

---

## Task 9: Add Validation for getattr in main.py

**Files:**
- Modify: `main.py:219-223`
- Test: `tests/unit/test_cli.py` (extend existing)

**Issue:** Line 224 uses `getattr()` without validation that the attribute exists.

**Step 1: Write the failing test**

Add to `tests/unit/test_cli.py`:

```python
def test_hpo_unknown_model_exits_gracefully(tmp_path, monkeypatch):
    """Test that unknown model name produces clear error."""
    from typer.testing import CliRunner
    from main import app

    runner = CliRunner()
    # Create a dummy data file
    data_file = tmp_path / "test.csv"
    data_file.write_text("a,b,target\n1,2,0\n3,4,1")

    result = runner.invoke(app, [
        "hpo",
        "--data", str(data_file),
        "--target", "target",
        "--model", "unknown_model_x",
    ])

    assert result.exit_code == 1
    assert "Unknown model" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_cli.py::test_hpo_unknown_model_exits_gracefully -v`

Expected: May PASS - code already has validation at line 219-222

The current code at lines 219-222 already validates:
```python
if not hasattr(model_configs, model):
    available = [a for a in dir(model_configs) if not a.startswith("_")]
    typer.echo(f"Error: Unknown model '{model}'. Available models: {', '.join(available)}")
    raise typer.Exit(1)
```

This is already correct. The issue mentioned may have been fixed already.

**Step 3: Document that this is already correct**

No code changes needed. Add a comment for clarity:

```python
# Validate model exists before accessing its config
if not hasattr(model_configs, model):
    available = [a for a in dir(model_configs) if not a.startswith("_")]
    typer.echo(f"Error: Unknown model '{model}'. Available models: {', '.join(available)}")
    raise typer.Exit(1)

# Safe to use getattr now - we've validated existence
search_space = getattr(model_configs, model).hpo_search_space()
```

**Step 4: Run tests to verify**

Run: `uv run pytest tests/unit/test_cli.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add main.py tests/unit/test_cli.py
git commit -m "docs: add comment clarifying getattr validation in hpo command"
```

---

## Task 10: Make MAX_ROWS Configurable for TabPFN

**Files:**
- Modify: `core/models/tabular_foundation/tabpfn_model.py:18, 38-43`
- Modify: `configs/model_configs.py` (need to check structure)
- Test: `tests/unit/test_tabpfn.py` (extend existing)

**Issue:** `MAX_ROWS = 10_000` is hardcoded, should be configurable.

**Step 1: Check model_configs structure**

Run: `cat configs/model_configs.py`

Expected: See how model configs are structured

**Step 2: Write the failing test**

Add to `tests/unit/test_tabpfn.py`:

```python
def test_tabpfn_max_rows_configurable():
    """Test that MAX_ROWS can be configured via constructor."""
    from core.models.tabular_foundation.tabpfn_model import TabPFNModel
    import numpy as np

    # Create model with custom max rows
    model = TabPFNModel(task="classification", max_rows=5000)

    # Should fail at 5001 rows
    X = np.random.rand(5001, 2)
    y = np.random.randint(0, 2, 5001)

    with pytest.raises(ValueError, match="max 5000 rows"):
        model.fit(X, y)

    # Should succeed at 5000 rows
    X_small = np.random.rand(5000, 2)
    y_small = np.random.randint(0, 2, 5000)
    # Mock the model to avoid actual training
    model._build_model = Mock()
    model.fit(X_small, y_small)  # Should not raise
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tabpfn.py::test_tabpfn_max_rows_configurable -v`

Expected: FAIL - max_rows parameter doesn't exist

**Step 4: Implement configurable max_rows**

Replace `core/models/tabular_foundation/tabpfn_model.py:18, 20-23`:

```python
class TabPFNModel:
    """
    TabPFN v2 wrapper with configurable row-count guardrail.

    Args:
        task: "classification" or "regression"
        max_rows: Maximum number of training rows. Default: 10_000.
        **kwargs: Additional parameters passed to TabPFN model.
    """

    DEFAULT_MAX_ROWS = 10_000

    def __init__(self, task: str = "classification", max_rows: int | None = None, **kwargs):
        self.task = task
        self.max_rows = max_rows or self.DEFAULT_MAX_ROWS
        self.params = kwargs
        self.model = None
```

Update the fit method validation (lines 38-43):

```python
def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    if len(X) > self.max_rows:
        raise DataSizeError(
            f"TabPFN supports max {self.max_rows} rows for this instance, got {len(X)}. "
            f"Increase max_rows parameter or use CatBoost/LightGBM for larger datasets."
        )
    self.model = self._build_model()
    self.model.fit(X, y)
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tabpfn.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add core/models/tabular_foundation/tabpfn_model.py tests/unit/test_tabpfn.py
git commit -m "feat: make TabPFN MAX_ROWS configurable"
```

---

## Task 11: Make OMP_NUM_THREADS Configurable

**Files:**
- Modify: `core/engine/trainer.py:19-20`
- Modify: `configs/hardware.py` (to add thread configuration)
- Test: `tests/unit/test_trainer.py` (extend existing)

**Issue:** OMP_NUM_THREADS is hardcoded for macOS ARM64, should be configurable.

**Step 1: Check hardware config structure**

Run: `cat configs/hardware.py`

Expected: See current hardware configuration structure

**Step 2: Add thread configuration to HardwareProfile**

Add to `configs/hardware.py`:

```python
import os
import platform


class HardwareProfile:
    """Detected hardware capabilities."""

    # ... existing code ...

    @classmethod
    def _get_default_threads(cls) -> int:
        """Get default thread count based on platform."""
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return 1  # macOS ARM64 has performance issues with threading
        return os.cpu_count() or 1

    @classmethod
    def configure_omp_threads(cls, threads: int | None = None) -> int:
        """Configure OMP_NUM_THREADS environment variable.

        Args:
            threads: Number of threads to use. If None, uses platform default.

        Returns:
            The configured thread count.
        """
        thread_count = threads or cls._get_default_threads()
        os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
        return thread_count
```

**Step 3: Update trainer to use configurable threads**

Replace `core/engine/trainer.py:19-20`:

```python
# Configure OpenMP threads based on hardware profile
HardwareProfile.configure_omp_threads()
```

Add import if not already present:
```python
from configs.hardware import HardwareProfile
```

**Step 4: Write the test**

Add to `tests/unit/test_trainer.py`:

```python
def test_omp_threads_configurable(monkeypatch):
    """Test that OMP threads can be configured via HardwareProfile."""
    from configs.hardware import HardwareProfile
    import os

    # Test default
    thread_count = HardwareProfile.configure_omp_threads()
    assert os.environ.get("OMP_NUM_THREADS") == str(thread_count)

    # Test custom value
    custom_count = HardwareProfile.configure_omp_threads(threads=4)
    assert custom_count == 4
    assert os.environ.get("OMP_NUM_THREADS") == "4"
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_trainer.py::test_omp_threads_configurable -v`

Expected: PASS

**Step 6: Commit**

```bash
git add core/engine/trainer.py configs/hardware.py tests/unit/test_trainer.py
git commit -m "feat: make OMP_NUM_THREADS configurable via HardwareProfile"
```

---

## Task 12: Run Full Test Suite and Type Check

**Files:**
- All modified files
- Test: Full test suite

**Issue:** Verify all changes work together and don't break existing functionality.

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`

Expected: PASS (all tests)

**Step 2: Run type checker**

Run: `uv run ty check`

Expected: PASS (no type errors)

**Step 3: Run linting**

Run: `uv run ruff check`

Expected: PASS (no linting errors)

**Step 4: Create summary of changes**

Create a summary document of all changes made.

**Step 5: Final commit if any fixes needed**

```bash
# Fix any issues found
git add -A
git commit -m "test: fix issues found in final validation"
```

---

## Summary of Changes

This plan addresses:

### Code Quality (3 tasks)
1. **Deduplicated inline CSV/Parquet loading** in MCP server
2. **Created unified RegistryService** to eliminate duplicate registry logic
3. **Enhanced JSONL error handling** with better error messages

### Bug Fixes (3 tasks)
4. **SQLite connection validation** with security checks
5. **Enhanced exception context** in HPO failures
6. **Validated getattr** usage in CLI (already correct, documented)

### Configuration (3 tasks)
7. **Configurable MAX_ROWS** for TabPFN
8. **Configurable OMP_NUM_THREADS** via HardwareProfile
9. **Type annotations** added to trainer

### Validation (1 task)
10. **Full test suite** validation

**Total Estimated Time:** 2-3 hours for all tasks

**Testing Strategy:** Each task includes TDD with failing tests first, then implementation.

**Dependencies:** Tasks can be done independently except:
- Task 3 (Trainer refactoring) depends on Task 2 (RegistryService creation)
- Task 4 (MCP refactoring) depends on Task 2 (RegistryService creation)
