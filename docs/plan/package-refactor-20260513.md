# iter8ml Package Refactor Plan

**Date:** 2026-05-13
**Status:** Approved
**Goal:** Convert `src/iter8ml` into a programmatic-first experimental platform package for tabular ML, with modern Python packaging, a clean public API, and plugin-ready architecture.

---

## Decisions

1. **Build backend:** Switch from `setuptools` to `hatchling` + `hatch-vcs` for faster builds and dynamic git-based versioning.
2. **Public API:** Programmatic-first. CLI becomes thin wrappers around `ExperimentSession`.
3. **DataFrame library:** Polars-only (consistent with internals).
4. **Backward compatibility:** Aggressively broken. No deprecated shims.
5. **Workspace:** Extract hardcoded `"workspace"` paths into a `Workspace` dataclass.
6. **Plugins:** Entry points added to `pyproject.toml` for `iter8ml.models`. Factory refactored to discover from both built-in registry and entry points.
7. **Config/workspace separation:** `workspace_dir` removed from `ExperimentConfig`. Workspace is purely a runtime concern.
8. **Env var:** `ITER8ML_WORKSPACE` supported as default workspace path.

---

## Phases

### Phase 1: Build System Modernization

**File:** `pyproject.toml`

- Switch `[build-system]` to `hatchling` + `hatch-vcs`
- Add `dynamic = ["version"]`
- Add PyPI metadata: `keywords`, `classifiers`, `project.urls`
- Add `[project.entry-points."iter8ml.models"]` for all built-in models
- Remove `[tool.setuptools.*]` sections
- Add `[tool.hatch.version]` config

### Phase 2: Workspace Abstraction

**New file:** `src/iter8ml/workspace.py`

Create `Workspace` dataclass that owns all filesystem paths:
- `root: Path`
- `experiments_path`, `registry_path`, `artifacts_dir`, `exports_dir`, `state_path`, `leaderboard_path`
- `init()` method to create directories and touch files
- Default factory respects `ITER8ML_WORKSPACE` env var

### Phase 3: Core Refactors (Breaking Changes)

**Files:**
- `src/iter8ml/config.py` — Remove `workspace_dir` field
- `src/iter8ml/engine/trainer.py` — Accept `workspace: Workspace` in `__init__`; remove `typer` dependency
- `src/iter8ml/services/registry.py` — Accept `workspace: Workspace`
- `src/iter8ml/services/export.py` — Accept `workspace: Workspace`
- `src/iter8ml/services/reporting.py` — Accept `workspace: Workspace`
- `src/iter8ml/engine/state_observer.py` — Accept `workspace: Workspace`
- `src/iter8ml/engine/tracker.py` — Add `from_workspace()` factory method

### Phase 4: Plugin-Ready Model Factory

**File:** `src/iter8ml/engine/models/factory.py`

Refactor `_MODEL_REGISTRY` into `_BUILT_IN_REGISTRY` + `_discover_models()` that merges built-ins with `importlib.metadata.entry_points(group="iter8ml.models")`.

Graceful fallback if entry points unavailable.

### Phase 5: Public API

**File:** `src/iter8ml/__init__.py`

Export the full programmatic API: `ExperimentConfig`, `ExperimentSession`, `Workspace`, `Trainer`, `Evaluator`, `Tracker`, `TaskType`, `CVStrategy`, `load_data`, etc.

**New file:** `src/iter8ml/session.py`

Create `ExperimentSession` as the primary high-level interface:
- `__init__(workspace, tracker)`
- `run(config, df)` → delegates to `Trainer`
- `leaderboard(limit)` → returns Polars DataFrame
- `export(key, output_dir)` → delegates to `ExportService`
- `promote(run_id, key)` → delegates to `RegistryService`
- `state(llm_enabled)` → delegates to `StateObserver`

### Phase 6: CLI Demotion

**Files:** `src/iter8ml/cli/*.py`

All CLI commands become thin `typer` wrappers that parse args and delegate to `ExperimentSession`.

Remove business logic from CLI modules.

### Phase 7: Pipeline Updates

**Files:**
- `src/iter8ml/engine/pipelines/executor.py` — Use workspace paths
- `src/iter8ml/engine/pipelines/nodes/train.py` — Use `workspace.artifacts_dir`

### Phase 8: Test Updates

Update constructor signatures in test files (no behavior changes):
- `tests/unit/test_trainer.py`
- `tests/unit/test_registry_service.py`
- `tests/unit/test_export_service.py`
- `tests/unit/test_state_observer.py`
- `tests/integration/test_full_pipeline.py`
- Any test using `ExperimentConfig(workspace_dir=...)`

### Phase 9: Verification

1. `uv build` succeeds
2. `uv run pytest tests/` passes
3. `python -c "import iter8ml; print(iter8ml.__version__)"` works
4. `python -c "from iter8ml import ExperimentSession; s = ExperimentSession()"` works
5. `iter8 init` CLI works
6. `iter8 run config.yaml` CLI works

---

## Before/After Developer Experience

### Before
```python
from iter8ml.config import ExperimentConfig
from iter8ml.engine.trainer import Trainer
from iter8ml.constants import TaskType
import polars as pl

config = ExperimentConfig(
    name="test",
    task=TaskType.CLASSIFICATION,
    target_col="y",
    data_path="data.csv",
    workspace_dir="workspace",
)
trainer = Trainer(config)
df = pl.read_csv("data.csv")
result = trainer.run(df)
```

### After
```python
import iter8ml as iml

config = iml.ExperimentConfig(
    name="test",
    task=iml.TaskType.CLASSIFICATION,
    target_col="y",
    data_path="data.csv",
)
session = iml.ExperimentSession()  # workspace auto-initialized
df = iml.load_data("data.csv")
result = session.run(config, df)
leaderboard = session.leaderboard()
session.export("test:classification")
```

---

## Files Affected Summary

| Status | File | Action |
|--------|------|--------|
| 🔧 Major | `pyproject.toml` | Build system, metadata, entry points |
| 🔧 Major | `src/iter8ml/__init__.py` | Full public API export |
| 🆕 New | `src/iter8ml/workspace.py` | Workspace abstraction |
| 🆕 New | `src/iter8ml/session.py` | Primary programmatic API |
| 🔧 Major | `src/iter8ml/config.py` | Remove workspace_dir field |
| 🔧 Major | `src/iter8ml/engine/trainer.py` | Workspace injection |
| 🔧 Medium | `src/iter8ml/engine/tracker.py` | Workspace factory method |
| 🔧 Medium | `src/iter8ml/engine/pipelines/executor.py` | Workspace paths |
| 🔧 Medium | `src/iter8ml/engine/pipelines/nodes/train.py` | Workspace paths |
| 🔧 Medium | `src/iter8ml/engine/models/factory.py` | Plugin-ready discovery |
| 🔧 Major | `src/iter8ml/services/registry.py` | Workspace injection |
| 🔧 Major | `src/iter8ml/services/export.py` | Workspace injection |
| 🔧 Major | `src/iter8ml/services/reporting.py` | Workspace injection |
| 🔧 Major | `src/iter8ml/engine/state_observer.py` | Workspace injection |
| 🔧 Major | `src/iter8ml/cli/*.py` | Thin wrappers |
| 🧪 Tests | ~10 test files | Signature updates only |
