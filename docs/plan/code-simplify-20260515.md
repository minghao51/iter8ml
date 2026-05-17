# Code Simplification & Refinement Plan

**Date**: 2026-05-15
**Status**: Ready for implementation
**Breaking change**: No — all changes preserve behavior
**Scope**: All code under `src/iter8ml/` (62 Python files)

## Guiding Principle

Explicit over clever. Clear over compact. Every change must preserve identical behavior — outputs, side effects, edge-case handling. When in doubt, leave it alone.

---

## Table of Contents

1. [Priority 1: Dead Code Removal](#priority-1-dead-code-removal-zero-risk)
2. [Priority 2: DRY Violations](#priority-2-dry-violations)
3. [Priority 3: Inconsistency Fixes](#priority-3-inconsistency-fixes)
4. [Priority 4: Readability Improvements](#priority-4-readability-improvements)
5. [Priority 5: Structural (defer)](#priority-5-structural-defer)
6. [Execution Order](#recommended-execution-order)
7. [Verification Checklist](#verification-checklist)

---

## Priority 1: Dead Code Removal (zero risk)

Safe to apply without tests — removing code that is provably unreachable or unused.

### 1.1 Delete `_find_last_run_id()` in `cli/run.py`

**File**: `src/iter8ml/cli/run.py`
**Lines**: 15–24

```python
def _find_last_run_id(config: ExperimentConfig) -> str | None:
    from iter8ml.utils.io import iter_events
    from iter8ml.workspace import Workspace

    ws = Workspace()
    last_run_id: str | None = None
    for e in iter_events(ws.experiments_path):
        if e.get("event") == "experiment_started" and e.get("run_id"):
            last_run_id = e["run_id"]
    return last_run_id
```

**Why**: This function is defined but never called anywhere in the codebase. The `run` command does not use it, and no other module imports it. It appears to be leftover from a previous `--resume` implementation that was replaced.

**Action**: Delete lines 15–24 entirely. No imports to clean up (all are local to the function body).

---

### 1.2 Delete `GPUUnavailableError` in `engine/models/tabpfn_model.py`

**File**: `src/iter8ml/engine/models/tabpfn_model.py`
**Lines**: 15–16

```python
class GPUUnavailableError(RuntimeError):
    pass
```

**Why**: This exception class is defined but never raised anywhere in the codebase. `TabPFNModel._resolve_device()` logs a warning and falls back to CPU — it never raises `GPUUnavailableError`. No other module imports or references it.

**Action**: Delete lines 15–16. The `DataSizeError` class (lines 11–12) must be kept — it IS raised in `fit()`.

---

### 1.3 Remove 3rd element from `_FLAT_DELEGATES` tuples in `config.py`

**File**: `src/iter8ml/config.py`
**Lines**: 118–137

Current:

```python
_FLAT_DELEGATES: dict[str, tuple[str, str, str | None]] = {
    "embedding_method": ("embedding", "method", None),
    "embedding_dim": ("embedding", "dim", None),
    ...
}
```

Every tuple's 3rd element is always `None`. It is unpacked in two places:

- `__getattr__` (line 202): `cfg_attr, field_attr, _ = _FLAT_DELEGATES[name]` — discarded with `_`
- `nest_flat_config_fields` (line 285): `for flat_key, (cfg_key, field_key, _) in _FLAT_DELEGATES.items()` — discarded

**Why**: The 3rd field appears to have been planned for a "type" or "default" hint but was never implemented. It adds noise to every tuple and every unpacking site.

**Action**:
1. Change the type to `dict[str, tuple[str, str]]`
2. Remove the `None` from every tuple: `"embedding_method": ("embedding", "method")`
3. Update unpacking at line 202: `cfg_attr, field_attr = _FLAT_DELEGATES[name]`
4. Update unpacking at line 285: `for flat_key, (cfg_key, field_key) in _FLAT_DELEGATES.items():`

---

### 1.4 Remove no-op hook methods in `engine/pipelines/hooks/tracking_hook.py`

**File**: `src/iter8ml/engine/pipelines/hooks/tracking_hook.py`

`run_before_node_execution` and `run_after_node_execution` are defined but their bodies are empty (just `pass` or no-op). They exist to satisfy the Hamilton hook interface but do nothing.

**Action**: If the hook class has other methods that ARE functional, keep the class but remove the no-op methods. Hamilton's hook protocol uses duck typing — missing methods are simply not called. If ALL methods in the class are no-ops, consider deleting the entire file and its import.

**Verify**: Read the file first to confirm which methods are no-ops vs. functional.

---

### 1.5 Remove unreachable query length check in `data/loader.py`

**File**: `src/iter8ml/data/loader.py`
**Lines**: 90–91

```python
if len(query_stripped) < 7:
    raise ValueError("Invalid SELECT query")
```

**Why**: Line 67 already checks `if not query_upper.startswith("SELECT")`. `"SELECT"` is 6 characters. After `.strip()`, any string that starts with `"SELECT"` is at minimum 6 characters. The `< 7` check is redundant (6-char `"SELECT"` without anything else would be caught by the earlier check producing an empty result set, not a security issue). Additionally, `len("SELECT * FROM t")` is 14 — any real query passes.

**Action**: Delete lines 90–91.

---

### 1.6 Remove redundant `_estimator_type` class attribute in `data/features.py`

**File**: `src/iter8ml/data/features.py`
**Line**: 147

```python
class _SKLearnAdapter:
    _estimator_type = "classifier"  # line 147 — class-level default

    def __init__(self, model, task, *, classes=None):
        ...
        if task != "classification":
            self._estimator_type = "regressor"  # line 153 — always overrides
```

**Why**: `__init__` always sets `self._estimator_type` based on `task`. The class-level `_estimator_type = "classifier"` is never read without being overwritten first (sklearn checks `isinstance` at call time, after `__init__` has run).

**Action**: Remove line 147. The `__init__` assignment at line 153 becomes the sole source of truth. Add `self._estimator_type = "classifier"` as the default at the top of `__init__`, then conditionally override for regression:

```python
def __init__(self, model, task, *, classes=None):
    self._model = model
    self.classes_ = classes
    self._estimator_type = "classifier" if task == "classification" else "regressor"
```

---

## Priority 2: DRY Violations

### 2.1 Consolidate `apply_overrides` across 5 model files

**Files involved**:
- `src/iter8ml/engine/models/gbdt_base.py:34-36` — `self.params.update(overrides)`
- `src/iter8ml/engine/models/catboost_model.py:56-58` — identical
- `src/iter8ml/engine/models/tabpfn_model.py:54-56` — identical
- `src/iter8ml/engine/models/tabnet_model.py:67-69` — identical
- `src/iter8ml/engine/models/ft_transformer.py:82-89` — DIFFERENT (updates config attrs)

**The issue**: `CatBoostModel`, `TabPFNModel`, and `TabNetModel` are standalone classes (not inheriting from `BaseGBDTModel`). All four have identical `apply_overrides` that just does `self.params.update(overrides)`. `BaseGBDTModel` also has it (inherited by LightGBM and XGBoost).

**Action**: Create a mixin:

```python
# In a new file or at the top of engine/models/base.py (which already exists as a Protocol)

class ParamsMixin:
    """Provides apply_overrides for models that store hyperparams in self.params."""
    params: dict[str, Any]

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        self.params.update(overrides)
```

Then:
- Add `ParamsMixin` to the MRO of `CatBoostModel`, `TabPFNModel`, `TabNetModel`
- Keep `BaseGBDTModel.apply_overrides` as-is (it's already a base class, removing would break subclasses)
- OR have `BaseGBDTModel` inherit from `ParamsMixin` and remove its own `apply_overrides`
- Leave `FTTransformerModel.apply_overrides` alone — it has different semantics (updates config attrs, falls back to instance attrs)

**Alternative (less invasive)**: Don't create a mixin. Just leave the 4 identical one-liners as-is. The DRY violation is minor (1 line each) and a mixin adds indirection. Pick based on your judgment.

---

### 2.2 Extract OOV methods into a mixin in `engine/models/sparse_embedder.py`

**File**: `src/iter8ml/engine/models/sparse_embedder.py`

`EntityEmbedding` (lines 68–79) and `TabularDAE` (lines 145–155) both define identical:

```python
def _init_oov_buffers(self) -> None:
    self._oov_means: dict[str, torch.Tensor] = {}
    for col in self._column_order:
        self.register_buffer(f"_oov_mean_{col}", torch.zeros(self.embedding_dim))

def _update_oov_means(self) -> None:
    for col in self._column_order:
        buf = getattr(self, f"_oov_mean_{col}")
        buf.data.copy_(self.embeddings[col].weight.data.mean(dim=0))
```

**Action**: Extract to a mixin:

```python
class _OOVEmbeddingMixin:
    """Shared OOV buffer management for embedding models."""
    _column_order: list[str]
    embedding_dim: int
    embeddings: nn.ModuleDict

    def _init_oov_buffers(self) -> None: ...

    def _update_oov_means(self) -> None: ...
```

Both `EntityEmbedding` and `TabularDAE` inherit from this mixin. The mixin methods are identical to the current implementations — just move them.

**Caveat**: Both classes already inherit from `nn.Module`. The mixin must NOT inherit from `nn.Module` itself (to avoid MRO issues). It relies on the host class having `register_buffer` and `getattr` via `nn.Module`.

---

### 2.3 Extract shared event-writing logic in `engine/hpo.py`

**File**: `src/iter8ml/engine/hpo.py`
**Lines**: 141–163 (`_log_hpo_trial`) and 165–190 (`_log_warning_event`)

Both functions share this pattern:

```python
if tracker is not None:
    tracker.log_event(event)
    return
if log_path is None:
    return
path = Path(log_path)
path.parent.mkdir(parents=True, exist_ok=True)
with _hpo_file_lock, open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(event) + "\n")
```

**Action**: Extract a `_write_event` helper:

```python
def _write_event(event: dict[str, Any]) -> None:
    if tracker is not None:
        tracker.log_event(event)
        return
    if log_path is None:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _hpo_file_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
```

Both `_log_hpo_trial` and `_log_warning_event` call `_write_event(event)` instead of duplicating the logic. Note: `tracker` and `log_path` are closure variables from `optimize_model` — the helper should also be a closure or accept them as parameters.

**Cleanest approach**: Make `_write_event` accept explicit args:

```python
def _write_event(
    event: dict[str, Any],
    tracker: Tracker | None,
    log_path: str | None,
) -> None:
    ...
```

---

### 2.4 Use `np.percentile` in `engine/hpo_importance.py`

**File**: `src/iter8ml/engine/hpo_importance.py`
**Lines**: 128–129 and 139–140

Current manual quartile computation:

```python
q25 = sorted_vals[len(sorted_vals) // 4]
q75 = sorted_vals[3 * len(sorted_vals) // 4]
```

**Action**: Replace with:

```python
q25, q75 = np.percentile(sorted_vals, [25, 75])
```

This is functionally equivalent for arrays with >4 elements and more readable. For int arrays, cast the result: `q25 = int(q25)` and `q75 = int(q75)` where needed.

---

## Priority 3: Inconsistency Fixes

### 3.1 Consolidate LLM env var naming

**Files involved**:
- `src/iter8ml/config.py:190` — uses `ITER8ML_LLM_MODEL`
- `src/iter8ml/engine/state_observer.py:41` — uses `TABBLUEPRINT_LLM_MODEL`
- `src/iter8ml/services/llm.py:25` — uses `TABBLUEPRINT_LLM_MODEL`

**Current behavior**: `config.py` reads one env var, while `state_observer.py` and `llm.py` read a different one. If a user sets only `ITER8ML_LLM_MODEL`, the LLM agent used by state_observer will fall through to `DEFAULT_LLM_MODEL` instead of reading the user's override. This is a silent bug.

**Action**: Change both `state_observer.py:41` and `llm.py:25` to use `ITER8ML_LLM_MODEL`:

```python
# state_observer.py line 41
return os.getenv("ITER8ML_LLM_MODEL", DEFAULT_LLM_MODEL)

# llm.py line 25
default_factory=lambda: os.getenv("ITER8ML_LLM_MODEL", DEFAULT_LLM_MODEL)
```

**Migration note**: If any users have `TABBLUEPRINT_LLM_MODEL` set, they'll need to switch to `ITER8ML_LLM_MODEL`. Consider a brief deprecation period where both are checked:

```python
return os.getenv("ITER8ML_LLM_MODEL", os.getenv("TABBLUEPRINT_LLM_MODEL", DEFAULT_LLM_MODEL))
```

---

### 3.2 Fix `selector.py` docstring to match code behavior

**File**: `src/iter8ml/engine/models/selector.py`
**Lines**: 7–17 (docstring) vs 44–47 (code)

**Docstring says**:
```
n_rows < 50k + GPU   -> [TabPFN, CatBoost, LightGBM]
n_rows < 50k no GPU  -> [CatBoost, LightGBM, XGBoost]
```

**Code actually does** (line 44–47):
```python
# TabPFN is now always included if GPU is present, regardless of row count.
if has_gpu:
    models.append("tabpfn")
```

TabPFN is included whenever GPU is available, regardless of `n_rows`. The row-count guard is handled inside `TabPFNModel.fit()` which raises `DataSizeError`.

**Action**: Update the docstring to reflect actual behavior:

```
GPU present           -> TabPFN always included (row limit enforced inside model.fit())
n_rows < 500k         -> [CatBoost, LightGBM, XGBoost]
n_rows >= 500k        -> [LightGBM, XGBoost]
vram_gb > 12 & n>=50k -> + FT-Transformer
vram_gb > 8           -> + TabNet
```

---

### 3.3 Replace `np.random.RandomState` with `np.random.default_rng`

**Files involved**:
- `src/iter8ml/data/leakage.py:62` — `rng_local = np.random.RandomState(42)`
- `src/iter8ml/analysis/domain_classifier.py:57` — `rng = np.random.RandomState(self.random_seed)`

**Why**: `RandomState` is the legacy NumPy random API. `default_rng` is the modern replacement. Both files use the RNG for shuffling/sampling, which has equivalent methods.

**Action for `leakage.py:62`**:

```python
# Before
rng_local = np.random.RandomState(42)
rng_local.shuffle(X_col[:, col_idx])

# After
rng_local = np.random.default_rng(42)
rng_local.shuffle(X_col[:, col_idx])
```

Note: `Generator.shuffle` works in-place like `RandomState.shuffle` — identical behavior.

**Action for `domain_classifier.py:57`**:

```python
# Before
rng = np.random.RandomState(self.random_seed)
idx = rng.choice(len(ref_np), self.max_rows, replace=False)

# After
rng = np.random.default_rng(self.random_seed)
idx = rng.choice(len(ref_np), self.max_rows, replace=False)
```

Note: `Generator.choice` signature is the same for this use case.

---

### 3.4 Remove `from_task_type()` helper from `constants.py`

**File**: `src/iter8ml/constants.py:57-61`

```python
def from_task_type(value: str | TaskType) -> TaskType:
    if isinstance(value, TaskType):
        return value
    return TaskType(value)
```

**Why**: `TaskType(value)` already handles both cases:
- `TaskType("classification")` → `TaskType.CLASSIFICATION`
- `TaskType(TaskType.CLASSIFICATION)` → `TaskType.CLASSIFICATION`

The `isinstance` check in `from_task_type` is redundant — Python's `Enum(value)` returns the enum member if `value` is already a member of that enum.

**Callers to update**:
- `src/iter8ml/cli/run.py:65` — `from_task_type(task)` → `TaskType(task)`
- `src/iter8ml/engine/hpo.py:81` — `from_task_type(task)` → `TaskType(task)`
- Any other imports of `from_task_type`

**Action**: Delete the function. Search for all `from_task_type` usages and replace with `TaskType(...)`. Remove the import from `__init__.py` if re-exported.

---

### 3.5 Fix XGBoost params mutation bug in `engine/models/xgboost_model.py`

**File**: `src/iter8ml/engine/models/xgboost_model.py:26`

```python
"seed": self.params.pop("random_seed", 42),
```

**Why**: `dict.pop()` mutates `self.params`. If `_build_params()` is called twice (e.g., during HPO re-evaluation), the second call will not find `random_seed` in `self.params` and will always use `42`, ignoring any override.

**Action**: Replace `pop` with `get`:

```python
"seed": self.params.get("random_seed", 42),
```

This preserves the key in `self.params` for subsequent calls. The random_seed is not needed in the params dict passed to XGBoost (it's extracted as `seed`), so there's no conflict.

---

### 3.6 Remove redundant `getattr` guards in `engine/models/gbdt_base.py`

**File**: `src/iter8ml/engine/models/gbdt_base.py:58,64`

```python
def _classify_predictions(self, preds):
    n_cls = getattr(self, "_n_classes", 2)  # line 58

def _format_proba(self, preds):
    n_cls = getattr(self, "_n_classes", 2)  # line 64
```

**Why**: `BaseGBDTModel.__init__` (line 21) always sets `self._n_classes = kwargs.pop("n_classes", 0)`. And `fit()` (line 42) updates it: `self._n_classes = self._n_classes or int(labels.size)`. By the time `_classify_predictions` or `_format_proba` are called (post-fit), `self._n_classes` is always set.

The `getattr` with a default of `2` masks potential bugs — if `_n_classes` were somehow unset, you'd get silent misclassification instead of an `AttributeError`.

**Action**: Replace both with direct attribute access:

```python
n_cls = self._n_classes
```

---

## Priority 4: Readability Improvements

### 4.1 Rewrite SQL sanitizer in `data/loader.py`

**File**: `src/iter8ml/data/loader.py`
**Lines**: 64–88

Current approach strips 15 SQL keywords one-by-one with chained `.replace()` calls:

```python
upper_no_select = query_upper.replace("SELECT", "").replace("FROM", "").replace("WHERE", "")
upper_no_select = upper_no_select.replace("AND", "").replace("OR", "").replace("NOT", "")
# ... 12 more .replace() calls
for keyword in blocked_keywords:
    if keyword in upper_no_select:
        raise ValueError(...)
```

**Problems**:
1. Fragile: removing `"IN"` also removes the `"IN"` from `"INNER"` or `"DISTINCT"` (though these are removed separately first, the ordering matters and is brittle)
2. Variable shadowing: `stripped_query` (line 71) shadows outer `query_stripped` (line 64)
3. Hard to reason about what passes and what doesn't

**Action**: Replace with an allowlist approach using word-boundary matching:

```python
import re

_ALLOWED_SQL_TOKENS = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "JOIN", "LEFT", "RIGHT",
    "INNER", "OUTER", "ON", "GROUP", "ORDER", "BY", "HAVING", "LIMIT", "AS",
    "IN", "IS", "NULL", "LIKE", "BETWEEN", "DISTINCT", "ASC", "DESC",
    "UNION", "ALL", "CASE", "WHEN", "THEN", "ELSE", "END", "EXISTS",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "CAST",
})

_BLOCKED_KEYWORDS = frozenset({"DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC", "EXECUTE"})

def _validate_query(query_stripped: str) -> None:
    query_upper = query_stripped.upper()
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are supported for security reasons")
    if ";" in query_stripped.rstrip(";"):
        raise ValueError("Multiple statements are not supported for security reasons")
    tokens = set(re.findall(r"[A-Z]+", query_upper))
    blocked = tokens & _BLOCKED_KEYWORDS
    if blocked:
        raise ValueError(f"Destructive keywords not allowed: {sorted(blocked)}")
```

This is:
- More readable (token extraction is a single regex)
- Not order-dependent (unlike chained replaces)
- Extensible (add tokens to the sets)

---

### 4.2 Break up `_render_state` in `engine/state_observer.py`

**File**: `src/iter8ml/engine/state_observer.py`
**Lines**: 63–194 (131 lines)

**Why**: `_render_state` is a 130-line method that builds a markdown document section by section. Each section (leaderboard, leakage, target transform, AFE, SHAP, drift, registry, pipeline DAG) is an independent block of 10–20 lines. The method is hard to scan and hard to test individually.

**Action**: Extract each section into its own method:

```python
def _render_state(self, report: ExperimentReport) -> str:
    latest = report.latest_run
    if latest is None:
        raise ValueError("Cannot render state: report has no latest_run")

    lines = [
        "# Current Experiment State\n",
        f"**Task:** {latest.task.title()}",
        f"**Dataset:** {latest.dataset}",
        f"**Rows / Features:** {latest.n_rows} / {latest.n_features}",
        f"**Latest Run ID:** {latest.run_id}\n",
    ]
    lines.extend(self._render_leaderboard_section(report))
    lines.extend(self._render_resource_section(latest))

    all_events = self._load_all_events()
    lines.extend(self._render_leakage_section(all_events))
    lines.extend(self._render_target_transform_section(all_events))
    lines.extend(self._render_afe_section(all_events))
    lines.extend(self._render_shap_section(all_events))

    if self._llm_enabled and (any(e.get("event") == "shap_explainability" for e in all_events) or report.latest_run):
        lines.extend(self._render_llm_commentary(all_events, report))

    lines.extend(self._render_drift_section(all_events))
    lines.extend(self._render_registry_section(report))
    lines.extend(self._render_pipeline_section())

    return "\n".join(lines) + "\n"

def _render_leaderboard_section(self, report: ExperimentReport) -> list[str]:
    lines = [
        "## Leaderboard\n",
        "| Rank | Model | Run ID | Primary Metric | Score | Duration |",
        "|---|---|---|---|---|---|",
    ]
    for index, entry in enumerate(report.leaderboard, start=1):
        lines.append(
            f"| {index} | {entry.model} | {entry.run_id} | {entry.primary_metric} "
            f"| {entry.primary_score:.4f} | {entry.duration_seconds}s |"
        )
    return lines

# ... one method per section
```

Each `_render_*` method returns `list[str]` and is independently testable.

---

### 4.3 Refactor `EmbeddingEngine.__init__` to accept `EmbeddingConfig`

**File**: `src/iter8ml/data/embedding.py`
**Lines**: 118–150

Current:

```python
class EmbeddingEngine:
    def __init__(
        self,
        task: str,
        workspace: Workspace,
        embedding_method: str = "entity",
        embedding_dim: int = 16,
        embedding_max_categories: int = 50,
        embedding_epochs: int = 10,
        embedding_lr: float = 1e-3,
        embedding_mlp_width: int = 128,
        embedding_mlp_depth: int = 2,
        embedding_ae_latent_dim: int = 32,
        embedding_ae_dropout: float = 0.2,
        random_seed: int = 42,
    ):
        self._task = task
        self._workspace_dir = workspace.root
        self._embedding_method = embedding_method
        self._embedding_dim = embedding_dim
        # ... 7 more self._embedding_* assignments
```

`EmbeddingConfig` already exists in `config.py:31-42` with all these fields.

**Action**:

```python
class EmbeddingEngine:
    def __init__(
        self,
        task: str,
        workspace: Workspace,
        config: EmbeddingConfig | None = None,
        random_seed: int = 42,
    ):
        self._task = task
        self._workspace_dir = workspace.root
        self._config = config or EmbeddingConfig()
        self._random_seed = random_seed
        # ... remove all self._embedding_* lines
```

Then replace all internal references:
- `self._embedding_method` → `self._config.method`
- `self._embedding_dim` → `self._config.dim`
- `self._embedding_max_categories` → `self._config.max_categories`
- etc.

**Caller update**: The caller (in `engine/pipelines/nodes/features.py` or similar) currently unpacks the config fields individually. Update to pass the `EmbeddingConfig` object directly.

---

### 4.4 Fix closure hack in `extract_cat_codes` in `data/embedding.py`

**File**: `src/iter8ml/data/embedding.py`
**Lines**: 71–74

```python
def _map_val(v: Any, _m: dict[Any, int] = mapping) -> int:
    return _m.get(v, 0)

code_series = series.map_elements(_map_val, return_dtype=pl.Int64)
```

**Why**: The `_m=mapping` default-arg pattern is a well-known Python closure gotcha. It works here but is confusing to read.

**Action**: Use a lambda with explicit capture:

```python
code_series = series.map_elements(lambda v: mapping.get(v, 0), return_dtype=pl.Int64)
```

Or if performance matters (called in a loop), keep the function but rename for clarity:

```python
def _make_mapper(m: dict[Any, int]):
    return lambda v: m.get(v, 0)

code_series = series.map_elements(_make_mapper(mapping), return_dtype=pl.Int64)
```

---

### 4.5 Tighten `track_errors` exception classification in `exceptions.py`

**File**: `src/iter8ml/exceptions.py`
**Lines**: 42–46

```python
def _classify(exc: Exception) -> type[TabularBlueprintError]:
    msg = str(exc).lower()
    if isinstance(exc, ValueError) and any(kw in msg for kw in _DATA_KEYWORDS):
        return DataLoadError
    return ModelFitError
```

**Why**: This is overbroad. Any `ValueError` with the word "data" in its message becomes `DataLoadError`, even if it came from model training (e.g., "invalid data shape for this model"). The keyword matching is too loose.

**Action**: Either:
1. Tighten the keyword list to only match errors from specific call sites
2. Or add the originating module/function to the classification
3. Or simply classify ALL `ValueError` as `DataLoadError` (since most ValueErrors in this codebase are data-related) and document the decision

The safest minimal change is to narrow `_DATA_KEYWORDS`:

```python
_DATA_KEYWORDS = frozenset({"target_col", "file not found", "unsupported file format", "invalid json"})
```

This matches only the specific error messages raised by the loader/adapter rather than catching any ValueError mentioning "data".

---

### 4.6 Fix variable shadowing in `data/loader.py`

**File**: `src/iter8ml/data/loader.py`
**Lines**: 64, 71

```python
query_stripped = query.strip()       # line 64
...
stripped_query = query_stripped.rstrip(";").strip()  # line 71 — shadows nothing but confusing name
```

**Action**: Rename for clarity:

```python
query_text = query.strip()                          # line 64
...
query_clean = query_text.rstrip(";").strip()        # line 71
```

Update all references in the function body.

---

## Priority 5: Structural (defer)

These are larger refactors that provide clarity but carry higher risk. Defer unless the codebase is actively causing problems.

### 5.1 Hamilton fallback duplication in pipeline nodes

**Files**: `engine/pipelines/nodes/prep.py`, `features.py`, `drift_detection.py`

Each file defines functions twice: once under `if _HAS_HAMILTON:` and once under `else:` (the else branch typically raises `ImportError`). This doubles the file size. A better approach would be to gate the module import itself and provide a single clear error at import time.

### 5.2 Extract export template from `services/export.py`

**File**: `services/export.py`

`PREDICTOR_TEMPLATE` is a 129-line f-string embedded in Python. Should be a `.py.j2` or `.py.tpl` template file loaded at runtime.

### 5.3 Simplify `nest_flat_config_fields` in `config.py`

**File**: `config.py:216-292`

79 lines with a nested `_upsert` closure. Could be simplified with a declarative mapping table that maps legacy keys to `(step_name, field_name, transform_fn)` tuples, then iterated in a loop.

---

## Recommended Execution Order

Apply in this order to minimize conflicts and maximize safety:

| Step | Items | Risk | Estimated Time |
|------|-------|------|----------------|
| 1 | P1.1–P1.6 (dead code) | Zero | 15 min |
| 2 | P3.5 (XGBoost mutation bug) | Low | 2 min |
| 3 | P3.1 (env var consolidation) | Low | 5 min |
| 4 | P3.4 (remove `from_task_type`) | Low | 5 min |
| 5 | P3.6 (remove getattr guards) | Low | 2 min |
| 6 | P3.2 (fix selector docstring) | Zero | 2 min |
| 7 | P3.3 (replace RandomState) | Low | 5 min |
| 8 | P2.3 (extract _write_event) | Low | 10 min |
| 9 | P2.4 (np.percentile) | Low | 3 min |
| 10 | P2.1 (apply_overrides consolidation) | Low | 15 min |
| 11 | P2.2 (OOV mixin) | Medium | 15 min |
| 12 | P4.1 (SQL sanitizer rewrite) | Medium | 15 min |
| 13 | P4.6 (variable shadowing) | Low | 2 min |
| 14 | P4.4 (closure hack) | Low | 3 min |
| 15 | P4.2 (state_observer breakdown) | Medium | 20 min |
| 16 | P4.3 (EmbeddingEngine config) | Medium | 15 min |
| 17 | P4.5 (tighten error classification) | Medium | 10 min |

---

## Verification Checklist

After each priority group, run:

1. **Lint**: `uv run ruff check src/iter8ml/`
2. **Type check**: `uv run mypy src/iter8ml/` (or `pyright`)
3. **Tests**: `uv run pytest tests/ -x`
4. **Manual smoke test** (if no tests cover a changed file):
   ```bash
   uv run python -c "from iter8ml.engine.models.factory import available_model_names; print(available_model_names())"
   uv run python -c "from iter8ml.config import ExperimentConfig; print('OK')"
   uv run python -c "from iter8ml.data.loader import load_data; print('OK')"
   ```

### Specific regression risks:

| Change | What could break | How to verify |
|--------|-----------------|---------------|
| P3.5 (XGBoost pop→get) | `random_seed` might end up in params passed to `xgb.train` | Run XGBoost model fit, check that `seed` is set correctly and `random_seed` is ignored by XGBoost |
| P3.1 (env var rename) | Users with `TABBLUEPRINT_LLM_MODEL` set | Grep codebase for all `TABBLUEPRINT` references, ensure none remain |
| P3.4 (remove from_task_type) | Any caller passing `TaskType` enum | Verify `TaskType(TaskType.CLASSIFICATION)` returns the correct member |
| P2.2 (OOV mixin) | MRO issues with `nn.Module` | Run embedding training end-to-end |
| P4.1 (SQL sanitizer) | Previously-valid queries might be rejected | Test with edge cases: subqueries, CTEs, quoted identifiers |

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `cli/run.py` | Delete `_find_last_run_id`, remove `from_task_type` import |
| `config.py` | Simplify `_FLAT_DELEGATES` tuples |
| `constants.py` | Delete `from_task_type()` |
| `exceptions.py` | Tighten `_DATA_KEYWORDS` |
| `data/loader.py` | Rewrite SQL sanitizer, fix variable shadowing, remove dead length check |
| `data/embedding.py` | Accept `EmbeddingConfig`, fix closure hack |
| `data/features.py` | Fix `_SKLearnAdapter._estimator_type` |
| `data/leakage.py` | Replace `RandomState` with `default_rng` |
| `engine/models/gbdt_base.py` | Remove `getattr` guards, possibly use ParamsMixin |
| `engine/models/catboost_model.py` | Possibly use ParamsMixin |
| `engine/models/tabpfn_model.py` | Delete `GPUUnavailableError`, possibly use ParamsMixin |
| `engine/models/tabnet_model.py` | Possibly use ParamsMixin |
| `engine/models/ft_transformer.py` | No change (different `apply_overrides` semantics) |
| `engine/models/xgboost_model.py` | Fix `pop` → `get` |
| `engine/models/selector.py` | Update docstring |
| `engine/models/sparse_embedder.py` | Extract OOV mixin |
| `engine/models/baselines.py` | No change |
| `engine/hpo.py` | Extract `_write_event`, remove `from_task_type` import |
| `engine/hpo_importance.py` | Use `np.percentile` |
| `engine/state_observer.py` | Break up `_render_state`, fix env var |
| `engine/pipelines/hooks/tracking_hook.py` | Remove no-op methods |
| `services/llm.py` | Fix env var |
| `analysis/domain_classifier.py` | Replace `RandomState` with `default_rng` |
