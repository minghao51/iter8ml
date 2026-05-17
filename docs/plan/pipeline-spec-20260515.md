# PipelineSpec + Expanded @config.when Implementation Plan

**Date**: 2026-05-15
**Status**: Approved, ready for implementation
**Breaking change**: Yes — removes legacy step-level bool fields from ExperimentConfig

## Motivation

The current `ExperimentConfig` is a flat bag of ~30 fields. Step enable/disable is scattered across
bool guards (`run_quality_audit`, `run_leakage_audit`, etc.) that live as top-level config fields.
There is no user-facing ordered step list — pipeline ordering is implicit via Hamilton's dependency
resolution. Users cannot see, skip, or reorder steps from config.

**Goal**: Give users sklearn-like explicit step ordering in config while keeping Hamilton DAG execution.
Make the pipeline composable, skippable, and inspectable.

## Design

### New Models (`config.py`)

```python
class StepName(str, Enum):
    DATA_PREP = "data_prep"
    QUALITY_AUDIT = "quality_audit"
    LEAKAGE_AUDIT = "leakage_audit"
    TARGET_TRANSFORM = "target_transform"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"
    HPO = "hpo"

class PipelineStep(BaseModel):
    name: StepName
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

class PipelineSpec(BaseModel):
    steps: list[PipelineStep] = Field(default_factory=PipelineSpec._default_steps)

    @staticmethod
    def _default_steps() -> list[PipelineStep]:
        return [
            PipelineStep(name=StepName.DATA_PREP),
            PipelineStep(name=StepName.QUALITY_AUDIT),
            PipelineStep(name=StepName.LEAKAGE_AUDIT),
            PipelineStep(name=StepName.TARGET_TRANSFORM),
            PipelineStep(name=StepName.FEATURE_ENGINEERING),
            PipelineStep(name=StepName.MODEL_TRAINING),
            PipelineStep(name=StepName.CALIBRATION),
            PipelineStep(name=StepName.EVALUATION),
        ]

    def is_enabled(self, name: StepName) -> bool:
        return any(s.name == name and s.enabled for s in self.steps)

    def step_params(self, name: StepName) -> dict[str, Any]:
        for s in self.steps:
            if s.name == name:
                return s.params
        return {}
```

### ExperimentConfig Changes

**Add:**
- `pipeline: PipelineSpec = Field(default_factory=PipelineSpec)`

**Remove** (step-level fields → absorbed into `PipelineStep.params`):
- `run_quality_audit` → `PipelineStep(name=QUALITY_AUDIT, enabled=...)`
- `run_leakage_audit` → `PipelineStep(name=LEAKAGE_AUDIT, enabled=...)`
- `auto_clean_noise` → `PipelineStep(name=QUALITY_AUDIT, params={auto_clean_noise: ...})`
- `noise_quality_threshold` → `PipelineStep(name=QUALITY_AUDIT, params={...})`
- `target_transform` → `PipelineStep(name=TARGET_TRANSFORM, params={method: ...})`
- `target_skewness_threshold` → `PipelineStep(name=TARGET_TRANSFORM, params={...})`
- `calibration` → `PipelineStep(name=CALIBRATION, params={method: ...})`
- `feature_strategy` → `PipelineStep(name=FEATURE_ENGINEERING, params={strategy: ...})`

**Update** `_FLAT_DELEGATES`:
- Remove all entries for deleted fields
- `embedding_*` and `afe_*` entries remain (they're params for their respective steps)

**Keep unchanged:**
- `name`, `task`, `target_col`, `data_path`, `cv_folds`, `cv_strategy`, `models`, `random_seed`, `metrics`
- `hpo: HPOConfig`, `quality: QualityConfig`, `afe: AFEConfig`, `embedding: EmbeddingConfig`
- `tracker`, `max_workers`, `data_sample`, `shap_enabled`, `llm_*`, `model_overrides`

### @config.when Variant Expansion

#### `pipelines/nodes/prep.py`

Split `quality_cleaned_df` into variants:

```python
@config.when(run_quality_audit=True)
def quality_cleaned_df__audit(
    validate_target: pl.DataFrame,
    target_col: str,
    auto_clean_noise: bool,
    noise_quality_threshold: float,
) -> tuple[pl.DataFrame, bool, int]:
    # existing logic

@config.when_not(run_quality_audit=True)
def quality_cleaned_df__skip(
    validate_target: pl.DataFrame,
) -> tuple[pl.DataFrame, bool, int]:
    return validate_target, False, 0
```

Split `leakage_report`:

```python
@config.when(run_leakage_audit=True)
def leakage_report__enabled(adapter_result, task) -> LeakageReport | None:
    ...

@config.when_not(run_leakage_audit=True)
def leakage_report__skip(adapter_result, task) -> None:
    return None
```

Split `target_transform_result`:

```python
@config.when(target_transform="none")
def target_transform_result__none(adapter_result) -> tuple:
    _, y = adapter_result
    return y, None, "none", 0.0, 0.0, False

@config.when_not(target_transform="none")
def target_transform_result__transform(adapter_result, target_transform, target_skewness_threshold) -> tuple:
    # existing logic
```

#### `pipelines/nodes/train.py`

Split calibration:

```python
@config.when(calibration="none")
def apply_calibration__none(model, task) -> object:
    return model

@config.when_not(calibration="none")
def apply_calibration__calibrate(model, task, calibration) -> CalibratedModel:
    return CalibratedModel(model, method=calibration)
```

### Executor Rewrite (`pipelines/executor.py`)

Replace hardcoded `_get_training_modules()`:

```python
def _resolve_training_modules(spec: PipelineSpec) -> list[Any]:
    from iter8ml.engine.pipelines.nodes import prep, train
    modules = [prep]
    if spec.is_enabled(StepName.FEATURE_ENGINEERING):
        from iter8ml.engine.pipelines.nodes import features
        modules.append(features)
    modules.append(train)
    return modules

def _resolve_hamilton_config(config: ExperimentConfig) -> dict[str, Any]:
    spec = config.pipeline
    cfg: dict[str, Any] = {
        "run_quality_audit": spec.is_enabled(StepName.QUALITY_AUDIT),
        "run_leakage_audit": spec.is_enabled(StepName.LEAKAGE_AUDIT),
        "target_transform": spec.step_params(StepName.TARGET_TRANSFORM).get("method", "none"),
        "target_skewness_threshold": spec.step_params(StepName.TARGET_TRANSFORM).get("skewness_threshold", 1.0),
        "calibration": spec.step_params(StepName.CALIBRATION).get("method", "none"),
        "feature_strategy": spec.step_params(StepName.FEATURE_ENGINEERING).get("strategy", "none"),
        "auto_clean_noise": spec.step_params(StepName.QUALITY_AUDIT).get("auto_clean_noise", False),
        "noise_quality_threshold": spec.step_params(StepName.QUALITY_AUDIT).get("noise_quality_threshold", 0.5),
    }
    return cfg
```

Update `run_training()`:

```python
def run_training(self, config, df, run_id, vram_gb=0.0, completed_models=None, workspace=None):
    modules = _resolve_training_modules(config.pipeline)
    hamilton_config = _resolve_hamilton_config(config)
    builder = self._driver_mod.Builder().with_modules(*modules).with_config(hamilton_config)
    # ... rest same
```

Simplify `_config_to_inputs()`:
- Remove `run_leakage_audit` from inputs (now in Hamilton config)
- Remove deleted fields from `_DIRECT_FIELDS`

Update `trainer.py`:
- Remove `run_leakage_audit` constructor param — read from `config.pipeline.is_enabled(StepName.LEAKAGE_AUDIT)`

### User-Facing YAML Example

```yaml
name: walkthrough
task: classification
target_col: target
data_path: ""
models: [catboost, lightgbm, xgboost]
cv_folds: 5
metrics: [roc_auc, f1_macro]

pipeline:
  steps:
    - name: data_prep
    - name: quality_audit
      params:
        auto_clean_noise: true
        noise_quality_threshold: 0.5
    - name: leakage_audit
    - name: target_transform
      params:
        method: auto
        skewness_threshold: 1.0
    - name: feature_engineering
      params:
        strategy: afe
        top_k: 15
    - name: model_training
    - name: calibration
      params:
        method: platt
    - name: evaluation
```

To skip: `- name: quality_audit` + `enabled: false`, or omit the step entirely.

### Inspection

Add to `PipelineExecutor`:
- `describe_pipeline(spec)` → returns ordered step list with enabled status
- Update `get_mermaid_graph()` to annotate disabled steps

---

## Implementation Phases

### Phase 1: Add PipelineSpec Models (config.py)

**Files:**
- `src/iter8ml/config.py` — add `StepName`, `PipelineStep`, `PipelineSpec`, `pipeline` field on `ExperimentConfig`
- `src/iter8ml/__init__.py` — export new types

**Verify:** Existing tests still pass (PipelineSpec defaults produce same behavior).

### Phase 2: Expand @config.when Variants (node modules)

**Files:**
- `src/iter8ml/engine/pipelines/nodes/prep.py` — split quality, leakage, target_transform
- `src/iter8ml/engine/pipelines/nodes/train.py` — split calibration

**Verify:** Existing tests still pass (Hamilton config still passes same values).

### Phase 3: Rewrite Executor Module Resolution (executor.py)

**Files:**
- `src/iter8ml/engine/pipelines/executor.py` — `_resolve_training_modules()`, `_resolve_hamilton_config()`

**Verify:** Existing tests still pass (default PipelineSpec → same modules + config).

### Phase 4: Remove Legacy Fields (config.py cleanup)

**Files:**
- `src/iter8ml/config.py` — remove `run_quality_audit`, `run_leakage_audit`, `auto_clean_noise`, `noise_quality_threshold`, `target_transform`, `target_skewness_threshold`, `calibration`, `feature_strategy`; update `_FLAT_DELEGATES`
- `src/iter8ml/engine/pipelines/executor.py` — trim `_DIRECT_FIELDS`
- `src/iter8ml/engine/trainer.py` — remove `run_leakage_audit` param; read from `config.pipeline`

**Verify:** All references to removed fields are gone. Tests updated.

### Phase 5: Update All Tests

**Files affected:**
- `tests/unit/test_config.py` — update for removed fields, add PipelineSpec tests
- `tests/unit/test_data_prep_nodes.py` — update `BASE_INPUTS` to remove deleted fields
- `tests/unit/test_training_nodes.py` — update for calibration variant
- `tests/unit/test_pipeline_executor.py` — update `_DIRECT_FIELDS` sync test, add spec-driven tests
- `tests/unit/test_trainer.py` — remove `run_leakage_audit` param
- `tests/unit/test_session.py` — update config construction
- `tests/unit/test_feature_engineering_nodes.py` — update config flow
- `tests/integration/test_full_pipeline.py` — update config
- `tests/integration/test_dag_execution.py` — update config
- All other tests that construct `ExperimentConfig`

### Phase 6: Update Notebook & Public API

**Files:**
- `notebooks/02_full_walkthrough.qmd` — update config example
- `src/iter8ml/__init__.py` — export `PipelineSpec`, `PipelineStep`, `StepName`
- `src/iter8ml/constants.py` — keep `FeatureStrategy` enum (valid values for step params)

### Phase 7: Inspection & Docs

- Add `describe_pipeline()` to `PipelineExecutor`
- Update `get_mermaid_graph()` to annotate disabled steps
- Update `engine/pipelines/__init__.py` exports

---

## File Change Summary

| File | Change | Phase | Risk |
|------|--------|-------|------|
| `src/iter8ml/config.py` | Add models, add `pipeline` field, remove legacy fields | 1, 4 | Medium |
| `src/iter8ml/constants.py` | No change (keep enums as valid param values) | — | Low |
| `src/iter8ml/engine/pipelines/nodes/prep.py` | Split into `@config.when` variants | 2 | Medium |
| `src/iter8ml/engine/pipelines/nodes/train.py` | Split calibration into variants | 2 | Low |
| `src/iter8ml/engine/pipelines/executor.py` | Spec-driven module resolution | 3, 4 | Medium |
| `src/iter8ml/engine/trainer.py` | Remove `run_leakage_audit` param | 4 | Low |
| `src/iter8ml/__init__.py` | Export new types | 1, 6 | Low |
| `src/iter8ml/engine/pipelines/__init__.py` | Export inspection utilities | 7 | Low |
| `notebooks/02_full_walkthrough.qmd` | Update config example | 6 | Low |
| 10+ test files | Update for removed fields, add new tests | 5 | Low |

---

## Test Plan

| Test | What it verifies |
|------|-----------------|
| `test_pipeline_spec_defaults` | Default spec matches old behavior (all steps enabled) |
| `test_pipeline_spec_from_yaml` | YAML parsing with `pipeline.steps` |
| `test_pipeline_spec_disabled_step` | `enabled: false` → step excluded from modules |
| `test_pipeline_spec_step_params` | Step params flow into Hamilton config dict |
| `test_quality_audit_variant_skip` | `quality_audit: false` → skip variant selected in DAG |
| `test_quality_audit_variant_run` | `quality_audit: true` → audit variant selected |
| `test_leakage_audit_variant_skip` | `leakage_audit: false` → skip variant selected |
| `test_target_transform_variant_none` | `target_transform.method: none` → passthrough variant |
| `test_target_transform_variant_auto` | `target_transform.method: auto` → transform variant |
| `test_calibration_variant_none` | `calibration.method: none` → no calibration |
| `test_calibration_variant_platt` | `calibration.method: platt` → calibrated |
| `test_feature_strategy_variant` | Existing test still passes with new config flow |
| `test_full_pipeline_with_spec` | End-to-end run with custom PipelineSpec |
| `test_describe_pipeline` | `describe_pipeline()` returns correct step list |
| `test_mermaid_annotates_disabled` | Mermaid graph marks disabled steps |
| All existing tests | 60+ tests pass after migration |

---

## Key Design Decisions

1. **PipelineSpec-only** (no legacy field sync) — clean break, user chose this explicitly
2. **StepName enum** — prevents typos, IDE autocomplete, single source of truth
3. **Nested sub-configs remain** (`HPOConfig`, `AFEConfig`, `EmbeddingConfig`, `QualityConfig`) — they're domain configs, not step toggles
4. **`@config.when` variants** — replace `if not enabled: return` guards with structural DAG variants
5. **Default PipelineSpec** produces identical behavior to current system — zero-change migration for users who don't customize
