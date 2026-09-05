# Phase 1 & Phase 2 Handoff Plan — Tabular Blueprint (iter8ml)

> **Audience:** Next agents tackling consolidation and architecture work.
> **Status:** Phase 0 (dead-code deletion + doc drift + small robustness) is **complete** — see git history / summary. Do **not** re-delete anything listed there.
> **Verification baseline (must stay green after your work):**
> ```bash
> uv run ruff check src/iter8ml tests
> uv run mypy src/iter8ml          # 91 source files, currently clean
> uv run pytest tests/unit -q      # 710 passed / 3 skipped
> ```

---

## 0. Context & constraints

The package (`iter8ml`) is a single-node tabular ML framework. Core code lives under `src/iter8ml`. Dependency direction is **acyclic and clean** (verified in audit): `utils`/`constants`/`config` are leaves → `services`→`utils` → `engine`→`services`+`data`+`config` → `orchestration`/`runtime`/`session` on top. The problems are **accumulated dead/parallel abstractions and two parallel pipeline systems**, not cycles.

**Known traps (learned in Phase 0):**
- Tests exercise more than `src/` — e.g. `test_medallion_contracts.py` uses `LocalOrchestrator` (cancellation); `test_pipeline_executor.py` / `test_export_package.py` use `PipelineExecutor.run_preprocessing` and `PipelineMode.EXPORT`. **Do not delete these**; "unused in `src/`" ≠ "dead".
- `safe_load` (bytes) is load-bearing for the security-unpickler tests; only `safe_loads` (str) was truly unused.
- `ModelFitError` and `ArtifactStore` are tested public API — keep.

---

## 1. PHASE 1 — Consolidation (Medium effort, Medium–Large impact)

Goal: collapse parallel abstractions and redundant code paths **without** restructuring the pipeline topology. Each workstream is independently shippable.

### 1.1 Unify the Hamilton driver builder (3 → 1 site)
- **Why:** Driver construction is duplicated in `engine/pipelines/executor.py` (`_try_import_hamilton`, ~L98) and `services/export.py` (~L34 `from hamilton import driver`). A third copy (`runtime/hamilton.py`) was deleted in Phase 0, so today there are exactly two live sites with no shared factory.
- **Files:** `engine/pipelines/executor.py`, `services/export.py`.
- **Approach:** Create one shared `build_driver(*modules, config=None) -> driver` (e.g. `engine/pipelines/_driver.py` or revived `runtime/hamilton.py`) returning the Hamilton `driver.Builder().with_modules(...).with_config(...)`. Both call sites use it. Keep the `HamiltonUnavailableError` raise path in `executor.py`.
- **Verify:** `pytest tests/unit/test_pipeline_executor.py tests/integration/test_export_package.py tests/unit/test_dag_execution.py`; ruff/mypy.

### 1.2 Unify drift detectors behind one protocol (Impact: **L**)
- **Why:** Three detectors implement the same "reference vs live → drift report" shape with **incompatible Pydantic schemas**: `DriftDetector` (KS+chi2) in `analysis/drift.py`, `PSIDriftDetector` in `analysis/psi.py`, `DomainClassifierDriftDetector` in `analysis/domain_classifier.py`. KS is **not** a valid engine config value (`config.py` ~L208) yet `cli/analyze.py` (~L38/L63) advertises it; `engine/pipelines/nodes/drift_detection.py` (L42–117) has 6 near-duplicate node functions differing only by `@when` decorator.
- **Files:** `analysis/drift.py`, `analysis/psi.py`, `analysis/domain_classifier.py`, `engine/pipelines/nodes/drift_detection.py`, `cli/analyze.py`, `services/mcp.py` (`detect_drift`).
- **Approach:**
  1. Define a `DriftDetector` protocol with `detect() -> DriftReport` and a single tagged-union `DriftReport` schema (or a common base).
  2. Add a `ks` branch + config value to `drift_detection.py` so the engine can run KS, and route CLI/MCP through the pipeline node instead of calling `analysis.drift` directly.
  3. Factor the 6 duplicated node functions into one helper dispatched via a loop over `{"psi","domain_classifier","both"}` (or a single `@when` over `drift_method`).
- **Verify:** `pytest tests/unit/test_drift.py tests/unit/test_psi_drift.py tests/unit/test_property_drift.py tests/unit/test_drift_nodes.py tests/unit/test_metamorphic_drift.py`; confirm `ks` flows end-to-end through the engine, not just the CLI.

### 1.3 GBDT config & training unification
- **Why:** `engine/models/model_configs.py` declares GBDT default hyperparams (`CatBoostConfig` ~L8–28, `LightGBMConfig` ~L31–51, `XGBoostConfig` ~L54–72) that are **never read** for training — only `.hpo_search_space()` is consumed (`engine/hpo.py` ~L188), and the GBDT wrappers (`xgboost_model.py`, `lightgbm_model.py`, `catboost_model.py`) hardcode their own defaults. The three wrappers also re-implement objective/metric mapping, `_train_model`, `_create_model`, and `load` with repetitive per-backend differences.
- **Approach:** Derive training defaults from the same source as the HPO search space (or delete the dead `model_configs` fields). Drive objective/metric/train/load differences from a small per-backend descriptor so `gbdt_base.py` owns the shared decode/format logic.
- **Verify:** `pytest tests/unit/test_gbdt_multiclass.py tests/unit/test_models*.py`; ensure no change in trained-model behavior (use a reproducibility/leaderboard snapshot test if available).

### 1.4 Public API hygiene (top-level `__init__`)
- **Why:** `src/iter8ml/__init__.py` (~L11–17) leaks DAG internals (`PipelineSpec`, `PipelineStep`, `StepName`); exports both `Iter8MLError` and `TabularBlueprintError` (only the latter used internally); and eagerly imports the entire graph (the project already defends this with a 2.5s budget in `tests/unit/test_import_time.py` ~L14–21).
- **Approach:** Thin the top-level `__init__` to `__version__` + lazily-exposed heavy names (PEP 562 `__getattr__`), or move advanced DAG types to `iter8ml.engine.pipelines`. Pick **one** canonical error name (`Iter8MLError` recommended, keep `TabularBlueprintError` only as a deprecated alias). Leave advanced bits on submodules.
- **Verify:** `pytest tests/unit/test_import_time.py`; ensure all existing `from iter8ml import X` usages still resolve.

### 1.5 Single source of truth for run persistence
- **Why:** Runs exist as `control/runs/<id>/run.json` (canonical) **and** catalog rows (`storage/catalog.py` ~L93 rebuilds from `run.json`); `services/docs_export.py` (~L25/L52) reads `run.json` directly, bypassing the catalog. The catalog file is misnamed `catalog.duckdb` but is actually SQLite.
- **Approach:** Pick one canonical store. Either make `DocsExporter`/`reporting` read from `LocalCatalogStore`, or drop the run-registration into the catalog and treat `run.json` as the only store. Rename `catalog.duckdb` → `catalog.sqlite` (or implement real DuckDB if intended).
- **Verify:** `pytest tests/unit/test_medallion_contracts.py tests/unit/test_docs_export*.py`; confirm docs still generate after the change.

### 1.6 `verification/` cleanup + naming
- **Why:** `verification/schema.py` `verify_product` is a thin passthrough over `LocalArtifactStore.verify` (only caller `cli/medallion.py`); `verification/leakage.py` shares a name with `data/leakage.py` but is a different concern (split validation vs feature permutation leakage).
- **Approach:** Drop `verify_product` and have `cli/medallion.py` call `store.verify(...)` directly (or give `verify_product` real value: strict `ProductManifest` schema / cross-product ref checks). Rename `verification/leakage.py` → `verification/split_validation.py` (`validate_split_frame` → `validate_split`).
- **Verify:** `pytest tests/unit/test_medallion_contracts.py`; grep for all importers before renaming.

### 1.7 Wire or retire `services/mcp.py`
- **Why:** `services/mcp.py` is a parallel reimplementation of CLI/session logic (run/hpo/registry/export/drift) — not registered as a CLI command, no `pyproject` entry point, not in `services/__init__`.
- **Approach (pick one):** (a) expose it as `iter8 mcp serve` backed by `ExperimentSession`, delegating to `session.py`; or (b) delete it if MCP is not a planned surface. Do **not** leave a dangling duplicate.
- **Verify:** ruff/mypy; if kept, an `iter8 mcp --help` smoke test.

### 1.8 Simplify `config.py` legacy flat-key shim
- **Why:** `_FLAT_DELEGATES` / `_LEGACY_PIPELINE_KEYS` / `__getattr__`/`__setattr__` overrides / `nest_flat_config_fields` (~L128–316) form a large compatibility layer.
- **Approach:** If legacy flat YAML configs are no longer supported, delete the shim. Otherwise document it as the single supported compat layer and add a per-key test matrix.
- **Verify:** `pytest tests/unit/test_config.py`; search for any flat-key configs in `examples/`, `demo/`, `workspace/`.

### 1.9 Make LLM state commentary reachable from CLI
- **Why:** `state_observer.py` (~L241) imports `TabularAgent` and supports `llm_enabled`, but `cli/analyze.py` `state` command constructs `StateObserver(workspace=ws)` without passing it through `session.state(llm_enabled=...)` (`session.py` ~L104).
- **Approach:** Add `--llm/--no-llm` to the `state` command, threading into `StateObserver`/`session.state`.
- **Verify:** `pytest tests/unit/test_state_observer.py`; a CLI smoke test with `llm_enabled`.

**Phase 1 exit criteria:** no duplicated Hamilton-driver construction; one drift protocol/report schema; GBDT defaults unified; top-level `__init__` thin; single run store; `mcp.py` resolved; config shim decided; ruff/mypy/pytest green.

---

## 2. PHASE 2 — Architecture restructuring (Large effort, Large impact)

> **Requires a design decision before coding.** Do not start 2.1/2.2 without aligning on the chosen topology.

### 2.1 Resolve the dual pipeline system (medallion vs engine)
- **Problem:** `dataflows/` (bronze→silver→gold→platinum) materializes a full second data lake with `ProductManifest`/artifact-store/digest infrastructure **and its own split implementation** (`gold.py` `build_split_frame`, ~L31–111). Training ignores medallion outputs: `orchestration/service.py` (~L140–143) reads `splits.parquet` then discards it; `engine/evaluator.py` (~L34–89) recomputes splits with a **weaker** splitter (no `group`, no `purged_time`/embargo). Gold "features" (`gold.py` ~L148–149) are just `frame.drop(target_col)`, not the AFE/embedding-engineered features from `engine/pipelines/nodes/features.py`.
- **Files:** `dataflows/{bronze,silver,gold,platinum_train}.py`, `orchestration/service.py`, `engine/trainer.py`, `engine/evaluator.py`, `runtime/plan.py`.
- **Two valid topologies (choose one):**
  - **(A) Medallion as consumed source of truth:** wire `gold` splits + features into the engine training path; unify split semantics so the engine uses the medallion `SplitManifest` instead of recomputing. High cohesion, larger blast radius.
  - **(B) Slim the medallion to a thin audit/lineage pass:** stop persisting full-frame copies at every tier and drop the parallel split implementation; keep medallion only for verification/auditing of the engine's actual outputs.
- **Also fix (both topologies):** `gold.py` (~L141/L163) hardcodes `overlap_checks_passed=True` and `quality_summary={"split_overlap": False}` instead of deriving from `validate_split_frame` result (audit finding B3).
- **Verify:** `pytest tests/unit/test_medallion_contracts.py tests/integration`; a full `cli/run.py` + `cli/medallion.py` round-trip confirming splits/features are consistent across both paths.

### 2.2 Single orchestration seam
- **Problem:** `Trainer` (`engine/trainer.py`) orchestrates the Hamilton DAG; `MedallionExecutionService` (`orchestration/service.py`) orchestrates medallion stages and calls `Trainer`; an unused `Orchestrator` protocol + `LocalOrchestrator` exist as a scheduler-neutral seam. (Note: `LocalOrchestrator` is **kept** — it powers cancellation in tests.)
- **Approach:** Decide one orchestration contract. Either (a) make `MedallionExecutionService` implement a single `Orchestrator` protocol and route all runs through it, or (b) collapse `Trainer` into the service seam. Remove the redundant abstraction only after the single seam is wired and tested.
- **Verify:** `pytest tests/unit/test_medallion_contracts.py` (esp. `test_cancellation_request_is_honored_at_next_stage_boundary` — must keep cancellation working).

### 2.3 (Small follow-up) Remove leftover OOV buffer machinery
- **Why:** `engine/models/sparse_embedder.py` still has `_OOVEmbeddingMixin` (`_init_oov_buffers`, `_update_oov_means`) registered/called in `EntityEmbedding`/`TabularDAE` `__init__`, but `get_oov_embeddings` (its only reader) was deleted in Phase 0 — leaving dead buffers.
- **Approach:** Remove `_OOVEmbeddingMixin` and its `__init__` calls (or implement a real `transform(df)` inference path that uses them — see audit B2).
- **Verify:** `pytest tests/unit/test_models*.py`; confirm embedding training/inference still works.

**Phase 2 exit criteria:** one pipeline topology (medallion consumed **or** slim audit pass), no recomputed/ignored splits, one orchestration seam with cancellation intact, OOV machinery resolved; ruff/mypy/pytest green including integration + e2e.

---

## 3. Out of scope / optional
- **Caching:** Phase 0 deleted `data/cache.py` (`PreprocessingCache`). If preprocessing caching is desired, reintroduce it wired into the training path rather than as dead code.
- **GPU detection** is duplicated (`Trainer` `HardwareProfile.detect()` vs `ModelSelector._has_gpu` re-importing torch). Pass a single gpu/flag through node inputs.
- **Reproducibility/non-determinism:** Phase 0 added `torch.manual_seed` in `EmbeddingEngine`. Audit also flagged numpy seeding and `evaluate_with_std` removal (done). Consider a global seed-propagation pass if full determinism is a goal.
- **Naming:** the `iter8ml` (import namespace) vs "Tabular Blueprint" (product/exception name) split — resolve once (Phase 1.4) and propagate to docs/examples.

## 4. Recommended agent execution order
1. Phase 1.1 (Hamilton factory) — low risk, unblocks 1.2/1.5 reuse.
2. Phase 1.4 (public API) — low risk, high clarity.
3. Phase 1.6 / 1.9 / 1.3 / 1.8 — independent, parallelizable.
4. Phase 1.2 (drift) — highest value, do after 1.1.
5. Phase 1.5 / 1.7 — persistence + MCP.
6. Phase 2 — **only after Phase 1 merged**, start with the 2.1 design decision.
