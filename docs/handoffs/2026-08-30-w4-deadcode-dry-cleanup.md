# W4 — Dead code & DRY cleanup

**Date:** 2026-08-30 · **Priority:** P3 · **Depends on:** W1, W2, W3, W5 all merged.
**Runs last** — every symbol must be re-grepped at execution time because
earlier workstreams legitimately delete or move overlapping items (config
fields → W1; prep fallback → W2; embedding persistence → W3; gold/vectorized
code → W5).
**Evidence:** findings §3 (dead code table), §4 (DRY themes).
**Files owned by this workstream:** everything in `src/iter8ml/` not actively
owned by an unfinished stream; plus `pyproject.toml` (only if an extra name
needs fixing — see W4-T9).

## Goal

Shrink the codebase honestly: delete verified-dead symbols, collapse the
entry-point duplications, and leave one home for each repeated pattern. No
behavior changes beyond deletions and pure refactors; every refactor keeps
tests green without edits beyond mechanical updates.

## Part A — Deletions (one `worker` subagent, sequential, re-grep each)

Re-verify with `grep -rn "<symbol>" src tests docs notebooks demo examples`
immediately before each deletion; if a consumer appeared, record it in this
file and skip.

1. `storage/catalog.py:132` `export_summary`; `:122` `products()` — or adopt
   `products()` somewhere if useful; default delete.
2. `workspace.py:55` `catalog_path` (logic duplicated in `catalog.py:26`).
3. `datasets/__init__.py:29` `BUNDLED_DATASETS` export.
4. `config.py:92` `StepName.HPO` — *skip if W1-T3 already removed it.*
5. `data/features.py:287-289` + `data/leakage.py:72-74` unreachable
   `neg_mean_squared_error` branches.
6. `engine/models/model_configs.py`: `TabPFNConfig.n_estimators`; all
   `random_seed` fields — **or** plumb them (prefer delete; W3-T3 covers
   seed handling in wrappers).
7. `selector.py:40` `task` param; `factory.py:28-29` py3.9 `TypeError` shim;
   `models/base.py` `AbstractModel` (delete or actually annotate the factory
   return type with it — prefer annotate, it's one line).
8. `tabnet_model.py:27,37` unused `n_classes` param on `_build_model`.
9. `hpo.py:29-36` `create_study(model_name, n_trials)` ignored params
   (drop them from the signature); `hpo_warmstart.py:94` unused `n_trials`;
   `:74-80` unused `value` in `_build_trial_data`.
10. `pipelines/hooks/tracking_hook.py:11` unused `_run_id`.
11. `executor.py:92-94` collapse `PipelineMode` to TRAINING/DRIFT
    (`_MODE_FINAL_VARS` + `_get_module`, `executor.py:51-60`); update the one
    test referencing the dropped modes.
12. `analysis/_protocol.py:21-31` `DriftDetectorProtocol` (or use it to type
    the engine's detector loader — prefer use).
13. `orchestration/local.py` `LocalOrchestrator` + `protocol.py:13-19`
    `Orchestrator` — delete the class and its test; either annotate
    `MedallionExecutionService` with the protocol or delete it too.
14. `engine/pipelines/nodes/prep.py:231-283` no-Hamilton fallback — *skip if
    W2-T2 already replaced it with `hamilton_stub`.*
15. `executor.py:169` eager `[prep]` driver built and discarded by
    `run_training` (`:252`) → build lazily.

## Part B — DRY collapses (one `worker` subagent; each item is small and
independently testable)

1. **One HPO wiring helper** shared by `cli/optimize.py:28-40` and
   `services/mcp.py:98-129` (put it in `engine/hpo.py` as
   `run_hpo_session(...)` or similar; W1-T1 already touched both call sites).
2. **Registry access via the service** — route `services/reporting.py:160-164`,
   `cli/export.py:39`, `cli/main.py:31` through `RegistryService` (W1-T6 may
   have done the CLI ones; verify).
3. **One drift-method map** — `cli/analyze.py:46-52` ≡ `services/mcp.py:170-175`
   → single constant in `analysis/__init__.py` or the executor; CLI
   `drift` command should call `session.drift_check` instead of building a
   bare `PipelineExecutor` (`analyze.py:44,54`).
4. **One table formatter** — `services/reporting.py:111-158` console/markdown
   builders + `services/mcp.py:59-72` hand-rolled markdown → one formatter
   with a mode flag.
5. **data/ baseline+CV helper** — `features.py:241-257`, `leakage.py:59-64`,
   `quality.py:39-41` build the same `make_pipeline(StandardScaler(), …)` +
   scoring ternary with drifting configs → one helper in `data/_cv.py`
   (private); delete the unreachable scoring branches if W4-A5 didn't.
6. **GBDT template method** — `gbdt_base` owns the objective/metric map and
   the train path; per-lib hooks only; stop calling `_build_params()` twice
   (`gbdt_base.py:48` + per-lib `fit`); drop `_create_model` for
   LightGBM/XGBoost (it's discarded). *Only if W3-T3 didn't already restructure
   fit.*
7. **`utils/io.py`** — `load_events = list(iter_events(...))` (`:22-38`);
   `safe_load_file` reuses `safe_load` (`:118-123`) — *skip items W3-T9
   already did.*
8. **`model_configs.py` vs wrapper defaults** — after A6/A-deletion-6, keep
   defaults in exactly one place (the wrappers) and make configs
   override-only.
9. **`pyproject.toml` extras** — install hints reference a nonexistent `deep`
   extra (`ft_transformer.py:30`, `sparse_embedder.py:21`, `tabnet_model.py:4`);
   either add `deep = [torch, pytorch-tabular, tabpfn]` or fix the hints to
   `train`/`full` (check what `pyproject.toml` actually defines and match).
10. **`pipelines/__init__.py:10-19`** — `describe_pipeline` instantiates an
    executor for a stateless method → free function; `visualize_pipeline`
    returns `""` silently for non-mermaid → raise or document.
11. **`session.py:79-94`** `leaderboard()` hand-mirrors report-entry fields →
    `[e.model_dump() for e in ...]` (tracks schema changes automatically).
12. **`orchestration/service.py:107-152`** — the four stage blocks repeat
    `append → _record_stage → register_product` → a `(name, factory)` loop;
    delete the unreachable `:330` raise.

## Suggested subagent orchestration

1. Coordinator confirms W1/W2/W3/W5 statuses are all "done".
2. `reviewer` (read-only): fresh dead-symbol sweep post-W1–W5 (the audit's §3
   list **will** have drifted). This agent's output replaces Part A's list —
   work from fresh data, not from this file alone.
3. `worker` A: Part A deletions in small commits-sized batches (3–4 items per
   batch, tests run between batches).
4. `worker` B (parallel, disjoint files): Part B items 1–4 (entry points).
5. `worker` C after A: Part B items 5–12.
6. `reviewer`: final diff review — the bar is "tests unchanged or mechanically
   updated; no behavior drift."
7. Coordinator: full gate + bookkeeping.

## Gotchas

- **Re-grep everything at execution time.** This list is a snapshot from
  2026-08-30; earlier streams invalidate rows (notes above flag the known ones).
- Deleting exports? Check `src/iter8ml/__init__.py` `_LAZY`/`__all__` and
  `docs/` for API references; the project publishes an API story — a documented
  export that becomes private needs a docs note.
- Pure-refactor discipline: if a test needs a *behavioral* change to stay
  green, stop and reconsider the refactor.
- Public-API surfave (`docs/models.md`, `docs/hpo.md`) may mention renamed/
  removed helpers — grep docs for each deleted symbol.
- `uv run mypy . --exclude 'build/'` locally (stale `build/` duplicate-module
  error).

## Validation gate

```bash
uv sync --all-groups
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy . --exclude 'build/'
make check-legacy-namespace
uv run python -c "import iter8ml; print(iter8ml.__version__)"   # import surface intact
```

## Definition of done

- [ ] Fresh dead-symbol sweep run; Part A executed against it
- [ ] Part B items 1–9 done (10–12 optional)
- [ ] Full validation gate green; net LOC reduction recorded in `REPORT_LOG.md`
- [ ] `CHANGELOG.md` entry ("internal cleanup, no behavior change")
- [ ] Statuses inline; no commit unless explicitly asked
