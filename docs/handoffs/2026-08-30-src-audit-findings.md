# Full `src/` audit findings — 2026-08-30

**Status:** evidence base for the W1–W5 workstream handoffs in this directory.
**Provenance:** 5 parallel `reviewer` subagents (core/platform, data, engine,
models, services/cli/analysis/orchestration) over 92 files / ~10.7k LOC; all
dead-code claims grep-verified across `src`/`tests`/`docs`. The coordinator
subsequently re-verified every Critical marked `[V]` against source on
2026-08-30. `[A]` = agent-verified only; re-confirm with a quick read before
acting.

**Legend:** `[V]` coordinator-verified · `[A]` auditor-verified.

---

## 1. Critical (correctness / integrity)

### Analytics correctness — inflates the CV metrics the project exists to prove
- `[V]` **Supervised feature construction leaks across CV folds.** Embeddings fit
  on full-data `y` (`data/embedding.py:140-197,224-274`); interaction discovery
  and pruning select on full-data CV (`data/features.py:213-346,360-417`);
  target transform fits on full `y` (`data/features.py:67-115`); prep imputation
  stats fit on all rows (`engine/pipelines/nodes/prep.py:65-77,186-201`). The
  gold `row_id/fold/role` split (ADR-0003) exists to police this boundary, but
  feature nodes are fold-agnostic. → W2
- `[V]` **HPO maximizes RMSE for regression.** `engine/hpo.py:284,289` hardcode
  `direction="maximize"`; the objective returns the first score
  (`hpo.py:~310`); regression defaults are `metrics=["rmse","r2"]`
  (`config.py:356-359`). `services/reporting.py:45-62` owns directionality but
  is bypassed. → W1
- `[V]` **Final champion artifact is fit on holdout (`role=="test"`) rows.**
  `engine/pipelines/nodes/train.py:24-56` maps only train/validation roles for
  CV, but `_train_one` calls `model.fit(X, y)` on the full matrix
  (`train.py:~248`) — including test rows the preprocessing stats already saw.
  → W2
- `[A]` **Pruning can drop every feature** (`data/features.py:398-405`, no
  keep-at-least-one guard). → W2
- `[A]` **Interaction validation on the same data used to propose them**
  (multiple-testing selection bias, `data/features.py:270-274`); candidate
  truncation biased to first top-k. → W2 (honest-status note acceptable)

### User-facing wrong results
- `[V]` **`iter8 export --target` silently discarded.** `cli/export.py:14,19`
  never forwards `target`; `session.export` (`session.py:96-98`) has no
  `target_col` param. Downstream: metadata `target_col=""` → generated
  `predict()` (`services/export.py:93-99`) doesn't drop the label column →
  positional `X = df.to_numpy()` shifts every feature → silently wrong
  predictions. → W1
- `[V]` **Registry champion compared across incomparable metrics.**
  `services/registry.py:174-179` compares candidate vs incumbent using only the
  candidate's `metric_name`; incumbent's metric is stored but unused.
  `resolve_primary_score` falls back to "any numeric metric"
  (`services/reporting.py:49-62`), so `accuracy=0.90` can displace
  `roc_auc=0.83` under one key. → W1
- `[A]` **Export predictor silent fallbacks:** unknown model → hardcoded
  CatBoost import (`services/export.py:297-299`); missing task key → defaults
  `"classification"` (`:302`). → W1
- `[A]` **`registry promote` advertised, unimplemented** — `cli/export.py:33,44`
  help says "show or promote"; `action=="promote"` falls through and exits 0.
  → W1
- `[A]` **HPO on baselines crashes confusingly** — name passes registry
  validation, then `getattr(ModelConfigs(), model)` raises `AttributeError`
  (`engine/hpo.py:188-189`). → W1

### Persistence / crashes
- `[V]` **FT-Transformer label contract broken:** raw `y` into
  `CrossEntropyLoss` (`ft_transformer.py:110-124`) — crashes on `{1,2}` or
  string labels; `predict` returns argmax indices with no decode
  (`:168`) diverging from GBDT label handling (`gbdt_base.py:47-67`).
  → W3
- `[V]` **FT-Transformer save/load roundtrip broken whenever accelerator set:**
  `save()` writes a bare state_dict via `accelerator.save` (`:180-190`) but
  `load()` expects a checkpoint dict (`:192-200`); tests bypass `save()`
  (`tests/unit/test_ft_transformer.py:80-86`). → W3
- `[V]` **TabPFN persistence incompatible with the safe-unpickle allowlist:**
  `save()` uses `safe_dump` (`tabpfn_model.py:82-86`); allowlist prefixes
  (`utils/io.py:56-65`) contain no `torch.` prefix; TabPFN v2 is torch-backed.
  → W3

### Contract violations
- `[V]` **ADR-0004 violated at the HPO/MCP/factory seams.** OMP cap runs only
  in `Trainer.__init__` (`engine/trainer.py:33`); `get_model_class`
  (`engine/models/factory.py:51-61`) does uncapped `importlib.import_module` of
  GBDT modules that import libgomp at top level
  (`lightgbm_model.py:2`, `xgboost_model.py:3`, `catboost_model.py:7`).
  Affected paths: `cli/optimize.py`, `services/mcp.py` `run_hpo`, any direct
  factory user. → W1
- `[V]` **Inert config surfaces** — `HPOConfig.run/n_trials`
  (`config.py:66-71`, validated at `:357-364`) never read by any execution
  path; `QualityConfig.run_audit/auto_clean_noise/noise_quality_threshold`
  (`config.py:73-80`) shadowed by pipeline step params
  (`executor.py:80-84`, shim `config.py:278-284`). `StepName.HPO`
  (`config.py:92`) has no DAG module. → W1
- `[V]` **Platinum cache is content-blind.** `dataflows/platinum_train.py:20-34`:
  pid from opaque `run_id`; `store.exists(pid)` returns stale manifest without
  comparing `results`; `specification_digest` covers only
  `{run_id, experiment}` — resolved training config not recorded. → W2

## 2. Warnings (should fix)

### Reliability / durability
- `[A]` Events-finalize race → archive wipe: `_finalized` computed once in
  `__init__`, never re-checked under lock (`domain/events.py:55-77`); stale
  instance finalizing touches a fresh hot file and `os.replace`s the archive.
  Also per-append `FileLock` leaves permanent `*.lock` files (`:54`).
- `[A]` sqlite connections committed but never closed (`storage/catalog.py`
  every method, `:26-133`).
- `[A]` `ProductWriter` not a context manager; `commit()` failure path never
  aborts → mkdtemp leak under `lake/` (`storage/local.py:47,106-149`).
- `[A]` `gc --apply` rmtree's *any* hidden dir under the lake — exactly where
  in-flight writes live (`services/retention.py:12` vs `local.py:47`).
- `[A]` `run_completed` logged for failed/cancelled runs
  (`engine/tracker.py:95-98`, `trainer.py:104-105`); resume scans only the live
  log file, missing rotated backups (`trainer.py:157-165`).

### Reproducibility
- `[A]` `shuffle=True` with `random_seed=None` → nondeterministic folds
  (`dataflows/gold.py:29,62-66`; `domain/manifests.py:106` default `None`).
- `[A]` `--data` CLI override not synced into `experiment_config.data_path` —
  manifests record the config value, not what was loaded (`cli/run.py:57-61`).
- `[A]` Silver pid collision: supplied contract silently drops `target_col`
  from the digest (`dataflows/silver.py:30`).
- `[A]` `--log` default CWD-relative `workspace/experiments.jsonl` hardcodes the
  layout (`cli/optimize.py:16-20`).

### Computed-but-discarded plumbing (recurring theme)
- `[A]` `state_observer.py:139-262` renders sections from events **nothing
  emits** (`leakage_audit`, `target_transform`, `afe_completed`,
  `shap_explainability`, `drift_check`); meanwhile `DataPrepResult.leakage_report`
  is computed (`prep.py:186-192`) and never consumed outside tests.
- `[A]` `model_completed` events omit `n_rows/n_features/dataset/hardware`
  (`trainer.py:121-131`; read at `services/reporting.py:160-171`) → report
  shows `? / ?` and always `Device: cpu`.
- `[V]` `config.max_workers` never wired into the executor inputs
  (`executor.py:22-48`) → node default `max_workers=1` always wins; the
  `ThreadPoolExecutor` branch (`train.py:328-343`) is dead (and an ADR-0001
  tension).
- `[A]` Warmstart injects guessed distributions that clash with the real search
  spaces → TPE silently ignores them; `n_trials_injected` overstates effect
  (`engine/hpo_warmstart.py:36-69`, vs `model_configs.py:20-23`).
- `[A]` All HPO evaluation failures masked as `TrialPruned`; all-pruned study
  crashes opaquely on `study.best_params` (`engine/hpo.py:314-320,226`).
- `[A]` Baselines silently vanish on error — bare `except: continue` with no
  event/error record (`train.py:148`).
- `[A]` Gold manifest fields vacuous after the raise-gate: `split_overlap` always
  False, `overlap_checks_passed` always True, `temporal_checks_passed` can never
  be False (`dataflows/gold.py:133-136,195-214`).

### Model layer
- `[A]` No early stopping in any GBDT adapter — fixed `num_boost_round` (default
  1000) while HPO searches up to 3000 iterations (`lightgbm_model.py:41-45`,
  `xgboost_model.py:38-41`, `catboost_model.py:60-62`); `BaseGBDTModel.fit`
  silently swallows `**kwargs` (`gbdt_base.py:39-50`).
- `[A]` Declared `n_classes` silently overrides fitted labels
  (`gbdt_base.py:44`).
- `[A]` `NaiveBaseline.predict_proba` can put all mass on the wrong class
  (`baselines.py:50-55`).
- `[A]` Calibration ignores `config.cv_folds` (always 3) and saves the unfitted
  base model when calibration applied (`calibration.py:29-31,86-93`).
- `[A]` Integer features >50 distinct values silently reinterpreted as
  categorical and **dropped** from X (`data/embedding.py:22-49,91-92`);
  `embeddings.astype(X.dtype)` truncates float embeddings on int X
  (`:113,115`).
- `[A]` Embedding checkpoints written (`{run_id}.pt` + mappings) with **no load
  path** anywhere in `src/` (`data/embedding.py:331-353`).
- `[A]` `data/quality.py:23-90` — unhandled failure modes (string features →
  LogisticRegression crash; regression target unsupported; <3 class members
  crash CV) and two inconsistent noise criteria (`find_label_issues` output
  computed but unused by `clean_noise`).
- `[A]` `data/leakage.py:38` docstring says *relative* drop, code computes
  *absolute* (`:74`).
- `[A]` `data/adapter.py:13-15` — the model seam does no dtype/null validation.
- `[A]` Install hints reference a nonexistent `deep` extra
  (`ft_transformer.py:30`, `sparse_embedder.py:21`, `tabnet_model.py:4`);
  pyproject has `gbdt/train/docs/full`.
- `[A]` Tabnet: LR set in two competing places; `early_stopping="valid_loss"`
  configured but only training data passed to fit (`tabnet_model.py:46-69`).
- `[A]` `selector.py:8-13` docstring contradicts code (TabPFN row limit enforced
  in selector `:63`, not model.fit).

### Services / CLI / orchestration
- `[A]` Flagship medallion lifecycle has **no CLI/MCP surface** —
  `session.medallion_run()` (`session.py:56-64`) reachable only from one test
  (`tests/unit/test_split_wiring.py:87`);
  `iter8 run` bypasses it (`session.py:31-42`); ADR-0003 untestable from CLI.
- `[A]` Registry read path triplicated outside the service, bypassing the lock
  (`services/reporting.py:160-164`, `cli/export.py:39`, raw write
  `cli/main.py:31`); MCP already uses `RegistryService.get_all()`
  (`services/mcp.py:154-158`).
- `[A]` `orchestration/local.py:24-42` duplicates `service.py:243-258` verbatim;
  `LocalOrchestrator.submit` fabricates a run_id that exists nowhere on disk.
- `[A]` Drift method map duplicated (`cli/analyze.py:46-52` ≡
  `services/mcp.py:170-175`); CLI builds a bare `PipelineExecutor` where
  `session.drift_check` wraps the identical call.
- `[A]` HPO wiring duplicated: `cli/optimize.py:28-40` ≡ `services/mcp.py:98-129`.
- `[A]` `_read_artifact`: URI without `/` raises raw `ValueError`
  (`orchestration/service.py:332-339`); `next(glob(...), None)` silently picks
  an arbitrary match.
- `[A]` `plan.py`/`session.py` — `materialization: str` untyped at the boundary
  (`runtime/plan.py:16,49`, `session.py:52`); should be a `Literal`.
- `[A]` `verification/split_validation.py:18,26-36` — `test` role accepted but
  gets no checks (no emptiness, no train∩test overlap).

## 3. Dead code (all `[A]`, grep-verified at audit time; **re-verify before deleting**)

| Item | Location |
|---|---|
| `export_summary`, `products()` | `storage/catalog.py:132,122` |
| `catalog_path` (dup of catalog logic) | `workspace.py:55` |
| `BUNDLED_DATASETS` export | `datasets/__init__.py:29` |
| `StepName.HPO` (no DAG module) | `config.py:92` |
| `HPOConfig.run/n_trials`, `QualityConfig.run_audit/auto_clean_noise/noise_quality_threshold` | `config.py:66-80` (W1) |
| Unreachable no-Hamilton fallback (~80 dup lines) | `engine/pipelines/nodes/prep.py:231-283` |
| `neg_mean_squared_error` branches (unreachable) | `data/features.py:287-289`, `data/leakage.py:72-74` |
| `TabPFNConfig.n_estimators`; all `random_seed` config fields (never plumbed) | `engine/models/model_configs.py` |
| `ModelSelector.select(task=…)` unused; py3.9 `TypeError` shim; decorative `AbstractModel` | `selector.py:40`, `factory.py:28-29`, `models/base.py` |
| `TabNetModel._build_model` `n_classes` param unused | `tabnet_model.py:27,37` |
| `create_study(model_name, n_trials=…)` ignores both params | `hpo.py:29-36` |
| `create_warmstarted_study(n_trials=…)` unused; `_build_trial_data(value)` unused | `hpo_warmstart.py:94,74-80` |
| `TrackingHook._run_id` stored, never read | `pipelines/hooks/tracking_hook.py:11` |
| `PipelineMode.EXPORT/HPO/INFERENCE` (collapse to TRAINING/DRIFT) | `executor.py:92-94` |
| `DriftDetectorProtocol` unused | `analysis/_protocol.py:21-31` |
| `LocalOrchestrator` (test-only); `Orchestrator` protocol comment-only | `orchestration/local.py`, `protocol.py:13-19` |

## 4. DRY / simplification themes

- **Entry-point duplication:** registry reads ×3 (see §2); HPO wiring ×2; drift
  map ×2; `status/cancel` verbatim ×2; table formatting ×3
  (`reporting.py:111-158` console+markdown, `services/mcp.py:59-72` hand-rolled).
- **GBDT trio:** objective/metric maps + predict boilerplate triplicated
  (`lightgbm_model.py:9-24`, `xgboost_model.py:10-28`, `catboost_model.py:43-55`)
  → `gbdt_base` template method; `_create_model` dead for LightGBM/XGBoost;
  `_build_params()` called twice per fit.
- **data/ triplicated baseline+CV scaffolding** with drifting configs
  (`features.py:241-257`, `leakage.py:59-64`, `quality.py:39-41`).
- **`utils/io.py`:** `load_events` ≡ `iter_events` (`:22-38`);
  `safe_load_file` re-implements `safe_load` (`:118-123`); stream-mode
  `safe_load` skips HMAC verification unlike the bytes path (`:111-115`).
- **`model_configs.py` vs wrapper defaults** identical in two places, config
  side dead (`TabPFNConfig.device/max_rows/n_estimators`, `TabNetConfig.*`,
  every `random_seed`).
- **prep.py unreachable fallback** duplicates the `@config.when` variants
  (~80 lines) while features/drift nodes use `hamilton_stub` — pick one.
- **ADR-0002 residue:** config-value branching inside node bodies
  (`prep.py:162`); executor resolver hardcodes fallback defaults that can shadow
  `PipelineSpec` (`executor.py:70-81`).
- Triplicated throwaway `ExperimentConfig`+`Evaluator` plumbing
  (`train.py:139-152,156-170`, `hpo.py:178-186`); eager `[prep]` driver built
  then discarded in `run_training` (`executor.py:169,252`); private
  cross-module import `hpo_warmstart.py:76 → hpo._build_pruner`.

## 5. Performance

- `[A]` `domain/hashing.py:24-54` — per-row Python-loop JSON+sha256 (dict per
  row), invoked on full frames in bronze (`bronze.py:20,41`), silver
  (`silver.py:33,63`), and again in gold (`gold.py:128`). Dominant medallion
  cost; vectorize/chunk.
- `[A]` `dataflows/gold.py:70-99` — split frame built via per-row dict appends
  (≈ n_rows × folds iterations) + per-fold `sorted()`; time column extracted
  twice (`:52-55,200`).
- `[A]` `data/embedding.py:44-49` per-column `n_unique()` loop;
  `:76` `map_elements` Python UDF → `replace_strict` orders of magnitude faster.
- `[A]` `data/features.py:259-262` — `np.column_stack([X, interaction])`
  deep-copies all of X per candidate pair (up to 200 copies); `leakage.py:92`
  same per column; joblib pickles X per worker (`:323-325`).
- `[A]` `storage/local.py` — `exists()/_manifest_path()/open_artifact()/verify()`
  each glob-scan the whole lake, O(products) per `store.begin()`; catalog exists.
- `[A]` `services/mcp.py:139`, `cli/analyze.py:116` — `load_events` parses the
  whole JSONL for a tail.

## 6. Boundary deviations needing an ADR note (conventions require recording)

- numpy well beyond the DataAdapter seam: `dataflows/gold.py:27-99` (sklearn
  splitters), `engine/calibration.py:5`, `engine/hpo.py:9`,
  `engine/evaluator.py:3`, `services/export.py:27`, `analysis/*`.
- pandas in `engine/models/tabnet_model.py:15` — guarded + documented, but no
  ADR records it.
- `ThreadPoolExecutor` inside the `train.py` node (ADR-0001 tension; currently
  dead code).

## 7. Verified-clean (positive assurance from the audit)

- Medallion contract correct: atomic manifest + `_SUCCESS` via tmp+fsync+
  `os.replace` (`storage/local.py:125-145`); deep-checksum idempotent resume
  (`:106-115,184-187`); overlap gate before gold commit (`gold.py:130`,
  `service.py:145`); events alone never a checkpoint (`service.py:260-268`).
- Restricted unpickler uses dot-safe prefix allowlist (`utils/io.py:63-93`);
  exported predictor checks `allowlisted_model_classes` before importing.
- ADR-0005 two-seam contract correct: event publication best-effort, state
  publish failure raises `TrainerStatePublishError`.
- Metric directionality genuinely centralized (`services/reporting.py:45-62`);
  no re-implementation found outside it (only the HPO bypass, §1).
- `ruff` + `mypy` clean across audited packages (excluding stale `build/`).
