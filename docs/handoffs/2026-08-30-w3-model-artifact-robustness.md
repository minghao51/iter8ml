# W3 — Model & artifact robustness

**Date:** 2026-08-30 · **Priority:** P2 · **Depends on:** W2-T6 (owns the fit-side of
`data/embedding.py` first; this stream owns persistence + everything else).
Independent of W1/W2 file-wise except `data/embedding.py` — sequence after W2.
**Evidence:** findings §1 (Persistence/crashes), §2 (Reliability, Model layer),
§4 (io.py).
**Files owned by this workstream:**
`engine/models/{ft_transformer,tabpfn_model,gbdt_base,lightgbm_model,xgboost_model,catboost_model,baselines,model_configs}.py`,
`engine/calibration.py`, `domain/events.py`, `storage/{catalog,local}.py`,
`services/retention.py`, `data/{quality,loader}.py`, `data/embedding.py`
(persistence only), `utils/io.py`.

## Goal

No silent failures at the model seam or in the durability layer: models that
save/load correctly, GBDTs that train efficiently and honestly, storage that
doesn't leak resources or race, quality/leakage audits that fail loudly instead
of crashing.

## Tasks

### W3-T1 — FT-Transformer: labels + persistence `[V]`

**Problems.**
- Raw `y` fed to `CrossEntropyLoss` (`ft_transformer.py:110-124`): crashes on
  `{1,2}` or string labels; `predict` returns argmax indices undecoded
  (`:168`) — label semantics diverge from the GBDT contract
  (`gbdt_base.py:47-67` re-encodes via `np.unique` + decodes).
- `save()` with accelerator writes a bare state_dict; `load()` expects
  `{"n_features","n_classes","model_state"}` (`:180-200`) → any
  accelerator-path artifact can't load. Tests bypass `save()`
  (`tests/unit/test_ft_transformer.py:80-86`).
- Constructor `n_classes=2` + `task="regression"` builds a 2-output head;
  `predict` squeezes to `(n,2)` (`:172`).

**Fix.** Mirror the `gbdt_base` label contract (`classes_` + decode in
`predict`/`predict_proba`). Always save the full checkpoint dict (works through
`accelerator.save` too — it's just `torch.save`). Clamp/assert
regression `n_classes==1`. Add roundtrip tests for **both** save paths
(accelerator set / not) with `{1,2}` and string labels.

### W3-T2 — TabPFN persistence `[V]`

**Problem.** `save()` uses `safe_dump` (`tabpfn_model.py:82-86`) whose
roundtrip passes through `RestrictedUnpickler`; the allowlist
(`utils/io.py:56-65`) has no `torch.` prefix and TabPFN v2 is torch-backed →
`UnpicklingError` on load.

**Fix (recommended).** Don't widen the unpickler allowlist (it's a security
boundary). Save state_dict + constructor args via `torch.save` /
`safe_dump`-able scalars, rebuild on `load` — the FT pattern from T1. Add the
missing roundtrip test. Alternative (only if state_dict is impractical): add a
narrow `torch.` prefix with a documented risk-acceptance note in the ADR —
prefer not to.

**Acceptance.** `TabPFNModel.save/load` roundtrips predictions exactly; test
runs without network/model download by monkeypatching or using the smallest
config.

### W3-T3 — GBDT adapters: early stopping + honest params

**Problems.** No early stopping anywhere — fixed `num_boost_round` (default
1000) while HPO searches up to 3000 iterations (`lightgbm_model.py:41-45`,
`xgboost_model.py:38-41`, `catboost_model.py:60-62` vs
`model_configs.py:23`); `BaseGBDTModel.fit` silently swallows `**kwargs`
(`gbdt_base.py:44`); declared `n_classes` overrides fitted labels
(`:44`); `NaiveBaseline.predict_proba` puts all mass on the wrong class for
string labels (`baselines.py:50-55`); `LinearBaseline` hardcodes
`random_state=42` (`:73`); CatBoost gets `cat_features` twice
(`catboost_model.py:47,62`).

**Fix.** Add eval-set early stopping (hold out a slice of the training rows
when the caller doesn't pass one; `use_best_model` for CatBoost; log a
one-line warning if `**kwargs` are dropped by a subclass). Trust fitted labels
over declared `n_classes` (validate or drop the override). Fix NaiveBaseline
mass placement + seeds. Keep `gbdt_base` param maps as-is here (template-method
collapse is W4).

**Acceptance.** Fit with a tiny `n_estimators` + eval set → stops early (assert
best_iteration < cap); kwargs warning test; `n_classes` mismatch raises or is
ignored-with-warning; string-label NaiveBaseline proba columns align with
`classes_`.

### W3-T4 — Event sink race + lock hygiene `[A]`

`domain/events.py:55-77`: `_finalized` computed once in `__init__`; a stale
second sink for the same `run_id` can `os.replace` the archive with a fresh
empty hot file (archive wipe). Fix: re-check `archive_path.exists()` inside the
lock in `append` and `finalize`; delete the `*.lock` file on finalize (or note
it for retention). Test: two sink instances, interleaved finalize/append.

### W3-T5 — Storage resource leaks `[A]`

- `storage/catalog.py`: every method `with self._connect()` commits but never
  closes → wrap in `contextlib.closing` (all methods, `:26-133`); extract the
  4× duplicated `INSERT OR REPLACE` into `_upsert` while there.
- `storage/local.py:47,106-149`: make `ProductWriter` a context manager;
  abort-on-exception so `mkdtemp` dirs don't leak.
- `services/retention.py:12`: skip hidden dirs newer than a grace period (or
  holding a product lock) so `gc --apply` can't rmtree an in-flight write
  (`local.py:47` temp dirs).

**Acceptance.** Test: catalog connections closed after each call (counter on
`sqlite3.connect`); writer abort removes its temp dir; gc with a fresh
in-flight-style dir (recent mtime) keeps it.

### W3-T6 — quality/leakage/adapter hardening `[A]`

- `data/quality.py:23-90`: add `task` param; guard non-numeric feature frames
  (skip or encode with a note); guard classes with < cv members; **unify the
  two noise criteria** (`find_label_issues` output currently computed and
  discarded by `clean_noise` — pick one mechanism).
- `data/leakage.py:38` vs `:74`: docstring says relative drop, code computes
  absolute — implement relative (or fix the docstring + add an absolute-drop
  column) and note that strong-but-legit features will be flagged.
- `data/adapter.py:13-15`: validate dtypes/nulls at the seam; raise a typed
  error that names the offending column instead of letting sklearn crash far
  from the cause.
- `data/loader.py:11-32`: wrap `infer_schema_length=1000` `ComputeError` in
  `DataLoadError` for `load_csv`/`load_parquet` too (only `load_data` is
  wrapped today, `:35`).

**Acceptance.** One test per guard (string-feature quality audit returns a
report, not a crash; regression task supported; adapter null → typed error
naming the column).

### W3-T7 — Embedding persistence load path `[A]`

`data/embedding.py:331-353` writes `{run_id}.pt` + `_mappings.json` with no
reader in `src/`. Implement `load` (reconstruct category→index mappings,
handling the stringified keys at `:352` — keep keys native) or remove the
write until inference-time encoding exists (honest choice; record in
`docs/feature-engineering.md`). **Land after W2-T6** (same file).

### W3-T8 (small) — Calibration + evaluator polish `[A]`

- `CalibratedModel` call ignores `config.cv_folds`
  (`train.py:250` → `calibration.py:29-31`): pass it through.
- `calibration.py:86-93`: `save` persists the unfitted base model when
  calibration applied — persist the fitted calibrated ensemble (or refit on
  load-fallback); one roundtrip test.
- `evaluator.py:98-100` unknown metric → `ValueError` naming the metric (not
  bare `KeyError`); `:114` hoist `requires_proba` out of the fold loop;
  `:176-177` `compute_lift` missing-metric default `0.0` → return `None`
  (±100% lift is misleading).

### W3-T9 (small) — io.py contract cleanup `[A]`

`utils/io.py`: `load_events` → `list(iter_events(path))` (`:22-38`);
`safe_load_file` → reuse `safe_load` (`:118-123`); stream-mode `safe_load`
skips HMAC verification unlike the bytes path (`:111-115`) — either verify in
stream mode or raise "stream mode cannot verify" (pick one; document).

## Suggested subagent orchestration

1. `reviewer` (read-only): confirm W2 landed first (embedding fit-side), then
   verify findings for this stream's files.
2. `worker` A (models): T1 → T2 → T3 → T8 (sequential; shared test files).
3. `worker` B (durability): T4 → T5 → T9 (disjoint from worker A; can run in
   parallel).
4. `worker` C: T6 (data/ guards) → T7 (embedding persistence) — after both.
5. `reviewer`: diff review per worker; focus on "does every save have a
   symmetric load + roundtrip test?"
6. Coordinator: gate + bookkeeping.

## Gotchas

- Do **not** widen `WHITELISTED_PREFIXES` for T2 without an explicit risk note;
  state-dict save is the portfolio-defensible answer.
- FT label encoding must apply identically in `fit`, `predict`,
  `predict_proba`, and the saved artifact (persist `classes_` in the
  checkpoint).
- Early stopping changes reported iteration counts → benchmark/leaderboard
  numbers shift; note in `REPORT_LOG.md`.
- `test_ft_transformer.py:80-86` hand-builds checkpoints — update it rather
  than preserving the bypass.
- OpenMP: models tests must run with the cap available (W1-T2 lands it in the
  factory; if W1 hasn't landed, set the cap in the affected tests).

## Validation gate

Same full gate, plus:

```bash
uv run pytest tests/unit/test_ft_transformer.py tests/unit/test_models.py tests/unit/test_medallion_contracts.py -q
```

## Definition of done

- [ ] T1–T9 implemented with roundtrip/guard tests
- [ ] No `save` without a symmetric, tested `load`
- [ ] Full validation gate green; `REPORT_LOG.md` entry
- [ ] `docs/models.md` + `docs/data-loading.md` updated where behavior changed
- [ ] Statuses updated inline; no commit unless explicitly asked
