# W6 — Report trustworthiness & audit follow-through

**Date:** 2026-09-01 · **Priority:** P1 · **Depends on:** W1 landed (it has — see
§0); T6/T7 coordinate with W2 (`verification/split_validation.py` and
`runtime/plan.py` are W2-owned)
**Evidence:** pipeline/config audit of 2026-09-01 (scout + 2 reviewer subagents,
all load-bearing findings re-verified against source; fix pass executed same
day — see `REPORT_LOG.md` entry "2026-09-01 — pipeline/config audit fixes").
Anchors below were re-verified on the current working tree after that fix pass.
**Files owned by this workstream:** `services/reporting.py`,
`utils/io.py`, `engine/tracker.py`, `engine/trainer.py` (event emit only),
`engine/evaluator.py` (`compute_lift` only),
`engine/pipelines/nodes/train.py` (baseline/lift call sites only),
`config.py` (`positive_class`, tracker settings), `engine/hpo.py` +
`cli/optimize.py` (config seam), `CHANGELOG.md`, `README.md` (claims only),
`docs/evaluation.md` (new section), `.github/workflows/` (CI job).
**Not owned (coordinate):** `verification/split_validation.py`,
`runtime/plan.py`, `dataflows/` → W2.

## §0 State of the tree — read before touching anything

- The working tree carries **two uncommitted changesets**: the 2026-08-30 W1
  work (`engine/hpo.py`, `cli/optimize.py`, `services/export.py`, …) and the
  2026-09-01 audit fix pass (`config.py`, `engine/evaluator.py`,
  `engine/pipelines/nodes/train.py`, `engine/pipelines/nodes/prep.py`,
  `engine/pipelines/executor.py`, `engine/trainer.py`, `engine/calibration.py`,
  `engine/models/ft_transformer.py`, `services/reporting.py`,
  `engine/state_observer.py`, `cli/run.py`, `verification/preflight.py` (new),
  `examples/*`, `docs/*`). **Commit these first** (suggest splitting: W1 batch,
  audit-fix batch), or at minimum `git stash list` / `git diff` review before
  starting. Per AGENTS.md: no commit unless the operator asks.
- **Do not re-audit the fixed items.** Already landed 2026-09-01: fail-fast on
  all-models-fail (`ModelFitError` → exit 1), `extra="forbid"` config,
  parse-time metric/`cv_strategy`/`primary_metric` validation, `--models`
  pre-validation, `random_seed` wired to CV splitters/model ctors/calibration/
  sampling, fold `cv_std` end-to-end, `primary_metric` config field,
  `data_sample` honored (guarded off when `split_frame` is present),
  `shap_enabled`/`drift_detection` removed with loud errors, `tracker` field
  selects backend, calibration pre-calibration score markers,
  `iter8 run --check` preflight (`verification/preflight.py`), `ignore_cols`,
  `iter8ml.metrics` entry-point group, regression example config, legacy-key
  pipeline-fragment bug fix (`accept_legacy_flat_keys` now seeds the default
  8 steps), docs de-rotation. Validation state: 820 tests green, ruff/format/
  mypy/legacy-namespace clean, mkdocs build clean (3 pre-existing warnings in
  `releases/v0.1.0.md` unrelated).

## Goal

Close the remaining accuracy/robustness findings from the 2026-09-01 audit so a
report produced by this framework is trustworthy **by construction, not by
inspection**: the event log stops being a fragile single source of truth, lift
numbers stop being fabricated, leaderboards stop mixing incomparable tasks, and
the last untrue README claims are corrected.

## Tasks (execute in order; each is one subagent-sized unit)

### W6-T1 — Event-log robustness: corrupt-line recovery + rotated-backup reads `[V]`

**Problem.** `experiments.jsonl` is the single source of truth for reports,
state, and resume:
- `utils/io.py:40-53` — `iter_events` raises `ValueError` on the **first**
  malformed line. A truncated final line (crash mid-write) permanently bricks
  `current_state.md` generation (`TrainerStatePublishError` via
  `StateObserver`) and `_load_completed_models` resume. Fail-loud is right for
  *mid-file* corruption, wrong for a torn trailing write.
- `engine/tracker.py:32-33` — rotation at 100 MB × 5 backups;
  `services/reporting.py:177` `_load_completed_events` reads **only the live
  file**, so old runs silently vanish from leaderboards/state and are
  unrecoverable after 5 rotations.
- `engine/trainer.py` resume (`_load_completed_models`) reads events only —
  in tension with ADR-0003 ("event history alone is never a checkpoint").

**Fix.**
1. `utils/io.py::iter_events`: add `on_error: "raise" | "skip" | "skip_trailing"`
   parameter (default `"raise"` to preserve current contract for callers that
   want strictness). `"skip_trailing"`: skip malformed lines only if **all
   subsequent lines are also malformed** (torn tail) — still raise on mid-file
   corruption. `load_events` (line 34-36) gets the same treatment.
2. `services/reporting.py::_load_completed_events`: read live file **plus**
   rotated backups (`experiments.jsonl.1` … `.jsonl.<backup_count>`, newest
   first) with `on_error="skip_trailing"`; dedupe on `(run_id, model,
   artifact_path)` keeping the latest timestamp.
3. Add `data/version` awareness cheaply: if a malformed line is skipped, emit a
   `logging.warning` with line number — silent data loss is not acceptable.
4. **ADR note, not code**: record in `docs/decisions/` (one paragraph in the
   next ADR or ADR-0003 addendum) that the flat path still resumes from events
   and the durable-fix direction is the medallion `run.json` manifest pattern
   (`orchestration/service.py:107-199`). Full manifest port is explicitly out
   of scope here.

**Acceptance.** Unit tests: (a) torn final line → events before it are yielded,
warning logged; (b) mid-file corruption → still raises; (c) report built from
live + `.jsonl.1` includes entries from both; (d) dedupe keeps newest. Existing
`tests/unit/test_io_utils.py`, `test_tracker_rotation.py`, `test_report_service.py`
keep passing.

### W6-T2 — `compute_lift` honesty: no fabricated zeros, no silent baselines `[V]`

**Problem.** `engine/evaluator.py:268-286` — missing metric on either side is
treated as `0.0` (model r2=0.8 vs missing baseline → "+100% lift"); baseline 0
→ 0.0. `engine/pipelines/nodes/train.py:155` — baseline evaluation failures are
swallowed (`except (ValueError, RuntimeError): continue`), so
`lift_over_baselines` silently disappears (NaiveBaseline has no `predict_proba`
→ roc_auc raises → baseline gone without a trace).

**Fix.**
1. `compute_lift` returns `float | None`: `None` when the metric is missing on
   either side; keep the 0.0-baseline guard but return `None` too (0-baseline
   lift is undefined, not zero). Call site in `train.py::_train_one` filters
   `None` before building the lift dict.
2. `train.py` baseline loop: on skip, `logger.warning("baseline %s skipped: %s",
   name, e)` and include the reason in `baseline_scores` as
   `{"error": str(e)}`? — No: keep the dict shape (float-valued scores only),
   log only.
3. Update `Evaluator.compute_lift` docstring; check `tests/unit/test_evaluator.py`
   and any registry/lift tests for the new `None` contract.

**Acceptance.** Unit tests: missing metric → `None`; baseline-0 → `None`;
happy path unchanged. A run with a `roc_auc` metric produces a warning when
NaiveBaseline is skipped, and `lift_over_baselines` omits that pair instead of
fabricating.

### W6-T3 — Leaderboard task isolation + `latest_run` by timestamp `[V]`

**Problem.** `services/reporting.py::build_report` sorts **all** historical
`model_completed` events by each entry's own primary metric — classification
(roc_auc) and regression (rmse/r2) entries from one workspace share a single
incomparable ranking (observed live in the 2026-09-01 smoke test). Also
`reporting.py:113` `latest_run = entries[-1]` is **file order**, not max
timestamp — a resumed or out-of-order log renders the wrong "latest" run in
`current_state.md`.

**Fix.**
1. `build_report(metric=None, limit=None, task=None)`: add optional `task`
   filter; default behavior groups the leaderboard by `entry.task` and emits
   `leaderboard: dict[task, list[LeaderboardEntry]]`? — **No**: that breaks
   `ExperimentReport.leaderboard: list[...]` consumers (`session.py::leaderboard`,
   `state_observer.py`). Instead: keep the flat list but sort with a
   `(task, metric_sort_value(...))` key so tasks never interleave, and add the
   `task` filter parameter for callers that want one task only.
2. `latest_run = max(entries, key=lambda e: e.timestamp)` (timestamps are ISO
   from the tracker; guard non-ISO with a try/except falling back to file
   order).
3. `ExperimentSession.leaderboard(task=...)` pass-through.
4. `state_observer.py::_render_state`: render "Latest Run" from the max-timestamp
   entry (comes free from #2) and note the task next to the leaderboard heading
   when entries span tasks.

**Acceptance.** Unit test: mixed classification+regression events → regression
entries never rank above/between classification entries; `latest_run` respects
timestamps under shuffled insertion. `session.leaderboard(task="regression")`
filters.

### W6-T4 — `positive_class` config + README claims honesty `[V]`

**Problem.** String/categorical targets are silently cast to codes
(`engine/pipelines/nodes/prep.py` encoding), so `roc_auc` orientation depends on
`np.unique` sort order — a 50/50 coin flip for labels like `"good"/"bad"`.
No imbalance handling exists anywhere (verified by grep), yet `README.md:33`
claims "the framework's class-weighting … steps are designed to close that
gap". `README.md:11` claims easy custom transformers/metrics — metrics are now
true (2026-09-01 `iter8ml.metrics` entry points); transformers still require code.

**Fix.**
1. `config.py`: `positive_class: str | float | None = None` — validated for
   classification only; threaded to the prep target-encoding node so the
   positive class maps to `1`. If provided and not found in the target's values
   → fail at first data touch (prep node raises with observed values).
2. `train.py`/`prep.py`: use it when building binary proba for `roc_auc`
   (ensure column order matches the encoded positive class).
3. **README**: rewrite the `:33` sentence to describe reality
   (`model_overrides` e.g. `scale_pos_weight`, plus the preflight imbalance
   warning); adjust `:11` to "custom models (entry points) and metrics
   (entry points); transformers are in-tree" or scope the claim.
4. Preflight (`verification/preflight.py`): when `positive_class` is set, note
   it in the target check line.

**Acceptance.** Unit tests: `positive_class="bad"` flips encoded polarity vs
default; unknown positive_class value raises in prep with observed classes;
regression + `positive_class` → config-time error. Doc build passes.

### W6-T5 — HPO config seam: `--config`, prep routing, min-trials guard `[V]`

**Problem.** HPO is honestly documented as not config-driven, but the ergonomics
are still wrong:
- `cli/optimize.py` takes raw `--data/--target/--task` and ignores
  `ExperimentConfig` folds/metrics; `docs/index.md` had to be corrected to stop
  advertising `--config`.
- `engine/hpo.py:170-189` `setup_hpo_components` applies `DataAdapter` to the
  **raw** frame — string categoricals that the training DAG encodes
  (`prep.py` chain) crash LightGBM/XGBoost HPO.
- `engine/hpo.py:362-368` — any evaluation exception → `optuna.TrialPruned`;
  a systematically broken model yields 100%-pruned studies that "complete";
  `_compute_hpo_result` then crashes on empty `study.best_value` or crowns a
  winner over a tiny survivor set.

**Fix.**
1. `cli/optimize.py`: add `--config` option; when given, load
   `ExperimentConfig.from_file` and reuse `task`, `target_col`, `data_path`,
   `cv_folds`, `metrics`, `random_seed`, `model_overrides` (explicit CLI flags
   override). Echo the resolved settings.
2. `hpo.py::setup_hpo_components`: run the raw frame through the same prep
   modules (`PipelineExecutor(mode=PipelineMode.HPO)` already exists in
   `executor.py::_MODE_FINAL_VARS` — it expects `processed_dataframe`; feed the
   frame through `run_preprocessing` + the existing prep chain, or minimally
   apply the `encoded_df`/null-fill nodes) before `DataAdapter`. Validate on a
   CSV with a string categorical column: HPO must not crash.
3. `hpo.py::_compute_hpo_result`: if completed trials `< max(3, n_trials // 10)`
   → raise `ValueError` listing the first pruned trial's exception instead of
   crowning/crashing.
4. Update `docs/hpo.md` + `docs/index.md` (they were de-rotted on 2026-09-01 —
   keep them true).

**Acceptance.** Integration test (≤100 rows, string categorical column,
lightgbm, `n_trials=3`): HPO completes via `--config`; all-pruned study raises
with the underlying error surfaced.

### W6-T6 — Split coverage check `[V]` — **coordinate with W2**

**Problem.** `verification/split_validation.py:10-40` validates schema, duplicate
`(row_id, fold, role)` memberships, and per-fold train/validation overlap — but
not **coverage**: rows missing from every fold (or present only as `test`)
pass silently, so some rows are never validated. W2 already owns this file
(its brief extends role handling).

**Fix.** (fold into W2's task if W2 is still open; otherwise do here)
Add a coverage assertion per `repeat`: `union of train ∪ validation ∪ test
row_ids == all row_ids`, and every row appears at least once as validation.
Raise with the count of uncovered rows.

**Acceptance.** Unit tests: drop a row from all folds → error; `test`-only row
→ error (or warning, matching W2's role policy); full split passes.

### W6-T7 — Seed parity between legacy CV and medallion splits `[A]` — **coordinate with W2**

**Problem.** For an identical config the two paths split differently:
legacy `iter8 run` CV uses shuffled kfold seeded by `config.random_seed`
(`engine/evaluator.py::get_cv_split` — since 2026-09-01), while the medallion
split (`runtime/plan.py:39`) uses `shuffle=strategy == "stratified"` — i.e.
regression kfold is **unshuffled** on the medallion path. Same config, two
different fold assignments; benchmark numbers are not comparable across paths.

**Fix.** Decide and document one policy in an ADR paragraph (suggested: always
shuffle except `timeseries`, seeded by `config.random_seed`, both paths).
Implement in `runtime/plan.py`. Add a metamorphic test: same data, different
seeds → different fold membership on **both** paths; same seed → identical
membership across paths (the actual parity assertion).

**Acceptance.** Parity test green; `docs/decisions/` ADR recorded; W2's
deterministic-splits task (if still open) is closed or updated to match.

### W6-T8 — CHANGELOG + version bump for the breaking config changes `[V]`

**Problem.** The 2026-09-01 fix pass is behavior-breaking for existing configs:
`extra="forbid"` (unknown keys now fail), `shap_enabled`/`drift_detection`
removed (loud errors), metric/`cv_strategy` validation at parse time,
`primary_metric` ranking semantics. Nothing records this.

**Fix.** `CHANGELOG.md` entry (repo uses Keep-a-Changelog style — check the
file's existing format first) under **Changed/Removed/Fixed**: the list in §0.
Version: `pyproject.toml` uses hatchling dynamic version — check
`[tool.hatch.version]` source (git tags most likely) and cut the next tag per
repo convention rather than editing a version field.

**Acceptance.** `uv run mkdocs build` clean; CHANGELOG renders in next release
notes; no version field drift.

### W6-T9 — "Before you run" guardrails doc section `[V]`

**Problem.** `--check`, `primary_metric`, `ignore_cols`, `data_sample`,
`iter8ml.metrics` plugins, and the fail-fast contract live only in the
`docs/pipeline-architecture.md` config table — too deep for onboarding.

**Fix.** Add a short section to `docs/evaluation.md` (or a new
`docs/guardrails.md` wired into `mkdocs.yml` nav + `docs/index.md` list):
"Validate before you train" (`iter8 run --check`, what it catches, exit codes),
"Choose the ranking metric" (`primary_metric`), "Drop leaky/ID columns"
(`ignore_cols` + the preflight ID warning), "Custom metrics" (entry-point
recipe, 10 lines), "What fails fast now" (parse-time errors list). Cross-link
from `QUICKSTART.md` if the repo root has one.

**Acceptance.** `uv run mkdocs build` clean; every code example in the section
is copy-paste runnable against `examples/*.yaml`.

### W6-T10 — Optional: tracker settings + CI validation job `[A]`

**Problem.** `tracker: wandb|mlflow` now selects the backend (2026-09-01) but
W&B project ("iter8ml") and MLflow experiment name are ctor defaults
(`engine/tracker.py:102,134`); no config path. Also no CI job runs the
AGENTS.md validation suite end-to-end.

**Fix.** `config.py`: `tracker_settings: dict[str, Any] | None = None` passed
as ctor kwargs in `engine/trainer.py::_build_tracker` (validate keys per
backend, fail loud on unknown). CI: add a workflow job running
`uv sync --all-groups --all-extras` + pytest + ruff + mypy
(`--exclude build/`) + `make check-legacy-namespace` + `mkdocs build` —
mirroring the AGENTS.md validation block. `pip-audit` is already a dev dep.

**Acceptance.** Unit test for tracker_settings pass-through (fake tracker);
CI config lint-clean (`actionlint` if available).

## Execution order & parallelism

```
T8 (changelog) ──────────────► anytime after §0 commit
T1 → T2 → T3 (reporting cluster, same files, strict order)
T4 (config+prep) ∥ T5 (HPO)   [disjoint files]
T6, T7 after/with W2          [owned files overlap]
T9 after T1–T4 (documents final behavior)
T10 last (optional)
```

## Gotchas learned in the 2026-09-01 pass (save yourself the rediscovery)

- **Typer `list[str]` options need repeated flags** (`--models a --models b`);
  comma-joined values arrive as one string. The unknown-model guard now rejects
  that loudly — don't "fix" it by splitting on commas silently.
- **`--metrics` is not a CLI option** on `iter8 run`; metrics are config-only
  (by design — parse-time validation lives in pydantic).
- **`data_sample` must stay off the medallion path** — sampling when
  `split_frame is not None` breaks `row_id`↔fold alignment
  (`orchestration/service.py:156-157` passes a split frame).
- **Hamilton optional inputs**: new DAG inputs need `= None` defaults or every
  mode executing the prep module (DRIFT/EXPORT/HPO/INFERENCE via
  `execute(inputs={"df": df})`) must supply them.
- **`config.py` ↔ `engine/evaluator.py` are circular** at module level — the
  established pattern is lazy imports inside validator functions
  (`_raise_if_unknown_model_names`, `validate_task_consistency`).
- **`extra="forbid"` + legacy shim order**: `accept_legacy_flat_keys` (mode
  `"before"`) pops legacy keys before the strict schema sees them; any NEW
  removed key must go in `_REMOVED_CONFIG_KEYS`, not just be deleted.
- **Test fakes must mirror real model ctors** (`**kwargs`): strict test doubles
  broke when `random_seed` started flowing to constructors.
- **Polars `Series.min()` type unions** upset mypy — `min(col.to_list())` is the
  pragmatic pattern (see `verification/preflight.py::_min_class_count`).
- Smoke-test exit codes through pipes: `cmd | tail; echo $?` returns tail's
  code — use `cmd > log 2>&1; echo $?`.

## Validation gate (unchanged from AGENTS.md)

```bash
uv sync --all-groups --all-extras
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy . --exclude 'build/'
make check-legacy-namespace
uv run mkdocs build          # docs changes; 'make docs' needs quarto installed
```

Live smoke (recommended after T1–T4): clean parquet + missing-target +
constant-target + wrong-metric-config runs via `iter8 run --check` / full runs;
assert exit codes 0/1/1/1 and `±std` + calibration legend in
`workspace/current_state.md`.

## Definition of done

Conventions: record ADR-level decisions (T1 policy note, T7 split policy) in
`docs/decisions/` **before** implementing; update `REPORT_LOG.md` newest-first
with AI-assistance disclosure at handoff close; no commit unless the operator
explicitly asks.

- [x] T1 event-log robustness (torn-tail recovery, rotated reads, dedupe) — done 2026-09-02, incl. ADR-0003 addendum + skip_trailing at state-observer/resume call sites
- [x] T2 `compute_lift` returns `None` + baseline-skip warnings — done 2026-09-02 (zero-baseline test flipped to None by design)
- [x] T3 leaderboard task isolation + timestamp `latest_run` — done 2026-09-02 (also fixed latent unscored-first sentinel bug)
- [x] T4 `positive_class` config + README claims corrected — done 2026-09-02 (incl. dtype-guard fix from review, export metadata)
- [x] T5 HPO `--config` + prep routing + min-completed-trials guard — done 2026-09-02 (incl. model_overrides flatten fix from review)
- [ ] T6 split coverage check — **folded into W2** per handoff coordination note (files W2-owned)
- [ ] T7 seed parity policy ADR + metamorphic test — **folded into W2** per handoff coordination note
- [x] T8 CHANGELOG entry covering the 2026-09-01 breaking config changes — done 2026-09-02 ([Unreleased]; hatch-vcs tag convention, no version field drift)
- [x] T9 "before you run" guardrails docs section — done 2026-09-02 (docs/evaluation.md + index.md cross-link)
- [x] T10 (optional) tracker settings + CI validation job — done 2026-09-02 (`tracker_settings` allowlisted per backend + fail-loud validation in `_build_tracker`; `validation-gate` CI job mirroring the AGENTS.md block, enforcing mkdocs/ruff/format on PRs)
- [x] Validation gate green (full suite, ruff, mypy, legacy-namespace, mkdocs) — 855 passed / 3 skipped 2026-09-02; live smoke: --check exit codes 0/1/1/1, full run green with calibration:none. **Known blocker (pre-existing, out of scope → W3):** `calibration: platt|isotonic` fails on in-tree models — `CalibratedClassifierCV` clones the estimator, wrappers lack `get_params`. See REPORT_LOG.md 2026-09-02.
- [x] Statuses inline; no commit unless explicitly asked — tree left uncommitted (operator has not asked)
