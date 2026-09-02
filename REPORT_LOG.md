# Report Log

Material findings, mid-iteration discoveries, and AI-assistance disclosure
(required by `AGENTS.md` → *Engineering conventions*). Newest entries first.

---

## 2026-09-02 — W6 report trustworthiness & audit follow-through (T1–T5, T8, T9)

**What changed** (handoff: `docs/handoffs/2026-09-01-w6-report-trust-and-followthrough.md`;
work executed by AI coding agents under operator direction — disclosure per AGENTS.md)

- **T1 event-log robustness**: `iter_events`/`load_events` gained
  `on_error: "raise" | "skip" | "skip_trailing"` (torn trailing writes recover,
  mid-file corruption still raises, all skips warn with line numbers);
  `ReportService` reads live log **and** rotated backups (newest first) with
  dedupe on `(run_id, model, artifact_path)` keeping the latest timestamp.
  `state_observer._load_all_events` and trainer resume now also use
  `skip_trailing` (they were the two brick symptoms in the brief).
  ADR-0003 addendum records the flat-path-resumes-from-events policy.
- **T2 `compute_lift` honesty**: returns `None` for missing metrics or a 0
  baseline (was: fabricated 0.0); `_train_one` filters `None` pairs;
  baseline skips now log a warning with the reason.
- **T3 leaderboard task isolation**: sort key composes unscored-sentinel-last
  (this also fixed a latent bug — the fix-pass key put unscored entries FIRST
  under `reverse=True`; no test had covered it), contiguous task blocks,
  best-first within task, newest-first timestamp tiebreak; `latest_run` is
  max-timestamp (try/except fallback to load order); `build_report(task=...)`
  and `session.leaderboard(task=...)` filters; mixed-task leaderboards say so
  in the `current_state.md` heading.
- **T4 `positive_class`**: new classification-only config field, threaded
  through `_DIRECT_FIELDS` → new prep node `target_oriented_df` (positive → 1,
  no-op without config, loud errors for unknown values / non-binary targets);
  `encoded_df` now dtype-guards so the oriented integer target skips the
  categorical cast; preflight notes/validates it; export metadata records it
  (`iter8 export --positive-class`); README claims corrected (no automatic
  class-weighting exists — binary `scale_pos_weight` / multiclass
  `class_weight` via `model_overrides`; transformers are in-tree).
- **T5 HPO seam**: `iter8 hpo --config` (flags override config; per-model
  `model_overrides` flattened to fixed params; `primary_metric` optimized
  first); `setup_hpo_components` routes the raw frame through the training
  prep chain via new `PipelineExecutor.run_prep` (string categoricals encode
  instead of crashing GBDT ctors; `ignore_cols`/`positive_class` parity);
  all-but-pruned studies raise with the first underlying error instead of
  crashing on empty `best_value` (threshold `min(n_trials, max(3, n//10))`);
  docs/hpo.md + docs/index.md updated true.
- **T8 CHANGELOG**: Keep-a-Changelog `[Unreleased]` entry covering the
  breaking config surface (extra=forbid, removed keys, parse-time validation,
  primary_metric semantics) plus the W6 behavior changes. Version is
  hatch-vcs — next release tag per repo convention, no field edit.
- **T9 guardrails docs**: "Validate before you train" section in
  `docs/evaluation.md` (--check, primary_metric, ignore_cols,
  positive_class, iter8ml.metrics recipe, fail-fast list); cross-linked from
  `docs/index.md`.
- **T6/T7**: not done here — folded into W2 per handoff (files W2-owned).
- **T10**: `tracker_settings` config field with per-backend allowlist validation
  (fail-loud on unknown keys, jsonl `log_path` deliberately excluded as
  workspace-managed) threaded through `_build_tracker`; `validation-gate` CI
  job in `.github/workflows/ci.yml` mirroring the AGENTS.md block
  (`uv sync --all-groups --all-extras`, full pytest, ruff check + format,
  mypy `--exclude build/`, legacy namespace, mkdocs — enforcing docs/format
  on PRs, which docs.yml's deploy-only trigger did not). Unit tests:
  `tests/unit/test_tracker_settings.py` (7).

**Review**: each cluster was reviewed by a separate reviewer subagent against
the brief (T1+T2, then T4+T5). Reviewer caught two integration blockers that
node-level tests missed, both fixed: (a) `encoded_df` re-cast the oriented
integer target to Categorical → InvalidOperationError — fixed with a dtype
guard + DAG-level regression test; (b) `iter8 hpo --config` passed the nested
per-model `model_overrides` dict unflattened into model ctors → every trial
pruned — fixed by flattening for the resolved model + CLI-level regression
test.

**Validation**: full gate green — 855 passed / 3 skipped (baseline 820),
ruff/format/mypy/legacy-namespace/mkdocs clean. Live smoke (`/tmp/w6smoke`):
`iter8 run --check` exit codes 0/1/1/1 for clean / missing-target /
constant-target / wrong-metric configs; full run (calibration: none) exits 0
with `current_state.md` rendering Latest Run + leaderboard; `cv_std` present
in events.

**Material finding — pre-existing, OUT OF W6 SCOPE, needs W3**: any run with
`calibration: platt|isotonic` on an in-tree model fails: `CalibratedClassifierCV`
sklearn-clones the estimator and the framework's model wrappers don't implement
`get_params` ("Cannot clone object '<LightGBMModel>' ... "). Reproduced in the
smoke: platt → run exits 1; none → exits 0. The bug predates W6 (uncommitted
2026-09-01 changeset only added `random_seed` plumbing; cloning was always
required with `cv=int`). Suggested direction for W3: prefit-style calibration
(`cv="prefit"` + held-out split, or `FrozenEstimator` on sklearn ≥1.6) or
sklearn-compatible param delegates on the wrappers.

---

## 2026-09-01 — pipeline/config audit fixes: fail-fast guards, honest knobs, report integrity

**What changed** (audit found via scout + 2 reviewer subagents; all load-bearing
findings verified against source before implementation)

- **Fail-fast on misconfiguration**: `ExperimentConfig` now `extra="forbid"`
  (typos like `cv_fold: 10` fail at parse), metrics validated against the
  task's registry at parse time (was: per-model `KeyError` swallowed mid-run),
  `stratified` CV rejected for regression at parse time, `--models` validated
  before overwriting, `--task` parse errors exit cleanly. All-models-fail now
  **fails the run** (`ModelFitError` from `training_state`, `iter8 run` exits 1)
  — previously a fully misconfigured run burned all prep/FE compute and exited 0
  with an empty leaderboard.
- **Reproducibility repaired**: `config.random_seed` now reaches CV fold
  splitters (`get_cv_split(seed)`), model construction (`random_seed` ctor
  param / params convention), calibration, and data sampling — it previously
  only reached the medallion split that `iter8 run` never used. Fold std is
  captured and reported (`evaluate_with_std`, `cv_std` in events/state/CLI as
  `mean ±std`). `experiment_started` records `data_digest` +
  `library_versions` + row counts on the flat path.
- **Single ranking metric**: new `primary_metric` config field (default
  `metrics[0]`, must be a member of `metrics`) — one rule for lift, leaderboard,
  and registry promotion (was: three disagreeing conventions). Unscored-sentinel
  entries sort last instead of outranking lower-is-better results.
- **Honest knobs**: `--quick`'s `data_sample` is now actually applied (was:
  validated-but-inert, trained on 100%); `shap_enabled`/`drift_detection`
  removed with loud deprecation errors (drift → `iter8 drift`, HPO → `iter8 hpo`);
  `tracker` config field now selects the backend (JSONL/W&B/MLflow).
- **Calibration honesty**: `model_completed` events + leaderboards mark
  pre-calibration CV scores with an asterisk legend; requested-but-unapplied
  calibration logs a warning instead of silently downgrading; `CalibratedModel`
  accepts `random_seed`.
- **New pre-run guardrail**: `iter8 run --check` — side-effect-free config↔data
  preflight (`verification/preflight.py`): target presence/nulls/constancy,
  regression target dtype, misdeclared-task warnings, CV feasibility
  (folds vs rows / rarest class), timeseries-without-dates, unknown
  `ignore_cols`, ID-like leakage hints, all-null columns.
- **New config surface for new datasets**: `ignore_cols` (drop IDs/leaky
  columns; unknown columns fail loudly; applied after `row_ids` so medallion
  split alignment is preserved); `iter8ml.metrics` entry-point group for custom
  metrics (`module:func`, optional `func.task` scoping + `func.lower_is_better`);
  regression example config (`examples/house_prices_regression.yaml`).
- **Pre-existing bug fixed en route**: `accept_legacy_flat_keys` replaced the
  default 8-step pipeline with a fragment when any legacy key was used (e.g.
  `run_quality_audit: true` in the flagship example → no FEATURE_ENGINEERING
  step → DAG crash). Legacy keys now seed the full default step list; explicit
  user `pipeline.steps` are still respected.
- **Docs de-rotted**: HPO claims in `examples/`, `configs_README.md`,
  `docs/index.md` now match the real CLI; dead knobs removed from
  `pipeline-architecture.md` config table; `training_features` variant names
  corrected; `get_data_hash` → `dataframe_digest`; `tabblueprint` → `iter8`.

**Validation**: 820 unit/integration/e2e tests pass (incl. 24 new: config
guards, preflight, seeded CV, `evaluate_with_std`, CLI `--check`/fail-fast,
legacy-pipeline regression); ruff + ruff format + mypy (98 files) +
`make check-legacy-namespace` clean; live smoke tests confirm exit codes and
state rendering for clean/missing-target/constant-target/bad-metric/typo-key
runs, classification + regression end-to-end.

---

## 2026-08-30 — W1 (config & API correctness) landed; hybrid-CPU libgomp live-lock fixed

**What changed** (workstream brief: `docs/handoffs/2026-08-30-w1-config-api-correctness.md`;
all findings re-verified read-only by a `reviewer` subagent before implementation)

- **T1 — HPO direction** (`engine/hpo.py`, `services/reporting.py` as sole direction
  home, `cli/optimize.py`, `services/mcp.py`): `optimize_model` now takes `metrics`,
  derives direction via `metric_higher_is_better`, threads it into both
  `create_warmstarted_study` and `create_study`; objective returns the primary
  metric's score. `iter8 hpo --task regression` now **minimizes** RMSE (was
  maximizing it). Result dict records `direction`/`primary_metric`.
- **T2 — OMP cap at the factory** (`engine/models/factory.py`): cap fires at the
  top of `get_model_class` before any libgomp load; `Trainer.__init__` cap kept.
- **T3 — inert config deleted** (`config.py`, `examples/`, `test_config.py`):
  `HPOConfig` + `hpo` field, `QualityConfig.run_audit/auto_clean_noise/noise_quality_threshold`,
  `StepName.HPO`, flat-key delegates/legacy consumption removed (ADR-0002: no
  second source of truth beside step params). Deprecation guard raises a clear
  error naming the replacement for flat + nested removed keys; `run_quality_audit`
  kept (live shim, not inert).
- **T4 — `iter8 export --target`** (`cli/export.py`, `session.py`, `services/export.py`):
  threaded to the already-fixed `ExportService(target_col=...)` seam; roundtrip
  test proves label-column CSVs predict identically. Unknown-model / missing-task
  suffix now raise (validation hoisted before filesystem writes); CatBoost /
  `"classification"` silent defaults gone.
- **T5 — same-metric champion comparisons** (`services/registry.py`): incumbent
  displaced only on metric match (legacy `metric_name=None` retains); same guard
  applied to `update_if_better` and `_select_best_run_event` (anchor metric =
  first event with a real metric). Rejection surfaced as `metric_mismatch` status.
- **T6 — registry via the service**: `iter8 registry promote <run_id> <key>` wired
  to `RegistryService.promote_run` (was a silent fall-through exit 0); `show`,
  `--force-reset-registry` (new locked `RegistryService.reset()`), and
  `reporting._load_registry` no longer touch the registry JSON raw.
- **T7 —** HPO on baseline models raises a clear `ValueError` (was bare `AttributeError`).
- **T-guard (beyond brief, reviewer-found):** warmstart value injection now injects
  `cv_scores[primary_metric]` only when the event actually scored it; otherwise the
  event is skipped and surfaced (`MetricMismatch` warning, `n_skipped_metric_mismatch`).
  Closes a maximize-into-minimize-study hole T1 made reachable.
- **Hybrid-CPU libgomp live-lock (the audit's gotcha #1, now root-caused and fixed):**
  on this Intel Core Ultra 5 225H (Arrow Lake-H, P+E cores, no SMT), LightGBM
  training hung indefinitely in its OpenMP barrier despite `OMP_NUM_THREADS=8` —
  the spin-wait, not the thread count, is the live-lock. `configure_omp_threads()`
  now also sets `OMP_WAIT_POLICY=passive`, and the GBDT wrappers default
  `num_threads`/`nthread`/`thread_count` to the cap (user overrides win). The full
  integration suite went from *hung* to **40 passed in ~40s**.
- Housekeeping: `tests/unit/test_property_leakage.py` got `deadline=None`
  (hypothesis 200ms default vs sklearn CV cost = environmental flake, 3 failures
  unrelated to any diff); 5 stray unformatted files reformatted (pre-existing from
  the 2026-08-21 track); dead `ReportService.registry_path` removed.

**Gate gotcha worth keeping:** `uv sync --all-groups` (as written in AGENTS.md and
the handoff) **prunes extras** — `optuna`/`xgboost`/`lightgbm` are extras
(`full`/`gbdt`/`train`), so the bare command broke the env mid-gate. Use
`uv sync --all-groups --all-extras`. (AGENTS.md/README fix deferred — flagging.)

**Validation:** unit 748 passed / 3 skipped; integration 40 passed; e2e 1 passed;
ruff check + format clean; mypy clean (97 files); `make check-legacy-namespace`
passed. No commit made.

**AI assistance disclosure:** implemented via `reviewer`/`worker` subagents per the
`docs/handoffs/README.md` protocol (one reviewer re-verification, three worker
pairs, two reviewer diff reviews — one 429 retry); coordinator applied the three
reviewer should-fix items, the hybrid-CPU fix, and all bookkeeping directly.

---

## 2026-08-30 — Full `src/` audit → handoff package in `docs/handoffs/`

**What happened**

- Ran a full audit of `src/iter8ml/` (92 files, ~10.7k LOC) via **5 parallel
  `reviewer` subagents** (core/platform, data, engine+pipelines, models,
  services/cli/analysis/orchestration). Two subagents initially hit 429 rate
  limits and were retried successfully.
- The coordinating agent then **re-verified every Critical finding against
  source** (12+ spot checks, all held): HPO maximizes RMSE for regression
  (`engine/hpo.py:284,289`), champion artifact fit on `test`-role rows
  (`train.py`), supervised feature construction leaks across CV folds
  (embeddings/interactions/pruning/prep stats), `iter8 export --target`
  silently dropped → shifted-feature predictions, registry champion compared
  across incomparable metrics, FT-Transformer/TabPFN persistence broken,
  ADR-0004 OMP cap missing at the HPO/factory seam, inert `HPOConfig`/
  `QualityConfig` fields, content-blind platinum cache, events-finalize race.
- **No `src/` changes made yet.** Output is a planning package only:
  `docs/handoffs/` — `README.md` (execution order + subagent orchestration
  pattern), `2026-08-30-src-audit-findings.md` (evidence base with `[V]`/`[A]`
  verification legend), and five self-contained workstream briefs:
  W1 config/API correctness (P0), W2 leakage integrity (P1), W3 model/artifact
  robustness (P2), W5 perf + `iter8 medallion-run` CLI + boundary ADR (P3),
  W4 dead-code/DRY cleanup (last). Added a `docs/README.md` map row.

**Key expected consequence to track:** W2 will make reported CV metrics *drop*
(fold-leakage removal) and W3 early stopping will shift GBDT iteration counts —
both are integrity improvements; before/after numbers to be recorded here by
the implementing agents.

---

## 2026-08-29 — Populated `.planning/codebase/` reference set (parallel subagents)

**What changed**

- Added the internal codebase reference docs the original `AGENTS.md` §5 expected
  but that never existed: `.planning/codebase/{STACK,STRUCTURE,CONVENTIONS,
  TESTING,INTEGRATIONS,CONCERNS,ARCHITECTURE}.md`.
- **Parallel `worker` subagents** (3 launched; 2 succeeded, 1 hit a rate limit):
  - worker → `STACK.md` + `STRUCTURE.md` (from pyproject/uv.lock/mkdocs/Makefile/
    pre-commit/CI + full src + tests tree).
  - worker → `INTEGRATIONS.md` + `CONCERNS.md` (from services/mcp/llm/tracker,
    litellm/mlflow/wandb seams, roadmap risks, handoff learnings, TODO/type-ignore scans).
  - worker (CONVENTIONS+TESTING) **failed on rate limit (429) then aborted on retry**;
    I wrote those two myself from `.planning/STYLE.md` + `pyproject.toml` config +
    verified code facts (OpenMP guard, `RestrictedUnpickler`, `TrainerStatePublishError`,
    `_hamilton_n`/`n` config variants, `conftest.py`/`strategies.py`).
- Added `.planning/codebase/ARCHITECTURE.md` index; re-pointed `CLAUDE.md` (restored
  the codebase-reference line) and added a row to `docs/README.md` map.

**Findings surfaced by subagents (worth tracking)**

- `mypy .` currently fails on this checkout due to a stale `build/lib/iter8ml/` copy
  (`Duplicate module named iter8ml`) — documented as a known gotcha; fix is
  `mypy . --exclude 'build/'` or delete `build/`.
- `ExperimentConfig.tracker` (`config.py:193`) is declared but **never wired** to
  W&B/MLflow construction — setting `tracker="wandb"` silently does nothing (dead
  surface). `litellm`/`mlflow` are hard base deps though no core importer uses them;
  `gitpython`/`cryptography`/`python-multipart` likewise unverified. This deviation
  from the extras-gating rule may warrant an ADR — flagged in `CONCERNS.md` (R2).

**AI assistance disclosure:** the codebase audit + doc generation was performed by
`worker` subagents at the user's direction; the CONVENTIONS/TESTING pair was
written directly by the AI agent after the subagent rate-limited. All `file:line`
claims were spot-checked against the actual tree.

---

## 2026-08-29 — Guidance rewrite + ADR/docs restructure

**What changed**

- `AGENTS.md` rewritten in the four-section *Project guidance* format
  (purpose & decision order → architectural boundaries → engineering
  conventions → validation), grounded in the accepted ADRs and the real
  validation commands (`uv run pytest/ruff/mypy`, `make check-legacy-namespace`).
- Fixed dangling references: old `AGENTS.md` §5 and `CLAUDE.md` pointed at
  `.planning/codebase/*.md`, which does not exist. Both now point at real docs
  (`ARCHITECTURE.md`, `docs/README.md`, `docs/decisions/`, `.planning/STYLE.md`).
- ADRs split out of `docs/design-decisions.md` (verbatim) into
  `docs/decisions/0001…0006-*.md` + `0000-adr-template.md` + index/process
  `README.md`; `docs/design-decisions.md` is now a redirect stub.
- New: `docs/README.md` (documentation map), `docs/plan/deferred-research.md`
  (consolidated deferred work from roadmap/ADRs/handoffs), `docs/archive/README.md`
  (archive policy), this log.
- mkdocs nav now exposes the six ADRs; `README.md` + `docs/index.md` link to
  `docs/decisions/README.md`.

**Findings worth keeping**

- Decision order is now explicit: roadmap/handoff digests → accepted ADRs →
  topic docs; superseded material only in `docs/archive/`.
- Deferred items had no single home (scattered across ADR-0003 "honest status",
  phase2-handoff pivots, roadmap "explicitly deferred") — now consolidated in
  `docs/plan/deferred-research.md`.

**AI assistance disclosure:** the restructure (AGENTS.md rewrite, ADR split,
new docs) was proposed and executed by an AI coding agent at the user's
direction; ADR bodies are verbatim splits of the user's prior
`docs/design-decisions.md`.
