# W1 — Config & API surface correctness

**Date:** 2026-08-30 · **Priority:** P0 · **Depends on:** nothing (can start immediately)
**Evidence:** `docs/handoffs/2026-08-30-src-audit-findings.md` §1 (Critical)
**Files owned by this workstream:** `config.py`, `engine/hpo.py`,
`engine/hpo_warmstart.py` (signature only), `engine/models/factory.py`,
`cli/export.py`, `cli/optimize.py`, `cli/main.py` (registry write),
`session.py`, `services/export.py`, `services/registry.py`,
`services/reporting.py` (helpers only), `services/mcp.py` (HPO call site).

## Goal

Eliminate every path where the framework **silently ignores user intent or
produces silently wrong results**: inert config, wrong-direction HPO, uncapped
libgomp on the HPO seam, dropped `--target`, cross-metric champion comparison,
and silent export fallbacks.

## Tasks (execute in order; each is one subagent-sized unit)

### W1-T1 — HPO optimization direction from the central registry `[V]`

**Problem.** `engine/hpo.py:284,289` hardcode `direction="maximize"`; the
objective returns the first score in the dict (`hpo.py:~310`). Regression
defaults are `metrics=["rmse","r2"]` (`config.py:356-359`) → `iter8 hpo --task
regression` **maximizes RMSE**. `services/reporting.py:45-62` owns
directionality but is bypassed.

**Fix.**
1. Add `metric_direction(metric_name) -> str` (or reuse
   `metric_higher_is_better`) in `services/reporting.py`.
2. `optimize_model(...)`: accept `metrics: list[str] | None = None`; primary
   metric = `metrics[0]` when provided (fall back to first returned key);
   `direction = "maximize" if metric_higher_is_better(primary) else "minimize"`;
   pass it to both `create_warmstarted_study` (already has the param) and
   `create_study` (thread it through, `hpo.py:29-36`).
3. Objective returns `scores[primary]` when present, else first key.
4. Update call sites: `cli/optimize.py:28-40`, `services/mcp.py:98-129` to pass
   the task-appropriate metrics.

**Acceptance.** Integration test (small synthetic regression frame, ≤100 rows,
lightgbm, `n_trials=2`): best trial's rmse ≤ worst trial's rmse and study
direction is `minimize`. Existing HPO tests updated.

### W1-T2 — OpenMP cap at the model seam (ADR-0004/0006) `[V]`

**Problem.** The cap runs only in `Trainer.__init__` (`engine/trainer.py:33`).
`get_model_class` (`engine/models/factory.py:51-61`) imports GBDT modules
whose top-level imports load libgomp (`lightgbm_model.py:2`,
`xgboost_model.py:3`, `catboost_model.py:7`) — uncapped on `iter8 hpo`,
MCP `run_hpo`, and any direct factory call. This is the exact hybrid-core
libgomp deadlock class ADR-0004 exists for (see phase-2 handoff learning #1).

**Fix.** Call `HardwareProfile.configure_omp_threads()` at the top of
`get_model_class` before `importlib.import_module` (the method is idempotent /
re-entrant — see `tests/unit/test_config.py:349`). Do **not** remove the
`Trainer.__init__` call. Add a one-line backstop: GBDT wrappers accept
`num_threads`/`nthread`/`thread_count` in params (already pass-through).

**Acceptance.** Test that `get_model_class("lightgbm")` triggers the cap
(monkeypatch `configure_omp_threads` with a recorder; assert called).
Note the fix in `docs/hpo.md` + `docs/models.md` ("safe by default at the
factory").

### W1-T3 — Inert config: delete or wire `[V]`

**Problem.** `HPOConfig.run/n_trials` (`config.py:66-71`, validated
`:357-364`) and `QualityConfig.run_audit/auto_clean_noise/noise_quality_threshold`
(`config.py:73-80`) are never read — the executor resolves behavior from
pipeline step params (`executor.py:80-84`, legacy shim `config.py:278-284`).
`StepName.HPO` (`config.py:92`) has no DAG module. Setting these in YAML is a
silent no-op — the worst failure mode for a config-driven framework.

**Decision (recommended): delete, don't wire.** Wiring would create a second
source of truth beside step params (ADR-0002's point). Remove the fields, the
`run_hpo` validation branch (`config.py:357-364`), `StepName.HPO`, and the
corresponding shim mappings (`config.py:278-284`); emit a clear
`ValueError`/warning if a user config still contains them (one release of
deprecation messaging is enough for a pre-1.0 portfolio). Grep `src tests docs
notebooks demo` for `run_hpo|hpo_n_trials|run_audit|auto_clean_noise` before
removing; update every fixture/README hit.

**Acceptance.** `grep -rn "run_hpo\|run_audit" src/` returns no config-field
hits; `tests/unit/test_config.py` updated; a YAML containing the removed keys
raises a clear error naming the replacement (step params).

### W1-T4 — `iter8 export --target` end-to-end `[V]`

**Problem.** `cli/export.py:14,19` accepts `--target` and drops it;
`session.export(key, output_dir)` has no `target_col` param
(`session.py:96-98`). Consequence: exported predictor doesn't drop the label
column → positional `X = df.to_numpy()` shifts every feature → **silently
wrong predictions** (`services/export.py:93-99`).

**Fix.** Thread `target_col: str | None` through `cli/export.py` →
`session.export` → `ExportService.export` → metadata + predictor template
(drop the column when set; keep positional behavior unchanged when `None`).
While in `services/export.py`, fix the silent fallbacks (`:297-299`: raise on
unknown model instead of defaulting to CatBoost; `:302`: require the task key).

**Acceptance.** Roundtrip test: train via `session.run` with a target, export
with `--target`, run `predictor.py` on a CSV that **includes** the label
column, assert predictions match the no-label-column run. Unknown-model export
raises.

### W1-T5 — Registry champion comparison: same metric only `[V]`

**Problem.** `services/registry.py:174-179` compares candidate vs incumbent
with `metric_value_is_better(metric_name, score, existing_score)` — the
incumbent's metric is stored (`"metric_name"` key) but never consulted.
`resolve_primary_score` can pick any numeric metric, so `accuracy=0.90`
displaces `roc_auc=0.83`.

**Fix.** Displace only when `existing.get("metric_name") == metric_name`
(handle legacy entries with `metric_name=None`: treat as compatible-with-nothing
→ keep incumbent, or migrate on first write). Log/surface "metric mismatch,
champion retained" rather than silently failing to promote.

**Acceptance.** Unit test: same key, incumbent `roc_auc=0.83`, candidate
`accuracy=0.99` → retained; same-metric better candidate → promoted.

### W1-T6 — `iter8 registry promote` + registry access via the service

**Problem.** `cli/export.py:33,44` advertises "show or promote"; `promote`
falls through and exits 0. Registry JSON is also read/written raw outside the
service (`cli/export.py:39`, `cli/main.py:31`), bypassing the file lock.

**Fix.** Wire `iter8 registry promote <run_id> <key>` to `RegistryService.promote_run`
(MCP already does the equivalent via `session.promote`); route `show` and the
`--force-reset-registry` write through `RegistryService` methods.

**Acceptance.** `iter8 registry promote` performs a real promotion (registry
updated, artifact path recorded); concurrent-show safety unchanged.

### W1-T7 (small) — HPO on baselines errors clearly `[A]`

`engine/hpo.py:188-189`: `naive_baseline`/`linear_baseline` pass name
validation but `getattr(ModelConfigs(), model)` raises `AttributeError`.
Reject non-configurable models with `ValueError` naming the reason. One test.

## Suggested subagent orchestration

1. `reviewer` (read-only): re-verify findings §1 for W1's files; confirm no new
   callers appeared. One agent, one prompt quoting the findings file.
2. Implement T1+T2 (hpo/factory/reporting) with a `worker` subagent — these
   touch disjoint files from T3.
3. In parallel (different files): `worker` for T3 (config.py + tests).
4. Then `worker` for T4 (cli/session/services.export), T5+T6
   (registry/cli/main), T7.
5. `reviewer`: diff review after each pair of tasks.
6. Coordinator: validation gate + bookkeeping (see below).

## Gotchas

- `uv run` everything; never bare `python` (AGENTS.md).
- `uv run mypy . --exclude 'build/'` — stale `build/` breaks mypy locally.
- Don't remove the `Trainer.__init__` OMP call when adding the factory one.
- `create_warmstarted_study` already takes `direction` — don't double-default.
- Registry entries written before `metric_name` existed may have `None` — test
  that path.
- Keep `reporting.py` the only home of direction logic; W1-T1 must import from
  it, not re-implement.

## Validation gate (run after each task; full gate before handoff close)

```bash
uv sync --all-groups
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy . --exclude 'build/'
uv run pytest tests/unit/test_hpo.py tests/unit/test_cli.py -q   # affected suites
```

## Definition of done

- [x] T1–T7 implemented with the acceptance tests above (→ done 2026-08-30; plus T-guard: warmstart metric-compatibility — reviewer-found hole T1 made reachable)
- [x] Full validation gate green (→ 2026-08-30: unit 748/3 skipped, integration 40, e2e 1, ruff/mypy/legacy-namespace clean. NOTE: gate needs `uv sync --all-groups --all-extras` — bare `--all-groups` prunes extras and breaks optuna/xgboost)
- [x] `REPORT_LOG.md` entry (→ 2026-08-30; incl. hybrid-CPU libgomp live-lock root cause: `OMP_WAIT_POLICY=passive`, not thread count)
- [x] `docs/hpo.md`, `docs/models.md`, `README.md` updated where CLI behavior changed (→ direction resolution + thread safety in hpo/models; `registry promote` line in README)
- [x] Checkbox updates in this file (statuses inline: `→ done <date>`)
- [x] No commit unless the user explicitly asks (→ no commit made; all work uncommitted in tree)
