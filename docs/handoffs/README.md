# Handoffs — post-audit workstreams (2026-08-30 → 2026-09-01)

**Entry point for any agent continuing after the 2026-08-30 full-`src/` audit
and the 2026-09-01 pipeline/config audit fix pass.**
Each file below is a self-contained brief: verified findings with `file:line`,
task breakdown sized for subagent delegation, acceptance criteria, gotchas,
and a validation gate. Agent behavior rules live in root `AGENTS.md`.

> This directory is **agent-facing working state**, not part of the mkdocs
> site (not in `mkdocs.yml` nav). Superseded handoffs move to
> `docs/archive/` per the policy in `docs/README.md`.

## Files

| File | What it is |
|---|---|
| [`2026-08-30-src-audit-findings.md`](2026-08-30-src-audit-findings.md) | Evidence base: full audit findings, `[V]`/`[A]` verification legend, dead-code table, DRY/perf/boundary lists. All workstreams cite this. |
| [`2026-08-30-w1-config-api-correctness.md`](2026-08-30-w1-config-api-correctness.md) | P0: HPO direction, OMP cap at the factory, inert config, `--target`, registry compare, export fallbacks. **Landed (uncommitted) 2026-08-30 — see REPORT_LOG.md.** |
| [`2026-09-01-w6-report-trust-and-followthrough.md`](2026-09-01-w6-report-trust-and-followthrough.md) | P1: **current entry point.** Event-log robustness, `compute_lift` honesty, leaderboard task isolation, `positive_class`, HPO config seam, split coverage, seed parity, changelog, guardrails docs, CI. Includes §0 state-of-the-tree (two uncommitted changesets) and gotchas from the 2026-09-01 fix pass. |
| [`2026-08-30-w2-leakage-integrity.md`](2026-08-30-w2-leakage-integrity.md) | P1: fold-safe feature fitting, holdout-clean champion, platinum content-addressing, deterministic splits. |
| [`2026-08-30-w3-model-artifact-robustness.md`](2026-08-30-w3-model-artifact-robustness.md) | P2: FT/TabPFN persistence, GBDT early stopping, event/sqlite/writer leaks, quality/adapter guards. |
| [`2026-08-30-w5-perf-and-surface.md`](2026-08-30-w5-perf-and-surface.md) | P3+P1: digest/split vectorization, `iter8 medallion-run` CLI, boundary ADR. |
| [`2026-08-30-w4-deadcode-dry-cleanup.md`](2026-08-30-w4-deadcode-dry-cleanup.md) | P3, **last**: dead-code deletion (fresh sweep) + DRY collapses. |

## Execution order & parallelism

```
W1 (landed, uncommitted) ──► W6 (current entry point) ──┬──► W2 ──► W3 ──► W5 ──► W4
                                                        │ (T6/T7 coordinate with W2)
```

**Simple rule: W1 is done. Start at W6 (commit the tree first — see W6 §0).
W6-T1→T2→T3 strictly in order; T6/T7 fold into or coordinate with W2;
after W6, resume W2 → W3 → W5 → W4.**
**Status: W1 → done 2026-08-30** (validation gate green, see `REPORT_LOG.md`);
next pickup: **W2** (Task-0 design ADR first).
Parallelism is possible (W1 and W3 share no files; W2-T4/T5 are disjoint from
W2-T1..T3/T6) but file ownership is the constraint — when in doubt,
sequential:

- `config.py`, `session.py`, `services/export.py`, `cli/export.py` → W1
- `engine/pipelines/nodes/*`, `data/embedding.py` (fit side), `dataflows/*` → W2
- `engine/models/*`, `domain/events.py`, `storage/*`, `utils/io.py` → W3
- `data/embedding.py` (persistence) → W3 only **after** W2
- `domain/hashing.py`, `dataflows/gold.py` (vectorizing), `cli/` (new command) → W5, after W2
- everything else → W4, after all of the above

Each handoff closes with checkboxes — mark them as you complete tasks so a
crashed/rotated agent can resume from this directory.

## How to run a workstream with subagents

Pattern that worked for the audit itself (and for `.planning/codebase/`):

1. **Re-verify first (read-only).** Spawn a `reviewer` subagent with the
   workstream file + findings file to confirm the cited findings still hold
   (`file:line` may have drifted if earlier streams landed). Cheap insurance.
2. **Implement in task order.** Spawn a `worker` subagent per task (or per
   2–3 tightly-coupled tasks) with the task section verbatim as its brief —
   each task section is written to stand alone. Respect the file-ownership
   rules above; never run two `worker`s on overlapping files.
3. **Review the diff.** Spawn a `reviewer` subagent on the accumulated diff
   after each task pair. Bar for W4: mechanical/test-only changes.
4. **Coordinator duties stay with you:** validation gate, `REPORT_LOG.md`
   entry, doc updates, checkbox updates, user approval for any scope change.
5. If a subagent hits a 429 rate limit, retry once; if it fails again, do the
   task yourself (precedent: `.planning/codebase/` population).

## Non-negotiables (from `AGENTS.md`)

- `uv run <cmd>` always; never bare `python`.
- Read before edit; `edit` over `write`; smallest change that proves the point.
- **No commits unless the user explicitly asks.**
- Any change to an architectural boundary needs an ADR **before** implementing
  (W5-T5 is exactly this; W2's Task-0 is a design ADR).
- Every change keeps runs reproducible: resolved config + hashes in workspace
  state.
- Update `REPORT_LOG.md` with material findings; disclose AI assistance.
- Validation gate before any handoff close:
  `uv sync --all-groups && uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy . --exclude 'build/' && make check-legacy-namespace`
  (`--exclude 'build/'` needed locally; stale `build/` breaks mypy).

## Known gotchas that cost the last agent real time

1. **OpenMP/libgomp hangs** on hybrid CPUs when GBDT libs load uncapped —
   W1-T2 fixed the seam, and the root cause turned out to be the **spin-wait**,
   not just the thread count: `configure_omp_threads()` now also sets
   `OMP_WAIT_POLICY=passive`, and GBDT wrappers pin `num_threads`/`nthread`/
   `thread_count` to the cap. Keep both layers when touching the model seam.
2. **mypy `build/` duplicate-module** (above).
3. **No local Quarto** — pre-commit `quarto-render` blocks commits with staged
   `.qmd`; use `--no-verify`; CI renders (only relevant if you touch
   notebooks).
4. **Metrics will drop after W2** — that is the point (removing leakage
   inflation). Frame it as integrity in docs/changelog, with numbers in
   `REPORT_LOG.md`.
5. **`uv sync --all-groups` prunes extras** — `optuna`/`xgboost`/`lightgbm` are
   extras (`full`/`gbdt`/`train`), not groups; the bare gate command breaks the
   env (ModuleNotFoundError mid-suite). Use `uv sync --all-groups --all-extras`
   (AGENTS.md validation block updated 2026-08-30).

## Provenance

- Audit: 5 parallel `reviewer` subagents over 92 files / ~10.7k LOC
  (2026-08-30); dead-code claims grep-verified at audit time.
- Criticals marked `[V]` in the findings file were re-verified against source
  by the coordinating agent on 2026-08-30; `[A]` items are auditor-verified
  only — re-confirm before acting.
- This handoff package was authored by the same coordinating agent; work is
  not started. Nothing in `src/` has changed as a result of the audit yet.
