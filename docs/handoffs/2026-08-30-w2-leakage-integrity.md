# W2 — Leakage discipline & training integrity

**Date:** 2026-08-30 · **Priority:** P1 (the portfolio story) · **Depends on:** W1-T1 (test
suites touch `engine/hpo.py`) is helpful but not blocking. **Must land before
W3 (owns `data/embedding.py` persistence) and W5 (owns `dataflows/gold.py`
vectorization).**
**Evidence:** findings §1 (Analytics correctness), §2 (Reproducibility,
Computed-but-discarded), §5.
**Files owned by this workstream:** `engine/pipelines/nodes/{train,prep,features}.py`,
`data/{embedding,features}.py` (fit logic), `data/{leakage,quality}.py`
(selection-related bits), `dataflows/{gold,silver,platinum_train}.py`,
`domain/manifests.py` (SplitSpec), `verification/split_validation.py`.

## Goal

Make the framework's central claim — *leakage discipline, proven* — actually
true end-to-end: fold-safe feature construction, a holdout-clean champion
artifact, content-addressed platinum caching, deterministic splits, and honest
manifest fields.

## Task 0 (do first) — Design note, then implement

This workstream changes *reported metrics* (they will go **down** — that is the
point). Before code:

1. Write a short decision record (`docs/decisions/NNNN-fold-safe-feature-fitting.md`,
   Context → Decision → Consequences) choosing per item: **per-fold refit**,
   **fit-on-train-folds-only**, or **documented fold-unsafe (honest-status)**.
   Recommended stance:
   - prep imputation stats → per-fold (cheap, big win);
   - embeddings (`data/embedding.py:140-197,224-274`) → fit on train folds;
     accept the cost, document it;
   - interaction discovery/pruning (`data/features.py:213-346,360-417`) →
     fit-on-train-folds or honest-status if cost is prohibitive;
   - target transform (`data/features.py:67-115`, `prep.py:186-201`) → per-fold.
2. Note in `docs/feature-engineering.md` + `docs/medallion.md` that metrics
   previously included fold leakage and are now conservative.
3. Record the metric shift in `REPORT_LOG.md` (material finding).

The plumbing already exists: `train.py:20-56` (`_fold_indices_from_split`) and
`evaluator`'s `fold_indices` show the seam to extend — nodes need the
fold/role mapping threaded in (executor already resolves `split_frame`,
`executor.py:~252`).

## Tasks

### W2-T1 — Champion artifact excludes holdout rows `[V]`

**Problem.** `_train_one` fits the saved champion on the full `X, y`
(`train.py:~248`) including `role=="test"` rows; preprocessing stats saw them
too. `verification/split_validation.py` accepts the `test` role, so with the
medallion split wired in, the persisted artifact is contaminated for any future
test evaluation.

**Fix.** Fit the final artifact on train+validation rows only (exclude
`role=="test"`), via the existing fold-index machinery. If no split_frame is
passed (legacy `session.run` path), keep current behavior but document it.
Also extend `verification/split_validation.py` to check the `test` role
(emptiness + train∩test overlap) — the checks exist for train∩validation only
(`:26-36`).

**Acceptance.** Test: build a split_frame with a test role, assert the saved
model's `n_rows` / fit scope excludes test rows (e.g., via a spy or row-count
assertion); split validator rejects overlapping test rows.

### W2-T2 — Fold-safe preprocessing (prep.py)

Median/mode imputation and target-transform stats fit on all rows
(`prep.py:65-77,186-201`). Refit per fold's train slice; apply to the
validation slice. Keep `@config.when` structure — no behavior branching beyond
the existing variants (ADR-0002). Delete the unreachable no-Hamilton fallback
while here (`prep.py:231-283`, findings §3) — replace with the `hamilton_stub`
pattern used by `features.py:221-229`/`drift_detection.py:124-135`.

**Acceptance.** Canary test: a synthetic dataset where one column's median
differs sharply between folds; assert per-fold imputation values are used.

### W2-T3 — Fold-safe supervised features (embedding + interactions + pruning)

Embeddings (`data/embedding.py:140-197,224-274`), interaction discovery and
pruning (`data/features.py:213-346,360-417`), leakage-audit-adjacent selection
— all fit on full data. Implement the Task-0 decision. Minimum viable: pass
train-fold indices into the fit stage and evaluate on validation slices;
per-fold refit only where cheap. Add the keep-at-least-one guard in
`prune_features` (`features.py:398-405`) and an honest-status note for
selection bias (`features.py:270-274`) if deferring.

**Acceptance.** Canary test demonstrating the previous leakage (metric drop
after fix, or a canary column that only leaks under the old path); no
0-column X possible after pruning.

### W2-T4 — Platinum content-addressing + config in the manifest `[V]`

**Problem.** `dataflows/platinum_train.py:20-34`: `pid = product_id("platinum",
experiment_name, gold.product_id, run_id)`; `store.exists(pid)` returns the old
manifest without comparing `results`; `specification_digest` covers only
`{run_id, experiment}` — resolved training config is not recorded anywhere.

**Fix.** Include a digest of the resolved training config (and/or results) in
the pid or check it on hit and treat mismatch as a cache miss (re-materialize).
Record resolved config in manifest `metadata`. Aligns with AGENTS.md: "config,
data hash, and metrics recorded."

**Acceptance.** Test: same `run_id`, different results → cache miss,
re-materialized; same inputs → hit. Manifest contains the resolved config.

### W2-T5 — Deterministic splits + honest gold manifest fields `[A]`

- `dataflows/gold.py:29,62-66`: `shuffle=True` with `random_seed=None`
  (`domain/manifests.py:106` default) → nondeterministic folds. Require a seed
  when `shuffle=True` (pydantic validator on `SplitSpec`) or derive one from
  the frame digest; record the effective seed in the manifest.
- `gold.py:133-136`: `split_overlap` computed after the raise-gate → always
  False; `overlap_checks_passed` always True; `temporal_checks_passed`
  (`gold.py:195-214`, field `domain/manifests.py:236`) can never be False.
  Either record real outcomes (catch + record + gate) or drop the fields.
- `gold.py:117-121`: cached-path miss raises bare `StopIteration` → raise
  `ArtifactError`.
- `dataflows/silver.py:30`: supplied contract silently drops `target_col` from
  the digest → merge contract + explicit target, don't short-circuit.

**Acceptance.** Two runs with same seed → identical fold assignment (assert
frame equality); validator rejects `shuffle=True` without seed; manifest
fields reflect a forced overlap failure; silver pid differs when target_col
differs.

### W2-T6 (small) — Embedding input guards (fit-side)

`data/embedding.py:22-49,91-92`: integer features >50 distinct values are
silently treated as categorical and **dropped** from X; `:113,115`
`astype(X.dtype)` truncates float embeddings on int X. Restrict
high-cardinality detection to String/Categorical dtypes (opt-in for ints);
upcast before `astype`. (W3 owns embedding *persistence*; land this first.)

**Acceptance.** Unit test: int column with 60 distinct values remains in X;
float embeddings survive on int feature matrices.

## Suggested subagent orchestration

1. `planner` subagent: produce the Task-0 decision draft from this file +
   `docs/decisions/0003` (medallion) for the ADR; coordinator approves.
2. `reviewer` subagent (read-only): map every full-data fit site in
   nodes/`data/` (there may be more than the ones cited) before T2/T3 start.
3. `worker` subagents: T1 (train.py + split_validation) → T2 (prep.py) → T3
   (data/features + embedding) sequentially — heavy file overlap, do not
   parallelize. T4 and T5 are disjoint files (`platinum_train.py`,
   `gold/silver/manifests`) and can run as a second parallel `worker`.
4. `reviewer`: diff review focused on "did any node keep full-data fitting?"
5. Coordinator: validation gate, REPORT_LOG metric-shift entry, doc updates.

## Gotchas

- **Metrics will get worse after this lands.** That is expected and is the
  story; update docs/leaderboard copy so the change reads as integrity, not
  regression. Note before/after numbers in `REPORT_LOG.md`.
- The canary tests are the deliverable — without them the fix is unverifiable.
- Keep node code branch-free (ADR-0002): fold-awareness enters via inputs,
  not `if` variants inside `@config.when` bodies.
- `prep.py` fallback deletion overlaps W4's dead-code list — W4 will re-verify;
  fine to delete here first.
- Gold/silver changes invalidate existing caches (new digests) — expected;
  mention in the run notes.
- Evaluation cost grows with per-fold refits; keep the demo dataset sizes from
  the roadmap constraints (~RAM 15 GB binding) and measure one full
  `session.run` before/after.

## Validation gate

Same as W1 (full gate), plus:

```bash
uv run pytest tests/unit/test_split_wiring.py tests/unit/test_medallion_contracts.py -q
ITER8ML_WORKSPACE=/tmp/w2check uv run iter8 run --help   # smoke
```

## Definition of done

- [ ] Task-0 ADR accepted and linked from `docs/feature-engineering.md` + `docs/medallion.md`
- [ ] T1–T6 implemented with acceptance tests (canaries included)
- [ ] Before/after metric comparison recorded in `REPORT_LOG.md`
- [ ] Full validation gate green
- [ ] Statuses updated inline in this file
- [ ] No commit unless the user explicitly asks
