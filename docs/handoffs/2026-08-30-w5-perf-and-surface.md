# W5 — Performance & flagship surface

**Date:** 2026-08-30 · **Priority:** P3 (perf) + P1 (medallion CLI visibility) ·
**Depends on:** W2 (owns `dataflows/gold.py` + `data/embedding.py` fit-side —
land those first). W4 runs after this stream.
**Evidence:** findings §5 (Performance), §2 (Visibility gap), §6 (ADR notes).
**Files owned by this workstream:** `domain/hashing.py`,
`dataflows/gold.py` (vectorization), `data/embedding.py` (perf only),
`data/features.py`/`data/leakage.py` (copy optimizations),
`storage/local.py` (index), `cli/` (new medallion command),
`session.py` (wiring), `docs/decisions/` (new ADR).

## Goal

1. Remove the Python-loop hot spots that cap the medallion path as datasets
   grow (the roadmap's binding constraint is RAM/CPU on a laptop — the story
   must survive 10⁵-row data).
2. Make the **flagship medallion lifecycle visible and demoable**: a CLI
   command, not a test-only API.
3. Record the boundary deviations the audit surfaced so `AGENTS.md`'s rules and
   reality match.

## Tasks

### W5-T1 — Vectorize the digest path (highest impact)

**Problem.** `domain/hashing.py:24-54`: `dataframe_digest` materializes a
sorted list of per-row digests (O(n) memory); `row_ids` does two
`canonical_json` serializations + two sha256 **per row** in a Python loop via
`iter_rows(named=True)`. Invoked on full frames in bronze (`bronze.py:20,41`),
silver (`silver.py:33,63`), and again in gold (`gold.py:128`).

**Fix.** Hash occurrence-inline (suffix the key into the hashed payload) and
process in chunks (e.g., 10k rows/batch) or use a polars-native row hash.
**Determinism is contractual** (deep-checksum resume, ADR-0003): the *digest
values* may change only if you accept a one-time cache invalidation across all
workspaces — prefer keeping the per-row digest format byte-identical and only
removing the Python overhead; if the format must change, bump the graph
version and document migration in `docs/medallion.md` (see deferred-research
"migration tooling").

**Acceptance.** Benchmark script (committed under `benchmarks/` or as a test
fixture): digest of a 10⁵×20 frame before vs after (target ≥5× speedup,
memory flat); determinism test (same frame → same digest, different order →
same digest where the contract requires order-independence); full medallion
test suite green.

### W5-T2 — Vectorize the gold split frame `[A]`

`dataflows/gold.py:70-99`: per-row Python dict appends (≈ n_rows × folds
iterations) + per-fold `sorted()`. Rebuild with concatenated index arrays +
`pl.repeat`/`pl.Literal` for fold/role columns. Also `gold.py:52-55,200`
extract the time column twice — once.

**Acceptance.** Equality test vs the old construction on a fixed seed
(temporarily keep both paths in the test); 10⁵-row timing improvement logged.

### W5-T3 — Embedding + feature-copy micro-optimizations `[A]`

- `data/embedding.py:44-49`: per-column `n_unique()` loop → single
  `df.select(pl.all().n_unique())`.
- `data/embedding.py:76`: `map_elements` Python UDF →
  `replace_strict(mapping, default=0)` (or join on unique values).
- `data/features.py:259-262`: `np.column_stack([X, interaction])` copies all of
  X per candidate pair (≤200×) → one preallocated scratch column.
- `data/leakage.py:92`: same per-column full-X copy.
- Land after W2 (same files).

**Acceptance.** Existing tests green; rough timings recorded in
`REPORT_LOG.md` (no formal benchmark required).

### W5-T4 — `iter8 medallion-run` CLI command (flagship visibility)

**Problem.** `session.medallion_run()` (`session.py:56-64`) and
`MedallionExecutionService` are reachable only from
`tests/unit/test_split_wiring.py:87`. The documented main path (`README.md:108`)
`iter8 run` goes `session.run → Trainer` directly — **no medallion artifacts**.
ADR-0003's contract is untestable from the CLI and invisible in demos.

**Fix.** New command (in `cli/medallion.py`, which already hosts the store/
docs-export helpers) wiring `session.medallion_run` with the usual options
(`--data`, `--target`, `--task`, `--experiment`, `--config`). Output: stage
progress + artifact paths + `iter8 medallion status`-style summary (reuse
`orchestration/service.py:243-258` status logic via the service, not a copy).
Update `README.md` quick start to show it (or present both paths honestly:
quick vs proven). **Do not** spawn any imperative training path (ADR-0001) —
wrap the existing service only.

**Acceptance.** End-to-end CLI test: `iter8 medallion-run` on the bundled demo
data produces bronze/silver/gold/platinum manifests + `_SUCCESS` markers;
`docs/medallion.md` + `README.md` updated; no new orchestration code outside
the service.

### W5-T5 — Boundary ADR(s)

Record the deviations so the rules and reality match (conventions require an
ADR before boundary changes — these are ratifications):

- **ADR-0007 (new): numpy/pandas boundary, restated.** numpy is allowed at the
  DataAdapter seam **and** where libraries demand arrays: sklearn splitters
  (`dataflows/gold.py`), statistical analysis (`analysis/*`), calibration/HPO/
  evaluator array plumbing (`engine/{calibration,hpo,evaluator}.py`),
  export templating (`services/export.py:27`). pandas: only the guarded
  `tabnet_model.py:15` pytorch-tabular shim. Update the boundary wording in
  `AGENTS.md` to match, or mark it superseded-by-ADR-0007 there.
- Optionally fold in the ThreadPool-in-node note (ADR-0001 footnote) — though
  W4 deletes that dead branch, so a sentence in 0007 suffices.

**Acceptance.** ADR accepted; `AGENTS.md` boundary paragraph updated;
`docs/decisions/README.md` index updated.

### W5-T6 (optional) — Storage index

`storage/local.py`: `exists()/_manifest_path()/open_artifact()/verify()` each
glob-scan the whole lake — O(products) per `store.begin()`
(`:47,106-149,177-217`). The catalog already exists — back these with
`LocalCatalogStore` or a sidecar index. Only if T1–T4 leave capacity; skip
freely at demo scale.

## Suggested subagent orchestration

1. `reviewer` (read-only): confirm W2 landed (gold/embedding ownership), then
   re-verify findings §5.
2. `worker` A: T1 (hashing) — the trickiest contract; do alone with the
   determinism tests.
3. `worker` B in parallel: T4 (CLI) — disjoint files from T1/T2.
4. `worker` C after both: T2 → T3 (dataflows + data perf).
5. Coordinator writes T5 (ADR — judgment call, not delegable).
6. `reviewer`: diff review; coordinator gate.

## Gotchas

- **Digest stability is a contract.** If T1 changes digest values, every
  existing workspace's resume breaks silently — bump graph version + document,
  or keep byte-compatible (preferred).
- `benchmarks/` and `demo/` may pin timing expectations; run the OpenML
  benchmark once after T1/T2 to confirm no regression (phase-2 learning #1:
  run without `OMP_NUM_THREADS` set to catch libgomp hangs).
- The CLI command must not bypass `MedallionExecutionService` — no new
  imperative path (ADR-0001).
- `docs/handoffs/` is agent-facing and **not** in the mkdocs nav; `docs/`
  topic pages you touch are — run `make docs` if you edit published pages
  (Quarto absent locally → commit `.qmd` changes with `--no-verify`; CI
  renders).

## Validation gate

Full gate, plus:

```bash
uv run pytest tests/unit/test_split_wiring.py tests/unit/test_medallion_contracts.py tests/unit/test_cli.py -q
ITER8ML_WORKSPACE=/tmp/w5check uv run iter8 medallion-run --demo  # after T4
```

## Definition of done

- [ ] T1–T5 done (T6 optional), determinism + CLI acceptance tests pass
- [ ] Before/after perf numbers in `REPORT_LOG.md`
- [ ] `README.md` + `docs/medallion.md` show the medallion path
- [ ] ADR-0007 accepted; `AGENTS.md` boundary updated
- [ ] Statuses inline; no commit unless explicitly asked
