# Open Concerns, Risks & Debt Register

- **Last updated:** 2026-08-29 (generated from codebase audit)
- **Audience:** internal — agents/contributors; not published to the mkdocs site.

Sources: `docs/plan/deferred-research.md` (consolidated deferred work),
`docs/plan/phase2-handoff-20260812.md` §3 (critical learnings),
`docs/plan/portfolio-roadmap-20260805.md` (risks & constraints), accepted ADRs
in `docs/decisions/`, plus a fresh scan on 2026-08-29
(`rg -n 'TODO|FIXME|XXX|HACK' src/ tests/` → **zero hits**, also clean in
active `demo/`/`scripts/`/`benchmarks/`; `# type: ignore` census → ~60 sites
in `src/`, clusters listed under Debt). Tiers: **Active risk / Known gotcha /
Debt / Deferred scope**.

## Active risk

| # | Concern | Impact | Where it bites | Mitigation / pointer |
|---|---|---|---|---|
| R1 | RAM is the binding constraint (15 GB total, ~10 GB free, no GPU) | Benchmark/training OOMs; roadmap calls it High likelihood | `benchmarks/` (covertype 581k rows), large uploads | CPU-tuned config `benchmarks/configs/cpu_benchmark.yaml`; sequential runs; row caps; demo cap 20k rows (`demo/app.py:43`). Roadmap *Constraints* + *Risks* |
| R2 | **Base-dep drift vs the additive-only rule**: `litellm>=1.83.10` and `mlflow>=3.11.1` are hard base dependencies (`pyproject.toml:31-32`); `gitpython`, `cryptography`, `python-multipart` are also base deps with **no direct import site** in `src/`/`scripts/`/`demo/`/`benchmarks/` (verified by rg; possibly transitive pins for litellm/mlflow — intent unverified) | Install weight + supply-chain surface for every user; core coupled to optional services; contradicts AGENTS.md "integrations behind extras" | Fresh `uv sync` / PyPI install; `pip-audit` surface (`.github/workflows/ci.yml` security job) | Behavior is still off by default (`llm_enabled=False`, `config.py:218`). If intentional, record an ADR; otherwise move behind extras. Cross-ref [INTEGRATIONS.md](INTEGRATIONS.md) |
| R3 | PyPI release exposes a packaging bug (Medium) | Public credibility hit | `uv publish` / `uvx iter8` flows | Smoke-test `uvx --from . iter8 run` before publish; cut `0.1.0rc1` first (roadmap *Risks*) |
| R4 | Live demo attacked/abused (Low-Med) | Free-tier quota exhaustion, bad UX | `demo/app.py` (if/when deployed) | Upload size caps, timeout limits, no persisted user data, throwaway workspace per request (`demo/app.py:86-95`); roadmap *Risks* |
| R5 | Deep-model omission (FT-Transformer/TabNet/TabPFN) read as a feature gap (Low) | Reviewer perception | Portfolio story | Frame explicitly as "CPU env; GPU-ready" per [ADR-0006](../../docs/decisions/0006-cpu-first-gpu-ready.md); hardware-aware routing is the talking point |
| R6 | Part-time solo capacity slips the schedule (Medium) | Roadmap dates | All of Phase 3 | Only P0 items blocking; P1/P2 slide; defer list absorbs overrun (roadmap *Risks*) |

## Known gotcha (each cost real debugging time — handoff §3)

| # | Gotcha | Where it bites | Mitigation / pointer |
|---|---|---|---|
| G1 | **OpenMP/libgomp deadlock** on hybrid (P+E-core) CPUs under Linux/WSL2: lightgbm/xgboost hang across all cores; silent exit 124 | Any GBDT entrypoint that forgets the cap | Bounded by [ADR-0004](../../docs/decisions/0004-hardware-aware-model-routing.md) + [ADR-0006](../../docs/decisions/0006-cpu-first-gpu-ready.md). `HardwareProfile.configure_omp_threads()` caps at 8 on Linux (`src/iter8ml/config.py:460-479`); called in `Trainer.__init__` (`src/iter8ml/engine/trainer.py:36`) and at import in `demo/app.py:38`. GBDT libs load lazily at first `get_model_class()` — new entrypoints must configure first; verify by running without `OMP_NUM_THREADS` set |
| G2 | **`DataAdapter.transform()` returns object-dtype X** with unencoded categoricals; direct HPO/SHAP/Evaluator paths do not encode | Direct `setup_hpo_components`/SHAP usage (skipped by full `session.run`) | Ordinal-encode object columns; `LabelEncoder` the y for string classification targets; mirror `benchmarks/openml_benchmark.py` and `demo/app.py:53-70`. Seam: `src/iter8ml/data/adapter.py:10-15` |
| G3 | **HPO with empty search space is silent**: `_parse_trial_params` returns `{}` when `search_space` falsy → every trial is defaults → `best_params == {}` | Direct `optimize_model(...)` calls omitting `search_space` | Canonical call: `src/iter8ml/cli/optimize.py:28,44` (uses `setup_hpo_components`, which loads `ModelConfigs().<model>.hpo_search_space()`, `src/iter8ml/engine/hpo.py:189`). Silent-return site: `src/iter8ml/engine/hpo.py:115-116`. Candidate micro-fix: warn or raise on empty space |
| G4 | **Pre-commit `quarto-render` hook blocks `.qmd` commits** when quarto isn't installed locally (hook runs `make notebooks-staged`) | Any commit with a staged notebook | Commit with `--no-verify`; CI renders on push to `main` (`.github/workflows/docs.yml:31`). Hook: `.pre-commit-config.yaml:40-44` |
| G5 | **Global `*.parquet` gitignore** silently swallows new bundled datasets | Shipping package data | Negations exist for `src/iter8ml/datasets/*.parquet` and `demo/*.parquet` (`.gitignore:55,58,59`); add a negation for any new parquet and verify with `uv build` + `python -m zipfile -l dist/*.whl` |
| G6 | **Stale `build/` directory causes spurious mypy "Duplicate module named iter8ml"** | Local `uv run mypy .` (CI is clean) | `build/` exists on this machine (verified 2026-08-29); run `uv run mypy . --exclude 'build/'` or delete `build/` (handoff §3.6) |
| G7 | **`.env` holds real secrets** (TABPFN_TOKEN, HF_TOKEN, …) | Accidental commit | Gitignored at `.gitignore:18`; `.env.example` is the tracked template; never `git add .env` (handoff §3.8) |
| G8 | **`max_workers > 1` + GBDT thread contention** | Parallel training config | `strict_thread_safety=True` disables cross-model parallelism when internally multi-threaded models are selected (`src/iter8ml/config.py:196-210`; enforcement in `src/iter8ml/engine/pipelines/nodes/train.py:357-367`) |

## Debt

| # | Item | Impact | Where | Pointer |
|---|---|---|---|---|
| D1 | `ExperimentConfig.tracker` (`TrackerType`, `src/iter8ml/config.py:193`; enum `src/iter8ml/constants.py:28`) is **declared but never wired** — nothing maps `TrackerType.WANDB/MLFLOW` to a tracker instance; trainer falls back to `JSONLTracker` (`src/iter8ml/engine/trainer.py:48-51`) | Misleading config surface: `tracker="wandb"` silently does nothing; W&B/MLflow are manual-injection only | `session.run` paths | Either wire a tracker factory or deprecate the field; see [INTEGRATIONS.md](INTEGRATIONS.md) |
| D2 | mlflow pinned twice with different floors: `mlflow>=3.11.1` base (`pyproject.toml:32`) vs `mlflow>=2.13` in `train` extra (`:58`) | Confusing resolution; extra pin is dead weight | Packaging | Align pins (part of R2) |
| D3 | Legacy naming residue: mkdocs `site_name: Tabular Blueprint` (`mkdocs.yml:1`); legacy env alias `TABBLUEPRINT_LLM_MODEL` (`src/iter8ml/services/llm.py:24`, `src/iter8ml/engine/state_observer.py:36-38`); fully-commented legacy example `examples/zenml_pipeline.py` | Brand inconsistency on the published site; stale example confuses agents | Public site, env handling | `make check-legacy-namespace` blocks only `tabular_blueprint`/`tabular-blueprint` tokens in scanned globs (`scripts/check_legacy_namespace.py:8-22`) — spaced/cased variants and `examples/` are not covered. Rename site or extend tokens/globs |
| D4 | Stale-doc pointers: `docs/plan/phase2-handoff-20260812.md` §1 and `docs/plan/deferred-research.md` §5 describe four plan-doc deletions (`code-simplify-20260515.md`, `package-refactor-20260513.md`, `pipeline-spec-20260515.md`, `docs/technical_roadmap.md`) as *pending commits* — **no longer true** (verified 2026-08-29: `git ls-files docs/plan/` tracks only the 3 current docs; `git status` shows no deletions). The real uncommitted state is the 2026-08-29 restructure: modified `AGENTS.md`/`CLAUDE.md`/`README.md`/`mkdocs.yml`/`docs/*` + untracked `docs/decisions/`, `docs/README.md`, `REPORT_LOG.md`, `.planning/codebase/`, `src/iter8ml/cli/mcp.py`, `src/iter8ml/analysis/_protocol.py`, `src/iter8ml/verification/split_validation.py` | Handoff/deferred docs mislead the next agent; roadmap item 1.5 ("queued refactors") loses its source docs once those land | `git status`; `docs/plan/` | Update the two stale §notes when next touched; commit or explicitly defer the restructure work. Use `ARCHITECTURE.md` as source material for the old roadmap content |
| D5 | `# type: ignore` clusters: ~60 sites in `src/` — `engine/models/ft_transformer.py` (10), `data/features.py` (6), `engine/calibration.py` (4), `config.py` (4), `orchestration/service.py` (3), `engine/models/sparse_embedder.py` (3), `engine/models/catboost_model.py` (3), rest scattered | Masks typing regressions at the numpy/model seam | The narrow `DataAdapter` seam is *allowed* to be loose per AGENTS.md, but the spread goes beyond it | Trend-watch per audit; refactor high-count files toward typed returns when touched |
| D6 | Roadmap 1.6 follow-ups partially landed: HMAC integrity on `safe_dump` done (`src/iter8ml/utils/io.py:96-132`, with honest "not tamper-resistant" docstring at `:3-7`); SQL sanitizer hardened (SELECT-only + keyword blocklist, `src/iter8ml/data/loader.py:66-95`) — keyword-token blocking remains bypassable in principle (e.g. inside strings) | Low; documented | `data/loader.py` | Accept as-is for local-file SQLite path, or switch to `sqlite3` authorizer |
| D7 | `mike>=2.0` in the `docs` extra (`pyproject.toml:68`) but unused by `.github/workflows/docs.yml` (uses `actions/deploy-pages`) | Unused dep; unclear versioning story for docs | Docs pipeline | Unverified whether mike has a manual role — confirm and either use or drop |
| D8 | Local stale artifacts: `build/` (see G6), `src/iter8ml.egg-info/`, `catboost_info/` on disk | Noise, mypy duplicate (G6) | Working tree | Gitignored; safe to delete locally |

## Deferred scope (deliberate, not debt)

Single home for deferred items: **`docs/plan/deferred-research.md`** — summary:

- **Medallion completion** (priority A, demoted per B>C>A): model-per-fold
  Platinum products, OOF artifacts, true DuckDB catalog over Parquet views,
  artifact-contract migration tooling — from
  [ADR-0003](../../docs/decisions/0003-medallion-artifact-contract.md) *Honest
  status* and roadmap 3.4.
- **Live-demo hosting**: HF Spaces blocked on free tier (PRO needed; 512 MB
  OOM). Gradio app + `scripts/deploy_hf.py` kept for Phase 3;
  `run_analysis()` must stay callable (it is the Phase-3 agent seam).
- **Phase 3 (reference)**: 3.1 agent showcase via MCP (P0 hero),
  3.2 external visibility, 3.3 repo surface polish, 3.5 ONNX/TorchScript
  export (`services/export.py`).
- **Explicitly out of scope** (solo/part-time capacity): uncertainty
  quantification, Optuna Dashboard, remote data loaders (S3/GCS), AFE pruning
  via RFE/null-importation, full DuckDB catalog.

## Fresh-scan appendix (2026-08-29)

- `rg -n 'TODO|FIXME|XXX|HACK' src/ tests/` → no matches (also none in active
  `demo/`, `scripts/`, `benchmarks/`; case-insensitive scan). Hits elsewhere only
  in generated `site/` assets and archived rendered notebooks
  (`docs/notebooks/html/`) — not maintained source. Codebase is marker-clean;
  this register is the TODO list.
- `# type: ignore` census → ~60 in `src/`, 1 in `tests/`; clusters in D5.
- Working-tree check: the handoff's "four uncommitted deletions" are resolved
  (see D4); current uncommitted work is the 2026-08-29 docs/ADR restructure
  (modified + untracked files, listed in D4).
