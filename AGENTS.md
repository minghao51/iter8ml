# Project guidance

## Purpose and decision order

Build and prove iter8ml — a Polars-native, CPU-first tabular ML iteration
framework — as a public portfolio showcase per `docs/plan/portfolio-roadmap-20260805.md`.
The framework needs proof, polish, and story more than new features. When
guidance conflicts, follow this order:

1. The approved roadmap and the latest handoff digest in `docs/plan/`
   (`portfolio-roadmap-20260805.md`, `phase2-handoff-20260812.md`).
2. Accepted ADRs in `docs/decisions/`.
3. Topic design docs in `docs/` (pipeline-architecture, medallion, hpo, drift);
   superseded and deprecated material lives only in `docs/archive/`.

## Architectural boundaries

- Training executes as a single Hamilton DAG; no parallel imperative
  orchestration path (ADR-0001).
- Pipeline behavior is data: `PipelineSpec` + `@config.when` variants; node code
  carries no behavior branching (ADR-0002).
- Polars end-to-end; numpy appears only at the narrow `DataAdapter` model seam;
  no pandas in `src/iter8ml/`.
- Durable artifacts follow the medallion contract (atomic manifest + `_SUCCESS`,
  deep-checksum resume, split-overlap verification); event history alone is
  never a checkpoint (ADR-0003).
- Two trainer seams with different reliability contracts: best-effort event
  adapter, required state adapter — state publish failure fails the run (ADR-0005).
- CPU-first, GPU-ready. Cap OpenMP threads before any GBDT library loads
  libgomp (ADR-0004/0006).
- Safe deserialization via restricted unpickler allowlist; metric directionality
  and registry promotion centralized in `services/reporting.py`.
- Optional integrations (wandb, mlflow, llm, mcp) stay additive behind
  extras/config; avoid heavy infrastructure until an experiment demonstrates need.
- Prefer the smallest change that proves the point. Record any change to these
  boundaries as an ADR in `docs/decisions/` before implementing it.

## Engineering conventions

- Start from the documentation map (`docs/README.md`); new ADRs go in
  `docs/decisions/NNNN-slug.md` using the template (`Context → Decision →
  Consequences`).
- Python 3.11+, `uv`, `uv run <cmd>` (never bare `python`); typed Pydantic
  boundaries; config over code.
- Read before edit; prefer `edit` over `write`; minimal scope, no new
  abstractions. Present a plan for approval before modifying code; never commit
  unless explicitly asked. Check + follow any matching skill before a task.
- Every run stays reproducible: config, data hash, and metrics recorded in
  workspace state; benchmarks and case studies re-runnable from one command.
- Update `REPORT_LOG.md` with material findings and disclose substantive AI
  assistance.
- Concise output: bullets over paragraphs, `file:line` references, no
  speculation about unread code.

## Validation

Run before handing off a change:

```bash
uv sync --all-groups --all-extras
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy . --exclude 'build/'
make check-legacy-namespace
```

`--all-extras` is required: `optuna`/`xgboost`/`lightgbm` are extras
(`full`/`gbdt`/`train`), not dependency groups — bare `--all-groups` prunes them
and breaks the environment. Locally, stale `build/` breaks mypy (hence the
exclude).

For docs/notebook changes also run `make docs` (Quarto render + `mkdocs build`).

A change is complete when relevant tests pass, public behavior is documented,
and experiment outputs contain resolved configuration and reproducibility
metadata.
