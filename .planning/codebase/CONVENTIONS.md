# Conventions

- **Last updated:** 2026-08-29 (generated from codebase audit)
- **Audience:** internal — agents/contributors; not published to the mkdocs site.

Source of truth is `CONTRIBUTING.md`, `.planning/STYLE.md`, `pyproject.toml`
(`[tool.ruff]` / `[tool.mypy]`), and the code itself. This file is a working
summary; where it disagrees with `STYLE.md`, `STYLE.md` wins and this should be
updated.

## Core principles (architecture-aligned)

- **Functional over class-heavy** — prefer pure functions; models are plain
  classes conforming to the `AbstractModel` `typing.Protocol` (no inheritance,
  see `STYLE.md` "Model Wrappers").
- **Explicit over magic** — no hidden state or silent fallbacks.
- **Polars as single source of truth** — no Pandas in `src/iter8ml/`.
- **Config is data, code is declarative** — pipeline behavior is a `n` config
  object, not `if/else` (see below).
- **Observability first** — JSONL event tracker, not a separate database.

## Polars-first data rule

- End-to-end data is Polars (`pl.DataFrame`); numpy `(X, y)` appears *only* at
  the narrow `DataAdapter` model seam — `src/iter8ml/data/adapter.py:7`
  (`class DataAdapter`, `.transform` returns `np.ndarray`).
- No pandas import anywhere under `src/iter8ml/`. Convert at the seam and
  nowhere else. (Cross-linked in `docs/decisions/README.md` standing convention.)
- `load_data()` dispatches on suffix → `load_csv` / `load_parquet`
  (`src/iter8ml/data/loader.py`); config loading dispatches on `.yaml/.toml/.json/.py`
  via `ExperimentConfig.from_file()`.

## Typed Pydantic boundaries

- All experiment/pipeline config lives in Pydantic `BaseModel`s (`config.py`,
  `model_configs.py`). Typed fields + `Field(default_factory=...)` for mutables.
- Nested configs compose (`HPOConfig`, `QualityConfig`, `EmbeddingConfig`).
- Validation via `@field_validator` / `@model_validator(mode="before"|"after")`;
  enums as field types for constrained choices (`TaskType`, `CVStrategy`, …);
  `@field_serializer` for enums.
- `disallow_untyped_defs = true` (mypy, `pyproject.toml`) — type your signatures.
- Result/data models are also `BaseModel` (`LeaderboardEntry`,
  `PromotionResult`, `DriftReport`).

## Config-over-code (data-driven pipelines)

- The `n` object *is* the `PipelineSpec`: a step-configuration enum/value that
  drives which Hamilton variants run (`src/iter8ml/config.py:class n`).
- Nodes stay **branchless**: variants are activated by `@config.when`-style
  decorators resolved via `_resolve_hamilton_config()` reading `n.step_params(...)`.
  Internally this is `@_hamilton_n(...)` / `@_hamilton_n_not(...)` — e.g.
  `src/iter8ml/engine/pipelines/nodes/prep.py` (`run_quality_audit`,
  `run_leakage_audit`, `target_transform="none"`).
- Adding a variant is **additive** (new decorated node); never edit branch logic
  into an existing node. See `docs/pipeline-architecture.md` §"n — Step
  Configuration" and `docs/decisions/0002-pipeline-spec-config-when.md`.

## Error handling

- Hierarchy: `TabularBlueprintError` → `DataLoadError` / `ModelFitError` /
  `RegistryError` (`src/iter8ml/exceptions.py`).
- `@track_errors()` decorator (`exceptions.py:51`) catches bare exceptions, logs
  to tracker, re-raises as the typed error; base exceptions carry a `context: dict`.
- **Two trainer seams with different reliability contracts** (ADR-0005):
  the event adapter is best-effort; the state adapter is **required** — if it
  fails the `Trainer` raises `TrainerStatePublishError`
  (`src/iter8ml/engine/trainer.py:140`). State loss is never swallowed.

## Logging / events

- Telemetry and state flow through the `Tracker` protocol
  (`src/iter8ml/engine/tracker.py`); default `JSONLTracker` appends to
  `workspace/experiments.jsonl`. Emit structured events, not free-text prints.
- Champion metadata in `workspace/registry.json` behind `filelock.FileLock`
  (`services/registry.py`).

## Lazy GBDT import + OpenMP guard ordering

- GBDT libs load **lazily** on first `get_model_class()`; on hybrid (P+E) CPUs
  under Linux/WSL2 their libgomp **deadlocks across all cores** (exit `124`).
- **Rule:** call `HardwareProfile.configure_omp_threads()` (caps at 8 on Linux)
  **before** any `get_model_class()` call. Entrypoints do this early —
  `notebooks/case_study_german_credit.qmd:23` and `demo/app.py:38` both call it
  at top. When verifying, run **without** `OMP_NUM_THREADS` set to confirm no hang.

## Serialization safety

- Model/artifact unpickling goes through `RestrictedUnpickler`
  (`src/iter8ml/utils/io.py:85`) — a restricted allowlist. Do not replace with a
  bare `pickle.load`.

## Naming / layout

- Package dirs, modules, functions `snake_case`; classes `PascalCase`;
  services `…Service`; protocols `Abstract…`; exceptions `…Error`; enums
  `PascalCase` with `UPPER_SNAKE_CASE` members; tests `test_<module>.py`.
- Plugin discovery via `[project.entry-points."iter8ml.models"]` + lazy
  `importlib` import cached in `_MODEL_CLASS_CACHE`
  (`src/iter8ml/engine/models/factory.py`).

## Commit / pre-commit

- Lint/format via `ruff`; types via `mypy .` (excluding `tests/`, `benchmarks/`,
  `notebooks/`, `demo/`, `workspace/`).
- Pre-commit runs `ruf-check --fix`, `ruff-format`, `pip-audit`, `mypy`,
  **and** `quarto-render` (renders staged `.qmd`). Quarto is **not** installed
  locally, so the render hook fails on commit → **commit `.qmd` with
  `--no-verify`** (CI renders in `.github/workflows/docs.yml`).
- `.gitignore` has a global `*.parquet` (for `workspace/`); bundled parquets need
  explicit negations (`!src/iter8ml/datasets/*.parquet`, `!demo/*.parquet`) or
  they silently won't ship.
- Do not `git add .env` (secrets; `.env.example` is the tracked template).
- `build/` is a stale gitignored artifact that makes `mypy .` report a spurious
  "Duplicate module named iter8ml" — run `mypy . --exclude 'build/'` or delete it.
