# Repo & Module Layout

- **Last updated:** 2026-08-29 (generated from codebase audit)
- **Audience:** internal — agents/contributors; not published to the mkdocs site.

Entry points for deeper context: root `ARCHITECTURE.md` (module map + Hamilton DAG modes/seams) and `docs/README.md` (documentation map incl. archive policy). This file is the directory-level companion.

## Top-level layout

| Path | What it is |
|---|---|
| `src/iter8ml/` | The installed package (src-layout; wheel ships only this — `pyproject.toml:114-116`) |
| `tests/` | `unit/`, `integration/`, `e2e/` + shared `conftest.py` and hypothesis `strategies.py` |
| `docs/` | Published mkdocs content: topic guides, `decisions/` (ADRs), `plan/`, `notebooks/` (generated stubs), `releases/`, `archive/`, plus `stylesheets/`, `javascripts/`, `img/` |
| `notebooks/` | Quarto `.qmd` sources (`01_quick_start.qmd`, hero demos/case studies, `archive/`), `_quarto.yml`, `_freeze/` render state |
| `benchmarks/` | OpenML benchmark harness: `run_openml_benchmark.py`, `openml_benchmark.py`, `render_results.py`, `configs/`, `results/`, `sweeps/`, `archive/` |
| `demo/` | Standalone Gradio app for Hugging Face Spaces (`demo/app.py:1-3`, own `requirements.txt`) + bundled Telco Churn sample — separate from the package |
| `workspace/` | Default run workspace (`ITER8ML_WORKSPACE` env / `workspace` root — `src/iter8ml/workspace.py:9-10`): `experiments.jsonl`, `registry.json`, `leaderboard.md`, `current_state.md`, `lake/`, `artifacts/`, `control/`, `data/`, `exports/`, `site-data/` — generated state, not source |
| `.planning/` | Internal agent working state (not published): `OVERVIEW.md`, `STATE.md`, `STYLE.md`, `PHASE_1_2_HANDOFF.md`, `codebase/` (this file) |
| `scripts/` | Repo tooling: `generate_notebook_docs.py`, `check_legacy_namespace.py`, `deploy_hf.py` |
| `examples/` | Example configs and scripts (`credit_risk.py/.yaml/.toml`, `zenml_pipeline.py`, `configs_README.md`) |
| `build/`, `dist/`, `site/`, `catboost_info/`, `.venv/`, caches | Generated artifacts — never edit; `build/` causes a mypy duplicate-module failure if left in place (see `.planning/codebase/STACK.md`) |
| `.github/workflows/` | `ci.yml`, `docs.yml`, `benchmarks.yml` (details in `.planning/codebase/STACK.md`) |
| Root files | `AGENTS.md` (agent rules), `ARCHITECTURE.md`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `REPORT_LOG.md` (findings log), `mkdocs.yml`, `Makefile`, `pyproject.toml`, `uv.lock`, `Dockerfile`, `docker-compose.yml`, `.devcontainer/`, `.env.example` |

## Package layout: `src/iter8ml/*`

Top-level files:

| File | Responsibility |
|---|---|
| `src/iter8ml/__init__.py` | Thin eager core: version, `ExperimentConfig`, `HardwareProfile`, constants, `load_data`, exceptions, `Workspace` — no heavy ML imports (`:12-25`); engine symbols lazy-loaded via `_LAZY` map (`:30+`) |
| `src/iter8ml/config.py` | Experiment/Pipeline config + hardware profiles; caps OpenMP threads before GBDT loads (ADR-0004/0006; `_OMP_THREAD_CAP = 8`, `src/iter8ml/config.py:27`) |
| `src/iter8ml/constants.py` | Type-safe enums: `TaskType`, `CVStrategy`, `EmbeddingMethod`, `TrackerType` (`:6-33`) |
| `src/iter8ml/exceptions.py` | Exception hierarchy rooted at `TabularBlueprintError` (alias `Iter8MLError`, `:48`) + `track_errors` decorator (`:51`) |
| `src/iter8ml/session.py` | `ExperimentSession` — primary programmatic API wiring Trainer, registry, reporting, export, medallion service (`src/iter8ml/session.py:1-27`) |
| `src/iter8ml/workspace.py` | `Workspace` dataclass resolving run paths (env-overridable root, `:9-20`) |
| `src/iter8ml/py.typed` | PEP 561 marker shipped in the wheel |

Subpackages:

| Subpackage | Responsibility | Key files |
|---|---|---|
| `analysis/` | Drift detection & explainability primitives | `drift.py` (KS+chi2), `psi.py` (PSI), `domain_classifier.py`, `_protocol.py` |
| `cli/` | Typer app + subcommands; `__init__.py` imports submodules for side-effect registration (`src/iter8ml/cli/__init__.py:1-8`) | `main.py` (`iter8 init`, `iter8 hardware`), `run.py`, `optimize.py`, `analyze.py`, `export.py`, `medallion.py`, `mcp.py` |
| `data/` | Data loading & preparation | `loader.py` (csv/parquet/sqlite + `load_data`), `adapter.py` (Polars→NumPy seam), `features.py` (AFE), `leakage.py`, `quality.py` (Cleanlab audits), `embedding.py` |
| `dataflows/` | Medallion data products: `materialize_bronze/silver/gold/platinum` (`src/iter8ml/dataflows/__init__.py:3-9`) | `bronze.py`, `silver.py`, `gold.py`, `platinum_train.py` |
| `datasets/` | Bundled demo datasets (`telco_churn.parquet`) for zero-download `iter8 init --demo` (`src/iter8ml/datasets/__init__.py:1-14`) | `__init__.py`, parquet files |
| `domain/` | Versioned shared contracts: events (`EventEnvelope`, JSONL sink), hashing (digests), ids, manifests (RunPlan/Split/Product/Run manifests) (`:3-16`) | `events.py`, `hashing.py`, `ids.py`, `manifests.py` |
| `engine/` | Training engine: `Trainer`, `Evaluator`, `JSONLTracker`, HPO (+importance, warmstart), calibration, state observer, trainer seams factory (`src/iter8ml/engine/__init__.py:3-7`) | `trainer.py`, `trainer_factory.py` (adapter seams, ADR-0005), `evaluator.py`, `hpo.py`, `hpo_importance.py`, `hpo_warmstart.py`, `calibration.py`, `tracker.py`, `state_observer.py` |
| `engine/models/` | Model wrappers + selection; registered via `iter8ml.models` entry points (`pyproject.toml:80-88`) | `base.py` (`AbstractModel`), `factory.py`, `selector.py` (`ModelSelector`), `gbdt_base.py`, `catboost_model.py`, `lightgbm_model.py`, `xgboost_model.py`, `tabpfn_model.py`, `ft_transformer.py`, `tabnet_model.py`, `baselines.py`, `model_configs.py`, `sparse_embedder.py` |
| `engine/pipelines/` | Hamilton DAG layer: `PipelineExecutor` + mode enum (`TRAINING/DRIFT/EXPORT/HPO/INFERENCE` — `ARCHITECTURE.md`), `describe_pipeline`/`visualize_pipeline` helpers (`src/iter8ml/engine/pipelines/__init__.py:12-24`) | `executor.py`, `preprocessing.py` (re-export shim over nodes), `hooks/tracking_hook.py`, `nodes/{prep,features,train,drift_detection}.py` (behavior switched by `@config.when` data, ADR-0002), `nodes/_hamilton_compat.py` |
| `orchestration/` | Local medallion run lifecycle: `MedallionExecutionService` + `ExecutionResult`, `LocalOrchestrator`, scheduler-neutral `protocol.py` (`src/iter8ml/orchestration/__init__.py:3-6`) | `service.py`, `local.py`, `protocol.py` |
| `runtime/` | Config→plan compilation: `compile_run_plan(ExperimentConfig) -> RunPlan` (`:13`) | `plan.py` |
| `services/` | Cross-cutting services: registry/promotion, reporting (metric directionality + promotion — `AGENTS.md`), export packaging, docs export, retention GC, LLM agent, MCP server (`src/iter8ml/services/__init__.py:3-19`) | `registry.py`, `reporting.py`, `export.py`, `docs_export.py`, `retention.py`, `llm.py`, `mcp.py` |
| `storage/` | Local persistence: `LocalArtifactStore`/`ProductWriter` (atomic writes) and `LocalCatalogStore` (rebuildable SQLite projection of manifests — `src/iter8ml/storage/catalog.py:1-4`) | `local.py`, `catalog.py` |
| `utils/` | I/O + safe pickle (`RestrictedUnpickler`, HMAC integrity — `src/iter8ml/utils/io.py:1-9,85`) and parallel-job capping | `io.py`, `parallel.py` |
| `verification/` | Verification gates for medallion products: `validate_split` on the split frame (`src/iter8ml/verification/split_validation.py:1-10`) | `split_validation.py` |

## Tests layout

- `tests/conftest.py` — session fixtures (`classification_data`, `regression_data`, `tmp_workspace`) built with sklearn `make_*` into Polars frames; auto-marks `unit/` items (`tests/conftest.py:29-31`).
- `tests/strategies.py` — shared hypothesis strategies (Polars dataframes, JSON, pickles — `tests/strategies.py:1-20`).
- `tests/unit/` — 69 fast isolated test files, roughly one per subsystem (`test_adapter.py`, `test_drift.py`, `test_property_*.py` for hypothesis, `test_metamorphic_*.py`, `test_contract_api.py`, `test_cli.py`, …).
- `tests/integration/` — cross-component flows with its own `conftest.py`: `test_full_pipeline.py`, `test_dag_execution.py`, `test_gdbt_models.py`, `test_hpo.py`, `test_model_selection.py`, `test_registry_and_drift.py`, `test_export_package.py`, `test_guardrails_large_input.py`.
- `tests/e2e/` — full-workflow tests: `test_smoke.py` (end-to-end pipeline with minimal data/models).
- Markers (`slow`, `integration`, `unit`, `e2e`, `serial`, `network`, `smoke`, `property`, `metamorphic`, `contract`, `differential`) are strict-registered in `pyproject.toml:168-180`; CI runs the three directories in that order (`.github/workflows/ci.yml`).

## Generated / published docs pipeline

- Sources: `notebooks/*.qmd` (Quarto; config `notebooks/_quarto.yml`, freeze in `notebooks/_freeze/`).
- `make notebooks` / `make notebooks-staged` render via `uv run quarto render` (`Makefile:3-11`); the staged variant only re-renders git-staged `.qmd` files and re-adds freeze + HTML outputs.
- `scripts/generate_notebook_docs.py` parses `.qmd` frontmatter and writes mkdocs stub pages + HTML under `docs/notebooks/` (`scripts/generate_notebook_docs.py:1-20`).
- `make docs` chains render → stubs → `mkdocs build` (`Makefile:13-15`); CI `docs.yml` does the same and deploys `site/` to GitHub Pages.
- `docs/notebooks/` is therefore **generated output** — edit the `.qmd` sources, not the stubs (legacy `docs/notebooks/archive/` predates the archive policy, per `docs/README.md`).
- Published nav lives in `mkdocs.yml:34-56` (guides, ADRs, notebook stubs); superseded material belongs in `docs/archive/` (`docs/README.md` archive-policy section).

## Related reading

- `ARCHITECTURE.md` (repo root) — module narrative, pipeline mode table, trainer seams, export/guardrail notes.
- `docs/README.md` — documentation map incl. where planning and internal state live.
- `.planning/codebase/STACK.md` — dependency/tooling counterpart to this file.
