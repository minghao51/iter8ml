# Stack & Dependencies Reference

- **Last updated:** 2026-08-29 (generated from codebase audit)
- **Audience:** internal — agents/contributors; not published to the mkdocs site.

Source of truth: `pyproject.toml` + `uv.lock`. Package manager is `uv` (`uv run <cmd>`, never bare `python` — `AGENTS.md`). CI always installs with `uv sync --frozen`, so `uv.lock` is authoritative for exact versions.

## Python version floor

- `requires-python = ">=3.11"` — `pyproject.toml:9`; classifiers list 3.11 and 3.12 (`pyproject.toml:16-17`).
- CI test matrix actually runs 3.11, 3.12, and 3.13 (`.github/workflows/ci.yml:39`).
- Tool floors match: ruff `target-version = "py311"` (`pyproject.toml:120`), mypy `python_version = "3.11"` (`pyproject.toml:124`).

## Build & packaging

- Backend: `hatchling>=1.25` + `hatch-vcs>=0.4` (`pyproject.toml:1-3`); version is dynamic from git tags (`[tool.hatch.version] source = "vcs"`, `pyproject.toml:111-112`).
- Wheel ships `src/iter8ml` including the `py.typed` marker (`pyproject.toml:114-116`).
- Console script: `iter8 = "iter8ml.cli:app"` (`pyproject.toml:90-91`).
- Model plugins register via the `iter8ml.models` entry point (8 models: catboost, lightgbm, xgboost, tabpfn, ft_transformer, tabnet, naive_baseline, linear_baseline — `pyproject.toml:80-88`).

## Runtime dependencies (`[project] dependencies`, `pyproject.toml:21-36`)

| Dependency | Why it's there |
|---|---|
| `polars>=1.0` | The end-to-end dataframe layer; pandas is banned in `src/iter8ml/` (Polars-first per `AGENTS.md` / ADRs) |
| `pydantic>=2.0` | Typed config/validation boundaries — `ExperimentConfig`, `PipelineSpec`, manifests (`src/iter8ml/config.py`, `src/iter8ml/domain/manifests.py`) |
| `scikit-learn>=1.4` | Metrics, baselines, calibration, drift stats, leakage/quality probes (imported in `src/iter8ml/engine/models/baselines.py`, `src/iter8ml/engine/evaluator.py`, `src/iter8ml/engine/calibration.py`, `src/iter8ml/data/leakage.py`, `src/iter8ml/data/quality.py`, `src/iter8ml/analysis/domain_classifier.py`) |
| `numpy>=1.26` | The single Polars→NumPy model seam via `DataAdapter` (`src/iter8ml/data/adapter.py:1-8`) |
| `typer>=0.12` | CLI framework (`src/iter8ml/cli/main.py:10`) |
| `rich>=13.0` | Terminal rendering in CLI subcommands (`src/iter8ml/cli/*.py`) |
| `filelock>=3.12` | Cross-process locking of `workspace/registry.json` champion updates (`src/iter8ml/services/registry.py:13`) |
| `pyyaml>=6.0` | YAML experiment configs and notebook frontmatter parsing (`src/iter8ml/config.py:12`; `scripts/generate_notebook_docs.py`) |
| `psutil>=5.9` | Hardware-profile detection (RAM/CPU) for CPU-first model routing (`src/iter8ml/config.py:11,453-454`) |
| `litellm>=1.83.10` | LLM provider access for the TabularAgent explanation service (`src/iter8ml/services/llm.py:79`, lazy import) |
| `mlflow>=3.11.1` | Optional tracking backend (`TrackerType.MLFLOW`, `src/iter8ml/constants.py:33`; lazy import at `src/iter8ml/engine/tracker.py:136`) |
| `cryptography>=46.0.7` | **Unverified purpose:** declared at `pyproject.toml:33` but no import of `cryptography` found anywhere in `src/` — possibly a security pin or vestigial |
| `gitpython>=3.1.50` | **Unverified purpose:** declared at `pyproject.toml:34`, no `import git` found in `src/` |
| `python-multipart>=0.0.27` | **Unverified purpose:** declared at `pyproject.toml:35`, no direct import found; typically a transitive requirement of mlflow/fastapi-style servers |

## Optional-dependency extras (`pyproject.toml:38-73`)

| Extra | Contents | What it unlocks |
|---|---|---|
| `gbdt` | catboost, lightgbm, xgboost, optuna, sf-hamilton (`:39-45`) | GBDT models + Optuna HPO + the Hamilton DAG driver. Note sf-hamilton lives in an extra, not core — `HamiltonUnavailableError` raises without it (`src/iter8ml/exceptions.py:36`) |
| `train` | everything in `gbdt` **plus** torch, accelerate, transformers, tabpfn, pytorch-tabular, wandb, mlflow, mcp, litellm, shap, cleanlab (`:46-63`) | Full training surface: neural models (FT-Transformer, TabNet, TabPFN), explainability (shap), label-noise audits (cleanlab), wandb/mlflow tracking, MCP server, LLM agent |
| `docs` | matplotlib, mkdocs-material, mkdocstrings[python], mike, pymdown-extensions (`:64-70`) | Site build: `make docs` / mkdocs; matplotlib for notebook plots. **Note:** `mike` (versioned docs deploys) is declared but CI deploys via `actions/deploy-pages@v4` instead (`.github/workflows/docs.yml`); no mike usage found in workflows |
| `full` | `iter8ml[train,docs]` (`:71-73`) | Everything; this is what CI installs (`--extra full`, `.github/workflows/ci.yml`) |

## Dev dependency-group (`[dependency-groups] dev`, `pyproject.toml:93-109`)

- Test: `pytest>=9.0.3`, `pytest-cov>=5.0`, `hypothesis>=6.0` (property-based tests via `tests/strategies.py`).
- Lint/type: `ruff>=0.4`, `mypy>=1.10`, `pre-commit>=3.6`, `pip-audit>=2.7`.
- Notebook tooling: `marimo>=0.23.4`, `jupyter`, `jupyter-server`, `nbformat`, `nbclient`, `mako`, `urllib3>=2.7.0` (the urllib3/mako floors look like transitive security/workaround pins — purpose beyond version floor **unverified**).
- Dev group is uv-only (PEP 735 `[dependency-groups]`), not published as an extra.

## uv.lock summary

- `uv.lock` is ~1.1 MB with **316 locked packages** (counted via `[[package]]` entries).
- Representative locked versions: polars 1.39.3, sf-hamilton 1.89.0, catboost 1.2.10, torch 2.11.0.
- CI uses `--frozen` everywhere, so PRs must keep the lock in sync (`uv sync --all-groups` then commit the updated lock per `AGENTS.md` validation steps).

## Tooling config

### Ruff (`pyproject.toml:118-159`)

- `line-length = 100` (`:119`), `target-version = "py311"` (`:120`), `extend-exclude = ["benchmarks/archive"]` (`:121`).
- Lint rules: `E, F, I, UP, B, SIM, C4, PT, RUF`, `fixable = ["ALL"]` (`:152-154`).
- Per-file ignores: `B008` (Typer default args) in `src/iter8ml/cli/**/*.py`, `F401` in `src/iter8ml/cli.py`, notebook cells relax `E402/I001/B018/E501/RUF001/F841` (`:156-159`).
- Commands: `uv run ruff check .` and `uv run ruff format --check .` (`AGENTS.md`).

### Mypy (`pyproject.toml:123-151`)

- `python_version = "3.11"`, `disallow_untyped_defs = true` (`:124-125`).
- Excludes: `tests/`, `benchmarks/`, `notebooks/`, `demo/`, `workspace/` (`:126`) — **`build/` is NOT excluded, see gotcha below**.
- Four `[[tool.mypy.overrides]]` blocks with `ignore_missing_imports = true` for untyped third-parties (`:128-151`): GBDT/Hamilton stack (catboost, lightgbm, xgboost, optuna, polars, sf_hamilton, hamilton); optional integrations (cleanlab, shap, wandb, mlflow, litellm, mcp); deep-learning stack (pytorch_tabular, tabpfn, torch, transformers); scientific/infra (matplotlib, sklearn, scipy, psutil, pandas, yaml).
- **Known gotcha — `build/` duplicate module:** after a local `pip`-style/bdist build, `build/lib/iter8ml/` contains a stale copy of the package *with* `__init__.py` files. Running `uv run mypy .` then fails with `Duplicate module named "iter8ml" (also at "./build/lib/iter8ml/__init__.py")` and "errors prevented further checking". Reproduced on 2026-08-29 against the current tree. Fix: delete the untracked `build/` directory (it is a build artifact); the mypy `exclude` in `pyproject.toml:126` does not cover it.

### Pytest (`[tool.pytest.ini_options]`, `pyproject.toml:161-180`)

- `testpaths = ["tests"]`, `pythonpath = ["src"]`, `--strict-markers -ra --durations=10 --import-mode=importlib` (`:162-164`).
- `filterwarnings` ignores `scipy.optimize` DeprecationWarnings (`:165-167`).
- Registered markers (`:168-180`): `slow`, `integration`, `unit`, `e2e`, `serial`, `network`, `smoke`, `property`, `metamorphic`, `contract`, `differential`.
- `tests/conftest.py:29-31` auto-applies the `unit` marker to `tests/unit/` items; hypothesis strategies live in `tests/strategies.py`.

### pre-commit (`.pre-commit-config.yaml`)

- `pre-commit-hooks v5.0.0`: trailing-whitespace, end-of-file-fixer (both excluding `docs/notebooks/html/` and `notebooks/_freeze/`), check-yaml (with `--allow-unknown-tags`), check-added-large-files (1 MB cap, excludes notebook render outputs), check-merge-conflict, debug-statements.
- `ruff-pre-commit v0.15.9`: `ruff-check --fix` + `ruff-format`, both excluding `notebooks/` and `benchmarks/archive/`. Note the pinned rev (0.15.9) is newer than the floor in `pyproject.toml` (`ruff>=0.4`).
- Local hooks: `pip-audit --skip-editable` and `mypy .` (`always_run: true`), plus `quarto-render` running `make notebooks-staged` when any `notebooks/*.qmd` is staged.

## Docs toolchain

- **Quarto** renders `notebooks/*.qmd` (Makefile `notebooks` / `notebooks-staged` targets, `Makefile:3-11`); freeze state lands in `notebooks/_freeze/`.
- **`scripts/generate_notebook_docs.py`** turns `.qmd` frontmatter into mkdocs stub pages under `docs/notebooks/` (script docstring, `scripts/generate_notebook_docs.py:1`).
- **mkdocs-material + mkdocstrings[python] + pymdown-extensions** build the site (`mkdocs.yml:1-23`; mkdocstrings google-style with `show_source`, superfences/mermaid, admonition).
- **mike** is declared in the `docs` extra but versioned deploys are not wired up in CI (GitHub Pages `deploy-pages@v4` is used instead, `.github/workflows/docs.yml`) — see note under extras.
- One command: `make docs` = quarto render + generate stubs + `mkdocs build` (`Makefile:13-15`). `make check-legacy-namespace` runs `scripts/check_legacy_namespace.py` (`Makefile:17-18`), which fails on any `tabular_blueprint`/`tabular-blueprint` token in maintained files (`scripts/check_legacy_namespace.py:9-21`).

## CI workflows (`.github/workflows/`)

| Workflow | Trigger | What runs |
|---|---|---|
| `ci.yml` — `pre-commit` | push/PR to `main`, `develop` | `pre-commit/action@v3.0.1` after `uv sync --frozen --group dev --extra full` |
| `ci.yml` — `typecheck` | same | `uv run mypy .` |
| `ci.yml` — `test` | same | pytest `tests/unit/`, `tests/integration/`, `tests/e2e/` on Python 3.11/3.12/3.13 matrix, with `TABPFN_TOKEN` secret |
| `ci.yml` — `coverage` | same | combined pytest with `--cov=src/iter8ml/engine --cov=src/iter8ml/services --cov=src/iter8ml/config.py`, `--cov-fail-under=70`, then the legacy-namespace check |
| `ci.yml` — `security` | same | `pip-audit -f json`, report uploaded as artifact |
| `benchmarks.yml` | pushes of `v*` tags | `benchmarks/run_openml_benchmark.py --config benchmarks/configs/cpu_benchmark.yaml --quick --fail-on-errors`; results uploaded from `benchmarks/results/` |
| `docs.yml` | push to `main` or `v*` tags | Quarto setup, `uv sync --extra train --extra docs`, quarto render, generate notebook stubs, `mkdocs build`, deploy `site/` to GitHub Pages |

## Conventions checklist (from `AGENTS.md`)

- `uv sync --all-groups` + `uv run pytest`, `ruff check .`, `ruff format --check .`, `mypy .`, `make check-legacy-namespace` before handoff; `make docs` for docs/notebook changes.
- Note the `build/` gotcha above can make the `mypy .` step fail spuriously on machines that have run a raw build — delete `build/` first.
