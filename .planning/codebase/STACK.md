# Stack

## Language & Runtime
- **Python** >=3.11 (CI tests: 3.11, 3.12, 3.13; Docker: 3.11)
- **Package manager**: `uv` (via `astral-sh/setup-uv` in CI, `FROM astral.sh/uv/install.sh` in Docker)
- **Build system**: setuptools + wheel

## Project
- **Name**: `tabular-blueprint` v0.1.0
- **Entry point**: `tabblueprint` → `tabular_blueprint.cli:app` (typer CLI)
- **Source layout**: `src/tabular_blueprint/` (configured in `[tool.setuptools.packages.find] where = ["src"]`)

## Core Dependencies (`dependencies` in pyproject.toml)
| Package | Version | Import path | Role |
|---|---|---|---|
| `polars` | >=1.0 | `polars` (as `pl`) | DataFrame engine |
| `pydantic` | >=2.0 | `pydantic` | Config models (BaseModel) |
| `catboost` | >=1.2 | `catboost` | Gradient boosting (CPU/GPU) |
| `lightgbm` | >=4.0 | `lightgbm` | Gradient boosting |
| `xgboost` | >=2.0 | `xgboost` | Gradient boosting |
| `optuna` | >=3.6 | `optuna` | Hyperparameter optimization |
| `scikit-learn` | >=1.4 | `sklearn` | Metrics, preprocessing, CV |
| `numpy` | >=1.26 | `numpy` (as `np`) | Array ops |
| `typer` | >=0.12 | `typer` | CLI framework |
| `psutil` | >=5.9 | `psutil` | Hardware detection |
| `rich` | >=13.0 | `rich` | CLI tables / formatting |
| `joblib` | >=1.3 | `joblib` | Parallel execution |
| `filelock` | >=3.12 | `filelock` | Cross-process file locking |
| `pyyaml` | >=6.0 | `yaml` | YAML config parsing |
| `sf-hamilton` | >=1.70 | `hamilton` / `sf_hamilton` | DAG orchestration |

## Optional Dependencies

### `[deep]` — Deep learning
| Package | Version | Import path |
|---|---|---|
| `torch` | >=2.3 | `torch` |
| `accelerate` | >=0.30 | `accelerate` |
| `transformers` | >=4.40 | `transformers` |
| `tabpfn` | >=2.0 | `tabpfn` |
| `pytorch-tabular` | >=1.0 | `pytorch_tabular` |
| `datasets` | >=2.14 | `datasets` |

### `[shap]` — Explainability
| `shap` | >=0.44 | `shap` |

### `[cleanlab]` — Data quality
| `cleanlab` | >=2.6 | `cleanlab` |

### `[llm]` — LLM integration
| `mcp` | >=0.9 | `mcp` | MCP server |
| `litellm` | >=1.40 | `litellm` | LLM provider proxy |

### `[wandb]` — Experiment tracking
| `wandb` | >=0.17 | `wandb` |

### `[mlflow]` — Experiment tracking
| `mlflow` | >=2.13 | `mlflow` |

### `[all]` — Meta extra
`tabular-blueprint[deep,shap,cleanlab,llm,wandb,mlflow]`

## Dev tooling (`[dependency-groups] dev`)

| Tool | Version | Purpose |
|---|---|---|
| `pytest` | >=9.0.3 | Test runner |
| `pytest-cov` | >=5.0 | Coverage reporting |
| `ruff` | >=0.4 | Linter + formatter |
| `pre-commit` | >=3.6 | Git hook framework |
| `mypy` | >=1.10 | Static type checking |
| `pip-audit` | >=2.7 | Dependency vulnerability scanning |
| `marimo` | >=0.23.4 | Reactive notebook environment |

### Docs dependencies (`[dependency-groups] docs`)

| Tool | Version | Purpose |
|---|---|---|
| `mkdocs-material` | >=9.5 | Documentation theme |
| `mkdocstrings[python]` | >=0.25 | Auto-generated API docs |
| `mike` | >=2.0 | Versioned docs deployment |
| `pymdown-extensions` | >=10.0 | Markdown extensions |

## CI/CD (GitHub Actions)

- **File**: `.github/workflows/ci.yml` — push to `main`/`develop` + PR
  - `pre-commit` job: `uv sync --frozen --group dev --extra llm` → `pre-commit/action@v3.0.1`
  - `typecheck` job: `uv sync --frozen --group dev --extra llm` → `uv run mypy .`
  - `test` job: 3.11 / 3.12 / 3.13 matrix → `uv run pytest tests/unit/`, `tests/integration/`, `tests/e2e/`
  - `coverage` job: 70% threshold on `engine/`, `services/`, `config.py`
  - `security` job: `uv run pip-audit -f json -o pip-audit.json`

- **File**: `.github/workflows/docs.yml` — push to `main` + tags `v*`
  - Exports notebook static HTML via `scripts/export-notebooks.sh`
  - Deploys via `uv run mike deploy --push --update-aliases`

## Pre-commit Hooks (`.pre-commit-config.yaml`)

| Hook | Source | Actions |
|---|---|---|
| `trailing-whitespace` | `pre-commit-hooks` v5.0.0 | Trim trailing whitespace |
| `end-of-file-fixer` | same | Ensure newline at EOF |
| `check-yaml` | same | Validate YAML |
| `check-added-large-files` | same | Warn on large files |
| `check-merge-conflict` | same | Detect merge conflict markers |
| `debug-statements` | same | Catch `pdb`/`ipdb` left in code |
| `uv-lock` | `uv-pre-commit` 0.6.14 | Keep `uv.lock` in sync |
| `ruff` | `ruff-pre-commit` v0.15.9 | Lint + auto-fix |
| `ruff-format` | same | Formatter |
| `mypy` | local (`uv run mypy .`) | Type checking |

## Ruff config

- Line length: 100
- Target: `py311`
- Selected rules: E, F, I, UP, B, SIM, C4, PT, RUF
- Fixable: ALL
- Per-file ignores: `cli.py` → B008; notebooks → E402/I001/B018/E501/RUF001/F841

## Mypy config

- `python_version = "3.11"`
- `disallow_untyped_defs = true`
- Excludes: tests/, benchmarks/, notebooks/, workspace/
- Overrides for third-party stubs (catboost, lightgbm, xgboost, optuna, polars, hamilton, cleanlab, shap, wandb, mlflow, litellm, mcp, torch, transformers, etc.)
- Error-ignored modules: `hpo`, `hpo_warmstart`, `hpo_importance`, `pipelines.nodes.*`, `trainer`

## Pytest config

- Test paths: `tests/`
- Source path: `src`
- Strict markers: slow, integration, unit, e2e, serial, network, smoke
- Default flags: `--strict-markers -ra --durations=10 --import-mode=importlib`

## Docker

- **Base**: `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
- **Python**: 3.11 via apt
- **uv**: installed via `curl -Ls https://astral.sh/uv/install.sh | sh`
- **Build**: `uv sync --frozen --no-dev` (no dev/optional extras in image)
- **docker-compose**: app service (build: .) + mlflow sidecar (`ghcr.io/mlflow/mlflow:v2.13.0`)

## Documentation (MkDocs)

- **Theme**: `mkdocs-material` with nav tabs, sections, top button, code copy
- **API docs**: `mkdocstrings[python]` with Google-style docstrings
- **Versioning**: `mike`
- **Extensions**: `pymdownx.blocks.caption`, `md_in_html`
- **Notebooks**: exported to `docs/notebooks/exports/` as HTML via `scripts/export-notebooks.sh`
- **Deployed at**: `https://minghao51.github.io/iter8ml/`
