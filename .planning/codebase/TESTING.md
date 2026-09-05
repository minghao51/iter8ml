# Testing

- **Last updated:** 2026-08-29 (generated from codebase audit)
- **Audience:** internal — agents/contributors; not published to the mkdocs site.

Grounded in `tests/conftest.py` (36 lines), `tests/strategies.py` (190 lines),
`pyproject.toml` `[tool.pytest.ini_options]`, and the `tests/` tree.

## Runner & config

- **pytest** with `--strict-markers`, `--import-mode=importlib`,
  `pythonpath = ["src"]` (`pyproject.toml`). Tests import `iter8ml.*` (absolute),
  never relative.
- **hypothesis** for property-based tests.
- `filterwarnings` ignores the `scipy.optimize` DeprecationWarning.

## Tiers & layout

| Tier | Dir | Count (≈) | Notes |
|------|-----|-----------|-------|
| `unit` | `tests/unit/` | 57 files | Fast, isolated; no heavy ML unless needed |
| `integration` | `tests/integration/` | 8 files | Multi-component; own `conftest.py`; auto-marked `slow` |
| `e2e` | `tests/e2e/` | 1 file | Full-pipeline smoke test (`test_smoke.py`) |

Auto-marking in `conftest.py` (`pytest_collection_modifyitems`): `unit/` →
`unit`; `integration/` → `integration`+`slow`; `e2e/` → `e2e`+`slow`.

**Markers** (`pyproject.toml`): `slow`, `integration`, `e2e`, `network`,
`serial`, `smoke`, `property`, `metamorphic`, `contract`, `differential`.
Examples: `test_contract_api.py`, `test_metamorphic_drift.py`,
`test_metamorphic_features.py`, `test_metamorphic_leakage.py`,
`test_mcp_tools.py`, `test_llm_agent.py`.

## Fixtures (`tests/conftest.py`)

- `classification_data` / `regression_data` — session-scoped synthetic frames
  (`make_classification` / `make_regression`) for expensive data generation.
- `tmp_workspace` — `tmp_path / "workspace"` for isolated `Workspace` use
  (overrides the default `workspace/` root). Also set env `ITER8ML_WORKSPACE`
  for scripts run against a temp root.

## Property tests (`tests/strategies.py`)

- Shared `@st.composite` generators: dataframes, numpy arrays, JSONL events.
- Imported by `property`-marked suites (metamorphic/contract/differential tests).
- Run only: `uv run pytest -m property`.
- Skip slow/network: `uv run pytest -m "not slow"`.

## How to run

```bash
uv run pytest tests/unit/ -v --tb=short          # fast tier
uv run pytest tests/integration/ -v --tb=short    # multi-component
uv run pytest tests/e2e/ -v --tb=short            # smoke
uv run pytest tests/                              # all (needs gbdt extra for some)
```

- **Runtime:** unit tier is seconds; integration/e2e pull in GBDT/Hamilton and are
  much slower (auto-`slow`). CI runs all three tiers + coverage.
- **gbdt extra:** a subset of unit/integration tests require `catboost`/`lightgbm`/
  `xgboost` (the `[gbdt]` extra) — run `uv sync --extra gbdt` first or those
  collect as skips/errors.
- **GPU paths:** not exercised without a GPU; no CUDA-tagged tests in-tree — do
  not assume GPU coverage.

## Invariants pinned (contract tests)

- `tests/unit/test_medallion_contracts.py` — enforces the medallion artifact
  contract (ADR-0003): atomic manifest + `_SUCCESS`, deep-checksum resume, and
  **train/validation split-overlap rejection within a fold**.
- `tests/unit/test_split_wiring.py` — split/fold wiring correctness
  (row_id / fold / role / repeat membership).

## CI invocation (`.github/workflows/`)

Per `STYLE.md`: unit/integration/e2e each verbose, then a combined run with
`--cov=src/iter8ml/engine,services,config.py --cov-fail-under=70`. (Verify the
exact workflow file names/commands before relying on this — CI yaml not audited
here.)

## Known gaps

- No pandas-related tests needed (Polars-only rule), but no negative test asserts
  "no pandas import in src" — `make check-legacy-namespace` covers legacy
  namespace, not pandas.
- Coverage gate (70%) gates the merged run; a green unit tier alone is not enough.
