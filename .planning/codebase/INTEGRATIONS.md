# External Integrations Reference

- **Last updated:** 2026-08-29 (generated from codebase audit)
- **Audience:** internal — agents/contributors; not published to the mkdocs site.

Governing rule (AGENTS.md → *Architectural boundaries*): optional integrations
(wandb, mlflow, llm, mcp) stay **additive** behind extras/config — an
integration failure must never break the core run loop. Reliability split per
[ADR-0005](../../docs/decisions/0005-two-reliability-seams.md): tracker events
are best-effort; state publication is required.

## Status board

| Integration | Seam | Gate | Status |
|---|---|---|---|
| MCP server | `src/iter8ml/services/mcp.py` + `src/iter8ml/cli/mcp.py` | `train` extra (`mcp>=0.9`) | used (CLI + unit tests); Phase-3 showcase pending |
| LLM commentary (litellm) | `src/iter8ml/services/llm.py` | config flag, off by default | used (opt-in) |
| W&B tracker | `src/iter8ml/engine/tracker.py:99` | `train` extra + manual injection | available, manual-only (not auto-selected) |
| MLflow tracker | `src/iter8ml/engine/tracker.py:131` | base dep + manual injection | available, manual-only |
| JSONL tracker (default) | `src/iter8ml/engine/tracker.py:23` | none — always available | used (the default) |
| HF Spaces demo deploy | `scripts/deploy_hf.py` + `demo/app.py` | manual, `HF_TOKEN` env | dormant (free tier blocked; kept for Phase 3) |
| GitHub Pages / mkdocs | `.github/workflows/docs.yml` | push to `main` or tag `v*` | used (CI-only) |

---

## MCP server (Model Context Protocol)

- **What:** exposes 9 atomic tools for LLM agents — `get_experiment_state`,
  `get_column_stats`, `run_baseline`, `run_hpo`, `get_event_log`,
  `registry_show`, `registry_promote`, `detect_drift`, `export_champion`
  (`src/iter8ml/services/mcp.py:59,65,80,97,132,143,153,160,189`). Tools
  delegate to `ExperimentSession` / existing services, not re-implementations.
- **Location (verified):** server + tools in `src/iter8ml/services/mcp.py`;
  CLI entry `iter8 mcp` in `src/iter8ml/cli/mcp.py:11` (registered via
  `src/iter8ml/cli/__init__.py`). Documented in `README.md:193`.
- **Enabled by:** `mcp>=0.9` in the `train` extra (`pyproject.toml:59`). FastMCP
  is initialized lazily (`_init_mcp` at `services/mcp.py:29`, module
  `__getattr__` at `:39`) so importing the module never requires `mcp`.
- **Failure mode:** `iter8 mcp` without the package prints an install hint and
  exits 1 (`cli/mcp.py:17-21`); core loop never imports `mcp` at module load.
  Unit tests skip gracefully (`pytest.importorskip` at
  `tests/unit/test_mcp_tools.py:15`).
- **Status:** used — tools tested in `tests/unit/test_mcp_tools.py` (TabPFN
  paths skip without `TABPFN_TOKEN`, `tests/unit/test_mcp_tools.py:29,82-84`).
  The consumer story (Claude Desktop driving the loop) is the pending Phase-3
  hero item 3.1 (`docs/plan/portfolio-roadmap-20260805.md`). Note
  `run_baseline` defaults to `models=["tabpfn", "catboost"]`
  (`services/mcp.py:90`) — TabPFN needs its license token (below).
- **Bounded by:** additive-only rule; no ADR specifically for MCP.

## LLM commentary via litellm

- **What:** `TabularAgent` generates natural-language SHAP/performance
  commentary embedded in `current_state.md` (`src/iter8ml/services/llm.py:34`).
- **Seam:** called only from `StateObserver._render_llm_commentary`
  (`src/iter8ml/engine/state_observer.py:252`, agent factory at `:240`), which
  the trainer's state adapter reaches through
  `src/iter8ml/engine/trainer_factory.py:42-53`.
- **Enabled by:** config flag, off by default — `ExperimentConfig.llm_enabled:
  bool = False` (`src/iter8ml/config.py:218`); also `iter8 analyze state
  --llm` (`src/iter8ml/cli/analyze.py:92-101`) and
  `ExperimentSession.state(llm_enabled=True)` (`src/iter8ml/session.py:104`).
  Model via env `ITER8ML_LLM_MODEL` → legacy alias `TABBLUEPRINT_LLM_MODEL` →
  default `claude-sonnet-4-20250514` (`config.py:29`, `services/llm.py:23-27`).
  Optional `api_key_env`/`api_base` on `LLMAgentConfig` (`services/llm.py:20`);
  provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are read by litellm
  itself (`.env.example:21-24`).
- **Package gate:** ⚠️ `litellm>=1.83.10` is a **base** dependency
  (`pyproject.toml:31`) and again in the `train` extra (`:60`). This deviates
  from the extras-gating pattern used for `mcp`/`wandb` — behavior is still
  gated by `llm_enabled=False`, but the install is not optional. See
  [CONCERNS.md](CONCERNS.md) (Active risk: base-dep drift).
- **Failure mode:** never raises into the run — missing litellm returns
  `"[LLM disabled: …]"`, any call error returns `"[LLM error: …]"`
  (`services/llm.py:101-104`); disabled agent returns empty commentary
  (`services/llm.py:38,50,62`). litellm is imported inside `_call_llm`
  (`services/llm.py:79`), never at module import.
- **Status:** used (opt-in); tested with mocks in
  `tests/unit/test_llm_agent.py`.

## W&B tracker (WandbTracker)

- **What:** mirrors all tracker events to a Weights & Biases run —
  `src/iter8ml/engine/tracker.py:99` (metrics/params/artifacts/events/finish).
- **Enabled by:** `wandb>=0.17` in the `train` extra (`pyproject.toml:57`); no
  core import (lazy `import wandb` in `__init__`, `tracker.py:104`). Auth via
  `WANDB_API_KEY` (`.env.example:29`) — read by the wandb SDK, not by iter8ml
  code. Instantiate manually: `ExperimentSession(tracker=WandbTracker(...))`
  (`src/iter8ml/session.py:31`).
- **Failure mode:** `ImportError` with install hint raised at construction
  (`tracker.py:108-110`) — fails fast at opt-in, before the run starts. Once
  running, tracker calls go through the best-effort event adapter
  (`src/iter8ml/engine/trainer.py:150-160` logs a warning and continues, per
  ADR-0005).
- **Status:** available / manual-only. ⚠️ `ExperimentConfig.tracker`
  (`TrackerType` enum, `src/iter8ml/config.py:193`, `src/iter8ml/constants.py:28`)
  is **declared but not wired**: nothing maps `TrackerType.WANDB` to a
  `WandbTracker` instance — the trainer falls back to `JSONLTracker` when no
  tracker is injected (`src/iter8ml/engine/trainer.py:48-51`). Setting
  `tracker="wandb"` in config currently has no effect. Tests mock wandb
  (`tests/unit/test_tracker_rotation.py:171-252`).

## MLflow tracker (MLflowTracker)

- **What:** logs metrics/params/artifacts/events to a local or remote MLflow
  server — `src/iter8ml/engine/tracker.py:131`; `log_event` maps scalars to
  params and dicts to `log_dict` (`tracker.py:159-161`).
- **Enabled by:** ⚠️ `mlflow>=3.11.1` is a **base** dependency
  (`pyproject.toml:32`) — and is *also* pinned in the `train` extra as
  `mlflow>=2.13` (`pyproject.toml:58`). Remote server via
  `MLflowTracker(tracking_uri=...)` or the `MLFLOW_TRACKING_URI` env var
  (`.env.example:30`, honored by the mlflow SDK when `tracking_uri=None`).
  Inject manually via `ExperimentSession(tracker=...)` like W&B.
- **Failure mode:** same as W&B — `ImportError` at construction with install
  hint (`tracker.py:135-145`); post-construction failures ride the best-effort
  event seam (ADR-0005).
- **Status:** available / manual-only; same unwired-`TrackerType` caveat as W&B.
  The duplicate/conflicting extra pin is tracked in
  [CONCERNS.md](CONCERNS.md).

## JSONL tracker (default)

- **What:** structured event log at `workspace/experiments.jsonl` with size
  rotation (100 MB × 5 backups) and a thread lock —
  `src/iter8ml/engine/tracker.py:23` (rotation `:46-96`, lock `:132-141`).
- **Enabled by:** none — this is the default. `Trainer` constructs it when no
  tracker is injected (`src/iter8ml/engine/trainer.py:51`); `run_hpo` in CLI
  and MCP construct it explicitly (`src/iter8ml/cli/optimize.py:30`,
  `src/iter8ml/services/mcp.py:110-113`).
- **Failure mode:** best-effort — event publish failures are logged, never
  fatal (ADR-0005; `src/iter8ml/engine/trainer.py:150-160`). Note: per
  [ADR-0003](../../docs/decisions/0003-medallion-artifact-contract.md) the
  event history alone is never a checkpoint; durable state lives in medallion
  artifacts + manifest.
- **Status:** used — the always-on telemetry path; tested in
  `tests/unit/test_tracker_rotation.py`, `tests/unit/test_jsonl.py`.

## Hugging Face Spaces demo deploy

- **What:** one-shot deploy of the Gradio demo (`demo/app.py`) to a public HF
  Space — `scripts/deploy_hf.py:22` self-discovers the username from the token,
  creates the Space idempotently, uploads `demo/` minus `README.md` (the
  Space's YAML front-matter must not be clobbered — `deploy_hf.py:36-42`).
- **Enabled by:** manual invocation only: `HF_TOKEN` env (`.env.example:47`) +
  `uv run --with huggingface_hub python scripts/deploy_hf.py`
  (`scripts/deploy_hf.py:8-10`). No CI wiring. The Space installs the released
  PyPI package (`iter8ml[gbdt]>=0.1.0`, `demo/requirements.txt`).
- **App guard rails** (free-tier friendly): 20k-row cap, CatBoost+XGBoost only,
  5-fold CV, `max_workers=1`, throwaway per-request workspace, OpenMP cap at
  import (`demo/app.py:38`, guard-rail constants `:43-46`); Gradio UI built
  lazily so the
  `run_analysis()` core stays importable without gradio (`demo/app.py:164`,
  `:218-225`).
- **Failure mode:** purely additive — nothing in `src/iter8ml` imports gradio
  or huggingface_hub; a failed deploy affects only the public URL.
- **Status:** **dormant** — HF free tier OOMs at 512 MB (needs PRO); Phase 2
  pivoted to Quarto→Pages + Colab. Script + app kept for Phase 3; the Gradio
  auto-API around `run_analysis()` is the planned seam for the Phase-3 agent
  showcase (`docs/plan/deferred-research.md` §2).

## GitHub Pages / mkdocs publishing

- **What:** renders Quarto notebooks → generates doc stubs → builds the mkdocs
  site → deploys to GitHub Pages — `.github/workflows/docs.yml:28-41`
  (`quarto render notebooks/`, `scripts/generate_notebook_docs.py`,
  `mkdocs build`, `actions/deploy-pages@v4`). Site URL:
  `https://minghao51.github.io/iter8ml/` (`mkdocs.yml:2`).
- **Enabled by:** push to `main` or tag `v*` (`.github/workflows/docs.yml:2-5`);
  one concurrent deployment (`concurrency: pages`).
- **Failure mode:** docs-only; a failed render cannot affect the package.
  Local equivalent: `make docs` (`Makefile`). Note the pre-commit
  `quarto-render` hook blocks local commits of `.qmd` changes when quarto is
  absent (`.pre-commit-config.yaml:40-44`) — see
  [CONCERNS.md](CONCERNS.md). `mike>=2.0` is in the `docs` extra
  (`pyproject.toml:68`) but is not used by `docs.yml` — unverified whether it
  has a manual versioning role.
- **Status:** used (CI-only publishing path).

## Adjacent external touchpoints (brief)

- **TabPFN license:** `TABPFN_TOKEN` env (`.env.example:12`) is consumed by the
  `tabpfn` library itself (no read in `src/`); CI injects it as a secret
  (`.github/workflows/ci.yml:41,56`); tests skip TabPFN paths without it
  (`tests/unit/test_mcp_tools.py:82-84`).
- **OpenML benchmarks:** `benchmarks/` fetches datasets and runs on tag pushes
  (`.github/workflows/benchmarks.yml`) — dev/benchmark surface, not a runtime
  integration.
- **ZenML example:** `examples/zenml_pipeline.py` is entirely commented out and
  references the legacy `tabular_blueprint` namespace — dormant documentation
  artifact; not covered by `make check-legacy-namespace` scan globs
  (`scripts/check_legacy_namespace.py:8-19`). See [CONCERNS.md](CONCERNS.md).
