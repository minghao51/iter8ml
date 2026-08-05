# iter8ml Portfolio Roadmap

**Date:** 2026-08-05
**Status:** Approved (brainstorm)
**Horizon:** 3 months → target portfolio-ready by ~2026-11-07
**Capacity:** Solo, part-time
**Goal:** Convert iter8ml from a feature-rich private framework into a
public, *proven* portfolio showcase that demonstrates applied-ML depth (B),
agentic-AI currency (C), and systems-design rigor (A) — in that priority order.

---

## Positioning

Lead with **B (Applied ML / AutoML)** as the backbone; use **C (agentic)** as
the hero showpiece; use **A (systems)** as optional depth.

| Priority | Angle | What the reviewer remembers |
|----------|-------|------------------------------|
| **B — lead** | AutoML researcher: fast baselines, HPO, calibration, model routing | "Gets strong results fast, knows the methods — and *proves it*." |
| **C — hero** | Agentic AI: MCP server driving the full loop from Claude | "Ships modern LLM-native ML tooling." |
| **A — depth** | Systems: medallion lakehouse, registry, export, observability | "Builds reliable ML infrastructure." |

**Core thesis:** the framework does *not* need more features. It needs
**proof, polish, and a story.** Every roadmap item maps to one of those three.

---

## Constraints (system-validated 2026-08-05)

| Resource | Value | Implication |
|----------|-------|-------------|
| CPU | Intel Core Ultra 5 225H, 14 cores | Parallel GBDT + CV: ✅ strong |
| RAM | 15 GB total, ~10 GB free | **Binding constraint** — cap dataset size |
| GPU | None (WSL2, no CUDA) | Deep models auto-skipped by `ModelSelector`; benchmarks are CPU-only |
| Disk | 951 GB free | Non-issue |
| Tooling | uv 0.12.1; only base extras installed | `uv sync --extra train` required first |

**Benchmark scope decision:** core results are CPU GBDTs
(CatBoost/LightGBM/XGBoost) + baselines. Frame as *"runs on a laptop,
competitive with cloud AutoML."* This is a stronger B-story than a GPU-dependent
one. Deep/foundation models (FT-Transformer, TabNet, TabPFN) are documented as
"supported when GPU present" — a design talking point, not a benchmark gap.

---

## Definition of Done (portfolio-ready)

A reviewer spending 5 minutes should see all of:

1. **README** opens with a **results chart + benchmark table** (proof).
2. **`pip install iter8ml` / `uvx iter8ml` works** — v0.1.0 on PyPI (credibility).
3. **One command** (`iter8 init --demo`) runs end-to-end on a bundled dataset (low friction).
4. **One deep case study** on a recognizable dataset, published as a blog/Quarto (story).
5. **One live demo URL** (upload CSV → leaderboard + SHAP) (accessibility).
6. **One agent showcase** — Claude via MCP running the loop autonomously (wow).
7. CI badges, clean issues/templates, pinned showcase links (hygiene).

---

## Phase 1 — Month 1: Credibility & Polish (prove it + ship it)

Theme: turn "feature-rich" into "proven + installable." This is the floor every
portfolio needs.

| # | Pri | Task | Why | Refs |
|---|-----|------|-----|------|
| 1.1 | **P0** | `uv sync --extra train`, then **run the OpenML benchmark suite** end-to-end on a CPU-tuned config. Commit results JSON + plots. | Highest single ROI: proof it works. | `benchmarks/run_openml_benchmark.py`, `benchmarks/configs/default_benchmark.yaml` |
| 1.2 | **P0** | Create **CPU benchmark config**: drop/cap `covertype` (581k → RAM risk), keep `adult` sequential, mark `tabpfn` optional. | 15 GB RAM constraint. | `benchmarks/configs/` (new `cpu_benchmark.yaml`) |
| 1.3 | **P0** | **Results chart + benchmark table at top of `README.md`**. | First thing reviewers see. | `README.md` |
| 1.4 | **P0** | **Tag v0.1.0 + publish to PyPI** (`uv build` → `uv publish`). Bump classifier `Alpha → Beta`. | Makes the README `uvx` one-liner real. | `pyproject.toml`, GitHub Releases |
| 1.5 | P1 | Land the **queued refactors** (code-simplify, pipeline-spec). | Removes visible debt; interview talking point. | `docs/plan/code-simplify-20260515.md`, `docs/plan/pipeline-spec-20260515.md` |
| 1.6 | P1 | **Cheap security/perf fixes**: MAC on `safe_dump`, harden SQL sanitizer, wire dead `max_workers` → parallel training. | Defensible engineering depth. | `utils/io.py:100`, `data/loader.py:76`, `config.py:163`, `engine/pipelines/nodes/train.py:150` |
| 1.6b | **P0** | **Fix OpenMP deadlock** for lightgbm/xgboost on hybrid-core (P+E) CPUs under WSL2/Linux. `HardwareProfile._get_default_threads()` returns `os.cpu_count()` (=14) which hangs libgomp; benchmark entrypoint already patched (cap 8), but `iter8 run` is still affected. | Framework itself hangs GBDTs on this hardware class. Found + verified 2026-08-05. | `config.py:444`, `benchmarks/run_openml_benchmark.py` |
| 1.7 | P2 | **CI badges + 30-sec demo gif** in README. | Polish. | `README.md`, `.github/workflows/` |

**Exit criteria:** benchmark results committed; README shows a chart; `uvx
iter8ml run` works against PyPI.

---

## Phase 2 — Month 2: Story & Depth (make it accessible & memorable)

Theme: one deep story beats nine shallow ones.

| # | Pri | Task | Why | Refs |
|---|-----|------|-----|------|
| 2.1 | **P0** | **Flagship case study** on a recognizable dataset (e.g. Kaggle Home Credit Default, Telco Churn, or German Credit). Full loop: ingest → preprocess → HPO → drift-over-time → SHAP → export. Publish as polished Quarto/blog. | The artifact reviewers actually read. | `notebooks/`, `docs/notebooks/` |
| 2.2 | **P0** | **Live demo** — tiny FastAPI/Streamlit app: upload CSV → leaderboard + SHAP plot. Deploy free tier (HF Spaces / Render / Fly). | Clickable URL > README. | `services/mcp.py`, new `demo/` |
| 2.3 | P1 | **Bundled demo dataset + `iter8 init --demo`** so the pitch is one command. | Zero-friction first run. | `examples/`, `cli/main.py` |
| 2.4 | P1 | **Trim notebooks to 3 hero docs** (quick-start, case-study, agent-demo); archive the rest under `docs/notebooks/archive/`. | Curation = signal. | `notebooks/` |
| 2.5 | P2 | **"Design decisions" blog post** (content already exists in `ARCHITECTURE.md` + `docs/technical_roadmap.md`). | Shows engineering judgment. | `docs/` |

**Exit criteria:** published case study; live demo URL in README; one-command
demo works.

---

## Phase 3 — Month 3: Differentiation & Reach (the "wow" + visibility)

Theme: the memorable, current showpiece + getting eyes on it.

| # | Pri | Task | Why | Refs |
|---|-----|------|-----|------|
| 3.1 | **P0** | **Agent showcase** — record/screen-capture Claude Desktop driving the full loop via the MCP server (baseline → HPO → drift → export). Post as short video/GIF + writeup. | The single most 2026-relevant artifact; plays to existing strength. | `services/mcp.py` |
| 3.2 | P1 | **External visibility**: Show HN / r/ML post timed to a release; concise LinkedIn/X thread. | Portfolio only works if seen. | — |
| 3.3 | P1 | **Repo surface polish**: GitHub Releases notes, topic tags, issue/PR templates, pin case-study + demo. | First-impression hygiene. | `.github/`, repo settings |
| 3.4 | P2 *(stretch)* | **Complete the medallion story** for show: model-per-fold Platinum + OOF artifacts (called out as future work). Then a "lakehouse for ML" writeup. | Systems-depth (priority A); demoted per B>C>A. | `docs/medallion.md`, `src/iter8ml/dataflows/` |
| 3.5 | P2 *(stretch)* | **ONNX/TorchScript export** — makes "production-ready" real. | Own roadmap "Medium" item. | `services/export.py` |

**Exit criteria:** agent showcase published; at least one external post; repo
surfaces polished.

---

## Explicitly deferred (honest scope for solo + part-time)

To keep the 3-month window realistic, the following are **out of scope** unless
capacity remains:

- Uncertainty quantification; Optuna Dashboard integration (Low in current roadmap).
- Remote data loaders (S3/GCS).
- AFE pruning via RFE/null-importance (Medium in current roadmap) — kept as-is.
- Full DuckDB catalog for medallion (later phase per `docs/medallion.md`).

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Benchmarks blow RAM (covertype/adult) | High | CPU-tuned config (1.2); sequential runs; cap rows. |
| Part-time capacity slips the schedule | Medium | P0 items only are blocking; P1/P2 slide gracefully. Defer list above absorbs overrun. |
| PyPI release exposes a packaging bug | Medium | Smoke-test `uvx --from . iter8 run` *before* publish; cut 0.1.0rc1 first. |
| Live demo attacked / abused | Low-Med | Size + timeout limits on upload; free-tier keep-alive; no persisted user data. |
| Deep-model omission read as a gap | Low | Frame explicitly as "CPU env"; document GPU path as a feature. |

---

## Verification Checklist (portfolio-readiness gate)

Before declaring portfolio-ready, confirm:

1. **Install:** `uvx iter8ml run --demo` works from a clean shell (no local clone).
2. **Proof:** `benchmarks/results/` contains committed JSON + rendered chart in README.
3. **Ship:** `pip install iter8ml == 0.1.0` resolves on PyPI; classifier = Beta.
4. **Story:** case-study URL renders and is linked from README.
5. **Demo:** live URL returns a leaderboard for a sample CSV upload.
6. **Wow:** agent-showcase video/GIF embedded in README.
7. **Hygiene:** CI badges green; Releases page populated; topics tagged.
8. **Debt:** `docs/plan/code-simplify-*` and `pipeline-spec-*` closed or re-scoped.
9. **Quality:** `uv run ruff check .`, `uv run mypy src/iter8ml/`, `uv run pytest` all green.

---

## Immediate next actions (Week 1)

1. `uv sync --extra train`
2. Author `benchmarks/configs/cpu_benchmark.yaml` (subset of `default_benchmark.yaml`).
3. `uv run python -m benchmarks.run_openml_benchmark` → inspect `benchmarks/results/`.
4. Commit results; draft README chart (1.3).
