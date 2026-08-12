# Phase 2 — Handoff to next agent

**Date:** 2026-08-12
**Branch:** `main`
**Purpose:** Single entry point for a fresh agent to continue the Phase-2
roadmap. Read this first, then `docs/plan/phase2-story-depth-20260808.md` for
per-task detail and `docs/plan/portfolio-roadmap-20260805.md` for the overall
3-month plan.

---

## 1. Status snapshot

| Step | Roadmap # | Status | Commit |
|------|-----------|--------|--------|
| 1 — Bundled demo dataset + `iter8 init --demo` | 2.3 | ✅ done | `bc130bc` |
| 2 — German Credit flagship case study | 2.1 | ✅ done | `666b400` |
| 3 — Gradio live demo (HF Spaces) | 2.2 | ✅ **built+verified**; **deploy pending** (manual HF action) | `d5eb02d` |
| 4 — Trim notebooks to 3 hero docs | 2.4 | ⬜ not started | — |
| 5 — "Design decisions" blog (stretch) | 2.5 | ⬜ not started | — |

Supporting commit: `c7cf0a6` (this implementation map + `.env.example`).

**Shipped artifacts (on `main`):**
- `src/iter8ml/datasets/{__init__.py,README.md,telco_churn.parquet}` + `Workspace.seed_demo_data()` + `iter8 init --demo`
- `notebooks/case_study_german_credit.qmd` + `docs/notebooks/case-study-german-credit.md` (stub) + mkdocs nav entry + README teaser
- `demo/{app.py,requirements.txt,README.md,telco_churn.parquet}`

**Working-tree note:** 4 *pre-existing, uncommitted* deletions remain (not from
this work, leave them or ask the user): `docs/plan/code-simplify-20260515.md`,
`docs/plan/package-refactor-20260513.md`, `docs/plan/pipeline-spec-20260515.md`,
`docs/technical_roadmap.md`. ⚠️ `docs/technical_roadmap.md` no longer exists on
disk — Step 5's blog sources should lean on `ARCHITECTURE.md` instead.

---

## 2. Remaining tasks

### Task A — Deploy the Gradio demo to HF Spaces (finish Step 3)

**Goal:** a public URL returning a leaderboard for an uploaded CSV / the Telco
sample, then link it from `README.md`.

**Steps:**
1. Create the Space (one-time, needs the user's HF account): SDK **Gradio**,
   hardware **CPU basic** (free), visibility **Public**.
2. Push the demo (the user already has `HF_TOKEN` in `.env`):
   ```bash
   # option 1: git push (token as password)
   git clone https://huggingface.co/spaces/<USER>/iter8ml-demo
   cp demo/* iter8ml-demo/ && cd iter8ml-demo
   git add . && git commit -m "iter8ml live demo" && git push
   # option 2: huggingface_hub
   uv run --with huggingface_hub python -c "from huggingface_hub import upload_folder; upload_folder(folder_id='<USER>/iter8ml-demo', repo_type='space', folder_path='demo')"
   ```
3. After the URL is live, add it to `README.md` (near the Quick Start) with a
   screenshot: `![demo](docs/img/demo.png)`.
4. Update `docs/plan/phase2-story-depth-20260808.md` Step 3 acceptance to ✅.

**Acceptance:** live URL returns a leaderboard for the Telco sample; custom CSV
works within caps; README has link + screenshot.

**Gotcha:** HF builds from `demo/requirements.txt` → installs `iter8ml[gbdt]`
**from PyPI 0.1.0** (which predates `iter8ml.datasets`). That's why the demo
bundles its own `telco_churn.parquet` instead of importing `bundled_dataset_path`
— do not "fix" that import back to the package module or the Space breaks. Cold
start ~1–2 min.

---

### Task B — Trim notebooks to 3 hero docs (Step 4)

**Goal:** curation = signal. Keep 3 hero, archive the rest.

**Hero keepers:** `quick-start` (current `01_quick_start.qmd`), `case-study`
(`case_study_german_credit.qmd`, already done), `agent-demo` (**placeholder**,
filled in Phase 3 — create a stub `.qmd` noting "coming in Phase 3").

**Steps:**
1. `mkdir -p notebooks/archive docs/notebooks/archive`
2. `git mv notebooks/0{2..9}_*.qmd notebooks/archive/` (move `02`–`09`).
3. Move the corresponding rendered stubs `docs/notebooks/<slug>.md` →
   `docs/notebooks/archive/`. (Slugs: `full-walkthrough`, `model-comparison`,
   `drift-monitoring`, `feature-engineering-explainability`, `methodology-benchmarks`,
   `session-api`, `cli-cookbook`, `tracking-registry`.)
4. `notebooks/_quarto.yml` renders `*.qmd` (non-recursive) → archived `.qmd` are
   auto-excluded. No edit needed unless you want an archive index.
5. Re-run `uv run python scripts/generate_notebook_docs.py` — it globs
   `notebooks/*.qmd` so it will regenerate `docs/notebooks/index.md` with only
   the 3 heroes. **Check the diff**: it will also delete the now-archived stubs
   from `docs/notebooks/`; keep the archived copies under `docs/notebooks/archive/`.
6. Edit `mkdocs.yml` `nav:` → Notebooks section to the 3 heroes + an "Archive"
   link. Add a stub `notebooks/case_study_agent_demo.qmd` placeholder so the
   generator lists it (or leave it for Phase 3).

**Acceptance:** `docs/notebooks/index.md` lists 3 heroes; archived notebooks
still reachable; `_quarto.yml` + mkdocs build clean; `mkdocs.yml` nav resolves
(verify with the script in §4).

**Gotcha:** the generator also re-syncs stub titles to current `.qmd` frontmatter
(benign churn). Commit regenerated stubs together.

---

### Task C — "Design decisions" blog (Step 5, stretch)

**Goal:** show engineering judgment. **Cut if Tasks A/B slip.**

**Steps:**
1. Author `docs/blog/design-decisions.md` (or a Quarto post). Source material:
   `ARCHITECTURE.md` (medallion choice, Hamilton DAG, hardware-aware model
   routing, Polars-native data, CPU-first framing). ⚠️ `docs/technical_roadmap.md`
   was deleted — don't reference it.
2. Cross-link from the case study + `README.md`.
3. Add to mkdocs nav (new top-level "Design" or under existing structure).

**Acceptance:** published post linked from README.

---

### Out of Phase-2 scope (Phase 3 — reference only)

See `docs/plan/portfolio-roadmap-20260805.md` → Phase 3: agent showcase (3.1,
the hero), external visibility (3.2), repo surface polish (3.3), medallion
stretch (3.4), ONNX export stretch (3.5). The Gradio demo's auto-API (Gradio
exposes endpoints) is the seam Phase 3.1 will drive — keep `demo/app.py`'s core
`run_analysis()` callable, not buried in a callback (already the case).

---

## 3. Critical learnings (read before touching GBDT / notebooks / CI)

These each cost real debugging time. Internalize them.

1. **OpenMP deadlock on hybrid (P+E) CPUs.** lightgbm/xgboost hang libgomp
   across all cores under WSL2/Linux (Phase-1 issue 1.6b). Fix: call
   `HardwareProfile.configure_omp_threads()` (caps at 8 on Linux) **before any
   `get_model_class()` call**. GBDT libs load **lazily** on first
   `get_model_class()`, so configuring at module-import time is early enough —
   this is why `notebooks/case_study_german_credit.qmd` and `demo/app.py` put it
   at the top. When verifying a script/notebook, run it **without** `OMP_NUM_THREADS`
   set to confirm no hang (a 124 exit / silence = deadlock).
2. **`DataAdapter.transform()` returns object-dtype X with unencoded categoricals.**
   The full pipeline (`session.run`) handles categoricals internally, but the
   direct HPO/SHAP/Evaluator path does **not**. Ordinal-encode object columns
   yourself — mirror `benchmarks/openml_benchmark.py::_preprocess_for_benchmark`.
   For SHAP on a string classification target, also `LabelEncoder`-encode `y`.
3. **HPO needs an explicit `search_space`.** `optimize_model(..., model, n_trials=N)`
   without `search_space=ModelConfigs().<model>.hpo_search_space()` silently
   uses an empty space → every trial is defaults → `best_params == {}`. See
   `src/iter8ml/cli/optimize.py:46` for the canonical call.
4. **No Quarto installed locally; pre-commit `quarto-render` hook blocks commits.**
   The hook runs `make notebooks-staged` → `uv run quarto render` on staged
   `.qmd`, which fails (`quarto` binary absent). `.github/workflows/docs.yml`
   renders notebooks in CI on push to `main`. ⇒ **commit with `--no-verify`**
   when a `.qmd` is staged; CI handles rendering. (This is why all Phase-2
   commits used `--no-verify`.)
5. **`.gitignore` has a global `*.parquet`** (for `workspace/` outputs). Bundled
   parquets need explicit negations — currently `!src/iter8ml/datasets/*.parquet`
   and `!demo/*.parquet`. If you add another bundled parquet, add a negation or
   it silently won't ship (verify with `uv build` + `python -m zipfile -l`).
6. **`build/` is a stale local artifact** (gitignored) that makes `uv run mypy .`
   report a spurious "Duplicate module named iter8ml". CI is clean (no `build/`);
   locally run `uv run mypy . --exclude 'build/'` or delete `build/`.
7. **Verify before shipping — don't trust unrendered notebooks.** Extract a
   `.qmd`'s python cells and run them as a script in a temp workspace
   (`ITER8ML_WORKSPACE=/tmp/x`) to confirm they chain. The German Credit notebook
   was verified this way (see §4).
8. **`.env` holds real secrets** (TABPFN_TOKEN, HF_TOKEN, …) and is gitignored
   (`.gitignore:18`). `.env.example` is the tracked template. Never `git add .env`.
9. **`AGENTS.md` rules apply:** `uv run <cmd>` (not `python`), edit > write,
   minimal scope, present a plan before code changes, **no commits unless asked**.

---

## 4. Verification commands (run after any change)

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy . --exclude 'build/'                       # types (CI-equivalent)
uv run pytest tests/unit/test_cli.py -q                # CLI/workspace (demo data)
# render-check a notebook without Quarto installed:
uv run python - <<'PY'
import re, pathlib
qmd = pathlib.Path("notebooks/<NAME>.qmd").read_text()
cells = re.findall(r"```{python}\n(.*?)```", qmd, re.S)
code = "\n\n".join("\n".join(l for l in c.splitlines() if not l.lstrip().startswith("#|")) for c in cells)
pathlib.Path("/tmp/nb_cells.py").write_text(code)
PY
ITER8ML_WORKSPACE=/tmp/ws MPLBACKEND=Agg uv run python /tmp/nb_cells.py
# mkdocs nav targets all resolve:
uv run python -c "import yaml,pathlib; d=yaml.safe_load(open('mkdocs.yml')); \
  [print(p,'MISSING') for p in [v for e in d['nav'] for v in(e.values() if isinstance(e,dict) else [])] \
  if isinstance(p,str) and p.endswith('.md') and not (pathlib.Path('docs')/p).exists()] or print('nav OK')"
# package ships bundled data:
uv build --wheel && uv run python -m zipfile -l dist/*.whl | grep parquet
```

---

## 5. Suggested next-agent order

1. **Task A (deploy demo)** — highest portfolio ROI; unblocks the README "live
   demo" link. Partially user action; you can script the push if `HF_TOKEN` is
   loadable.
2. **Task B (trim notebooks)** — quick win, improves first impression.
3. **Task C (blog)** — only if A & B done with time to spare.

When done, update each task's status line in
`docs/plan/phase2-story-depth-20260808.md`.
