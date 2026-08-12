# Phase 2 — Story & Depth (Implementation Map / Handoff)

**Date:** 2026-08-08
**Status:** Draft (awaiting execution)
**Horizon:** ~1 month (Month 2 of `portfolio-roadmap-20260805.md`)
**Capacity:** Solo, part-time
**Theme:** One deep story beats nine shallow ones. Make iter8ml *accessible*
(one-command demo) and *memorable* (a case study reviewers actually read + a
clickable live demo).

**Source:** `docs/plan/portfolio-roadmap-20260805.md` → Phase 2 (tasks 2.1–2.5).
**Phase 1 status:** shipped — v0.1.0 on PyPI, CPU benchmark results committed,
OpenMP fix landed, CI/badges added. This phase builds on that proof.

---

## Decisions locked (2026-08-08)

| ID | Decision | Choice |
|----|----------|--------|
| D1 | Flagship datasets | **German Credit** (case study, 2.1) + **Telco Churn** (bundled `--demo` + live demo default, 2.2/2.3) |
| D2 | Demo stack | **Gradio on HF Spaces** (auto-exposes an API for Phase 3 agent reuse) |
| D3 | Drift framing | Constructed reference-vs-current split (matches `iter8 drift` interface); stated honestly |
| D4 | Case study host | **GitHub Pages** (renders `.qmd`, zero cost, same repo) |
| D5 | P2.5 design-decisions blog | **Stretch / in-if-time** (content exists in `ARCHITECTURE.md`) |
| D6 | Sequencing + bundling | Re-order **2.3 first**; bundle demo parquet in-package (<100KB) |

**Rationale for D1 split:** German Credit gives continuity with the Phase 1
benchmark (`benchmarks/results/credit-g_default.json` → 0.796 AUC) so the case
study reads as *"defaults got 0.796; after HPO + FE + calibration we hit X."*
Telco Churn (~7k rows) gives a more recognizable, bigger-feeling first touch
for `iter8 init --demo` and the live demo. Using two datasets across roles also
shows the loop generalizes.

---

## Sequencing (re-ordered; 2.3 unblocks the P0s)

| Step | Roadmap # | Pri | Task | Blocks / unblocks |
|------|-----------|-----|------|-------------------|
| 1 | 2.3 | P1 | Bundled Telco Churn + `iter8 init --demo` | Unblocks 2.1 (flow) & 2.2 (default data) |
| 2 | 2.1 | **P0** | German Credit flagship case study (Quarto → GH Pages) | The artifact reviewers read |
| 3 | 2.2 | **P0** | Gradio live demo on HF Spaces | Clickable URL in README |
| 4 | 2.4 | P1 | Trim notebooks → 3 hero docs, archive rest | Curation = signal |
| 5 | 2.5 | P2 *(stretch)* | "Design decisions" blog post | In-if-time |

---

## Task breakdown

### Step 1 — 2.3: Bundled demo dataset + `iter8 init --demo`

**Status: ✅ Implemented 2026-08-10** (pending PyPI release to surface `--demo`)

**Goal:** one command (`iter8 init --demo`) drops a Telco Churn parquet into the
workspace and prints the exact next command to run, so first-run is zero-friction.

**Steps:**
1. Add bundled dataset under `src/iter8ml/datasets/telco_churn.parquet` (~7k
   rows, <100KB). Source: Kaggle IBM Telco Churn (document license in
   `src/iter8ml/datasets/README.md`).
2. Confirm hatchling ships it (files under `src/iter8ml/` are included by
   default; verify with `uv build` + inspect wheel). Add explicit include in
   `pyproject.toml` only if the wheel is missing it.
3. Add `--demo` flag to `init` in `src/iter8ml/cli/main.py:12`:
   - `--demo` → copy bundled parquet to `workspace.data_dir / "telco_churn.parquet"`,
     echo a ready-to-paste `iter8 run --data ... --target churn` line.
4. Add a `Workspace` helper (e.g. `workspace.seed_demo_data()`) rather than
   inlining IO in the CLI — keeps it testable.
5. Smoke-test from a clean install: `uvx --from 'iter8ml[gbdt]' iter8 init --demo`.

**File refs:** `src/iter8ml/cli/main.py:12`, `src/iter8ml/workspace.py`,
`src/iter8ml/data/loader.py:30` (`load_parquet`), `pyproject.toml:116`.

**Acceptance:**
- `iter8 init --demo` works from a clean shell (no local clone) against PyPI.
- Bundled parquet present in the installed wheel.
- One unit test for the `--demo` path.

---

### Step 2 — 2.1: German Credit flagship case study

**Status: ✅ Implemented 2026-08-11** — `.qmd` authored + verified executable
end-to-end (all cells chain, real numbers). Auto-publishes via the **existing**
`.github/workflows/docs.yml` Quarto→mike pipeline (no new workflow).

**Goal:** a published, polished Quarto case study a reviewer reads in 5 minutes
and remembers. Full loop: ingest → preprocess → HPO → drift → SHAP → export,
framed as a before/after on the same dataset Phase 1 benchmarked.

**Steps:**
1. Author `notebooks/case_study_german_credit.qmd` (new hero doc). Narrative:
   - **Setup:** German Credit, 1k×20, binary classification (credit risk).
   - **Baseline:** reuse Phase 1 result — CatBoost 0.796 AUC with defaults
     (`benchmarks/results/credit-g_default.json`).
   - **Improve:** Optuna HPO (40 trials) on the champion; report new AUC +
     parameter importances (calibration curve omitted — would need an unverified
     OOF-prob API; the in-pipeline `CALIBRATION` step is noted in prose).
   - **Explain:** SHAP beeswarm + global importance bar — top risk drivers
     (`checking_status`, `credit_history`, `savings_status`).
   - **Drift:** constructed reference-vs-current split (D3) via
     `iter8 drift --reference train.parquet --new batch.parquet`; flag the
     strongest shifted feature.
   - **Export:** `iter8 export` → show the portable bundle.
2. `_quarto.yml`'s `*.qmd` glob auto-includes it (no edit needed).
3. **Publishing:** the existing `.github/workflows/docs.yml` already runs
   `quarto render notebooks/` + `scripts/generate_notebook_docs.py` +
   `mike deploy` on push to main. The case study flows through automatically;
   the script auto-generates the mkdocs stub
   (`docs/notebooks/case-study-german-credit.md`). A manual `mkdocs.yml` nav
   entry was added (no new `quarto-pages.yml` workflow needed).
4. **OMP guard:** the `.qmd` has a hidden setup cell calling
   `HardwareProfile.configure_omp_threads()` before any GBDT import — without
   it, lightgbm/xgboost deadlock in CI on hybrid-core CPUs (Phase-1 issue 1.6b).
   Verified: the notebook executes with no env vars set.
5. Teaser + link added to `README.md` right after the `credit-g` benchmark row.

**File refs:** `notebooks/` (new `.qmd`), `notebooks/_quarto.yml`,
`benchmarks/results/credit-g_default.json`, `examples/credit_risk.{py,yaml}`
(reusable config scaffold), `src/iter8ml/services/export.py:229`
(`ExportService`), `README.md`.

**Verified results (German Credit, credit-g):** baseline CatBoost **0.7913**
(LightGBM 0.769, XGBoost 0.760); HPO (40 trials) → **0.7963**; param importances
all <0.05 (lr > iter > depth > l2) → the dataset, not the knobs, is the ceiling.
Narrative framed honestly: defaults match the published 0.796 benchmark; HPO
edges to the SOTA band; the value is the integrated loop.

**Acceptance:**
- Case study URL renders on GitHub Pages and is linked from `README.md`.
- Before/after metric is stated explicitly and reproducible from the `.qmd`.
- Drift section honestly labels the split as constructed.

---

### Step 3 — 2.2: Gradio live demo on HF Spaces

**Status: ✅ Built + verified 2026-08-12 — deploy pending (manual HF Space push,
a user action requiring the HF token).**

**Goal:** a public URL where a reviewer uploads a CSV (or uses the default
Telco Churn) and gets a leaderboard + SHAP plot. Free-tier, no persisted user
data.

**Steps:**
1. New `demo/` dir: `demo/app.py` (Gradio), `demo/requirements.txt`
   (`iter8ml[gbdt]` + `gradio` + `shap`), `demo/README.md`.
2. `app.py` flow: file upload (or "use sample Telco Churn" button) → target
   column picker → `iter8 run` under the hood → render leaderboard table +
   SHAP summary for the champion model. Reuse `ExportService` /
   `Predictor` (`src/iter8ml/services/export.py:41`) for SHAP.
3. **Hardening (free-tier + anti-abuse):** cap upload size (~5MB), row cap
   (~20k), wall-clock timeout (~120s), ephemeral temp dir, no DB.
4. **Deploy (manual first):** create HF Space (SDK=gradio, CPU basic, free),
   `git push` the `demo/` contents to the Space repo (token auth).
5. **Deploy (automated, optional later):** `.github/workflows/deploy-hf.yml`
   using `HF_TOKEN` secret + `huggingface_hub.upload_folder(..., repo_type="space")`.
6. Put the live URL + screenshot in `README.md`.

**Infra prerequisites:**
- HF account + access token (**write** scope).
- Manual path: token used as git password to push the Space. No CI secret needed.
- Automated path: `HF_TOKEN` stored as a GitHub Actions secret.

**File refs:** new `demo/`, `src/iter8ml/services/export.py:41,229`,
`src/iter8ml/cli/main.py` (reuse `run` entrypoint logic), `README.md`.

**Acceptance:**
- Public Space URL returns a leaderboard for the bundled Telco Churn sample.
- Custom CSV upload works end-to-end within size/row/time caps.
- URL + screenshot in `README.md`.

**Verified locally (deploy pending):** `run_analysis()` exercised on the Telco
sample (CatBoost roc_auc=0.84, SHAP top driver `Contract`) and a synthetic
regression CSV (r2=0.96); bad-target → `ValueError`. No OMP deadlock without
env vars (lazy GBDT load). Gradio Blocks UI builds (14 children). `ruff` clean;
`demo/` excluded from `mypy` (entrypoint code, like `notebooks/`). README link
withheld until the Space is live (no dead links).

---

### Step 4 — 2.4: Trim notebooks to 3 hero docs

**Goal:** curation = signal. Keep 3 hero docs, archive the rest.

**Hero keepers (per roadmap):**
- `quick-start` ← current `01_quick_start.qmd`
- `case-study` ← new from Step 2 (`case_study_german_credit.qmd`)
- `agent-demo` ← **placeholder**, filled in Phase 3 (3.1)

**Steps:**
1. `git mv` `02`–`09` `.qmd` + their rendered `.md` under
   `docs/notebooks/archive/`.
2. Rewrite `docs/notebooks/index.md` to list only the 3 hero docs (remove the
   9-entry index currently there).
3. Update `notebooks/_quarto.yml` render list to the 3 hero `.qmd`.
4. Add a one-line pointer in `index.md` to the archive for depth-seekers.

**File refs:** `notebooks/*.qmd`, `docs/notebooks/index.md`,
`notebooks/_quarto.yml`.

**Acceptance:**
- `docs/notebooks/index.md` shows exactly 3 hero docs.
- Archived notebooks still render (links not broken).
- `_quarto.yml` builds cleanly.

---

### Step 5 — 2.5 (stretch): "Design decisions" blog post

**Goal:** show engineering judgment. Content already exists — mostly a
narrative repackaging of `ARCHITECTURE.md` + `docs/technical_roadmap.md`.

**Steps:**
1. New `docs/blog/design-decisions.md` (or a Quarto post): medallion choice,
   Hamilton DAG, hardware-aware model routing, Polars-native data layer,
   why CPU-first framing.
2. Cross-link from the case study and README.

**Acceptance:** published post linked from README. **Cut if Steps 1–4 slip.**

---

## Phase-3 reuse (forward notes)

- The Gradio demo's auto-API (D2) is the seam Phase 3.1 (agent showcase) will
  drive — design `app.py` so its core predict function is callable directly,
  not buried in a callback.
- The case study's drift section doubles as a reference for the agent demo
  narrative.

---

## Exit criteria (Phase 2 done)

1. `iter8 init --demo` works from a clean shell against PyPI.
2. Case study published on GitHub Pages; URL + teaser in `README.md`.
3. Live demo URL on HF Spaces returns a leaderboard; URL + screenshot in
   `README.md`.
4. `docs/notebooks/index.md` shows 3 hero docs; rest archived.
5. (If time) design-decisions post published.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| HF Spaces free-tier cold starts / abuse | Medium | Size + row + timeout caps; no persistence; keep-alive optional |
| Case study HPO slow on 1k rows (noisy optima) | Medium | Cap trials (50); fix seeds; report CV variance honestly |
| GitHub Pages Quarto render breaks CI | Low | `freeze: auto` already set; pin Quarto version in workflow |
| Bundled parquet licensing | Low | Telco Churn (IBM) + German Credit (OpenML/UCI) terms documented in-dataset README |
| Scope creep into Phase 3 (agent demo) | Medium | Agent-demo kept as Phase 3 placeholder only; do not start in Phase 2 |

## Verification checklist (per change)

- `uv run ruff check .` green
- `uv run mypy src/iter8ml/` green
- `uv run pytest` green (add tests for `--demo` path)
- `uv build` → inspect wheel includes bundled parquet
- Smoke-test any PyPI-facing claim with `uvx` from a clean shell
