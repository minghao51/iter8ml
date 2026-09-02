# Deferred research & future work

**Status:** living digest — consolidated 2026-08-29 from the roadmap, ADRs, and
phase handoffs so deferred items have one home. Sources are cited per section;
when an item ships, move it to the changelog and delete it here.

---

## 1. Medallion completion (systems depth — priority A, demoted per B>C>A)

From [ADR-0003](../decisions/0003-medallion-artifact-contract.md) *Honest status*
and roadmap item 3.4 (`docs/plan/portfolio-roadmap-20260805.md`):

- Model-per-fold **Platinum** products.
- **OOF (out-of-fold) artifacts** for downstream stacking / calibration analysis.
- A true **DuckDB catalog** over Parquet views (replacing the local JSON/Parquet
  catalog slice; see `docs/medallion.md`).
- **Migration tooling** for artifact-contract version changes.

Payoff: a "lakehouse for ML" writeup — systems-depth showcase for the portfolio.

## 2. Live-demo hosting (phase-2 pivot)

From `docs/plan/phase2-handoff-20260812.md` §1 Step 3 + Task A:

- **HF Spaces is blocked on the free tier** (needs PRO; the Gradio app would OOM
  at 512 MB). Phase 2 pivoted to Quarto→Pages + a Colab notebook.
- The Gradio app (`demo/app.py`) and one-shot deploy script
  (`scripts/deploy_hf.py`) are **kept for Phase 3** — deploy if a PRO/paid tier
  or a smaller footprint becomes available.
- `demo/app.py`'s core `run_analysis()` must stay a callable (not buried in a
  callback) — the Gradio auto-API is the seam the Phase-3 agent showcase (3.1)
  will drive.

## 3. Phase-3 roadmap items (reference only)

From `docs/plan/portfolio-roadmap-20260805.md` → Phase 3:

- **3.1 (P0) Agent showcase** — record Claude Desktop driving the full loop via
  the MCP server (`services/mcp.py`); the `docs/notebooks/case-study-agent-demo.md`
  placeholder is filled in then.
- **3.2 (P1) External visibility** — Show HN / r/ML post timed to a release.
- **3.3 (P1) Repo surface polish** — releases notes, topic tags, issue/PR templates.
- **3.5 (P2, stretch) ONNX/TorchScript export** — makes "production-ready" real
  (`services/export.py`).

## 4. Explicitly deferred scope (honest solo/part-time capacity)

From `docs/plan/portfolio-roadmap-20260805.md` → *Explicitly deferred*:

- Uncertainty quantification.
- Optuna Dashboard integration.
- Remote data loaders (S3/GCS).
- AFE pruning via RFE / null-importance.

## 5. Housekeeping debt

From `docs/plan/phase2-handoff-20260812.md` §1 (pre-existing, uncommitted):

- Deleted-on-disk plan docs pending commit: `docs/plan/code-simplify-20260515.md`,
  `docs/plan/package-refactor-20260513.md`, `docs/plan/pipeline-spec-20260515.md`,
  `docs/technical_roadmap.md` (⚠️ referenced by old blog plans — use
  `ARCHITECTURE.md` as source material instead).
