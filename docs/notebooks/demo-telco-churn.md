---
hide:
  - navigation
  - toc
---

# Live Demo — Telco Churn in One Call

The 60-second tour of iter8ml: one call to run_analysis() on the bundled Telco Churn sample produces a cross-validated CatBoost-vs-XGBoost leaderboard and a SHAP explanation. This is the static render of the exact same core that powers the Gradio app.

<div class="iframe-container" id="iframe-wrapper-demo-telco-churn">
  <div class="iframe-controls">
    <button type="button" class="md-button notebook-expand-btn">Expand</button>
    <a href="/iter8ml/notebooks/html/demo_telco_churn.html" target="_blank" rel="noopener noreferrer" class="md-button">Open in New Tab</a>
  </div>
  <iframe src="/iter8ml/notebooks/html/demo_telco_churn.html" allowfullscreen loading="lazy"></iframe>
</div>

## Run Locally

```bash
uv run quarto preview notebooks/demo_telco_churn.qmd
```
