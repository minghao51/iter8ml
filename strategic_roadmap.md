# Technical Roadmap - Tabular Blueprint

This roadmap outlines the evolution of Tabular Blueprint from its early stage to a robust, enterprise-ready machine learning iteration framework.

## Phase 1: Foundational Strengthening & Hamilton Integration (Next 3-6 Months)
*Focus: Robustness, Type Safety, and DAG-based Data Pipelines.*

### 1.1 Hamilton-Powered Data Layer
- [x] **Hamilton Integration**: Re-architected to use Hamilton for a lite, DAG-like pipeline.
- [x] **Functional Data Components**: Preprocessing logic converted into Hamilton nodes.
- [ ] **Visual Lineage Surface**: Surface the Mermaid graph in the `state` command and experiment reports.
    - *Implementation*: Add a section to `StateObserver` to render the `pipeline_lineage` logged in `experiments.jsonl`.

### 1.2 Type Safety & Error Handling
- [ ] **Strict Typing**: Finalize remaining Mypy errors and transition to `strict = true` in `pyproject.toml`.
- [x] **Custom Exception Hierarchy**: Implemented `DataLoadError`, `ModelFitError`, and `RegistryError` with `@track_errors` decorator.

### 1.3 Quality, Benchmarking & Iteration UX
- [x] **Smart Baselines**: Automatically run "Naive" and "Linear" baselines with "Lift over Baseline" metrics.
- [x] **Configuration Diffing**: Implemented `tabblueprint diff <id1> <id2>` for side-by-side terminal comparison.
- [x] **Leakage Detection Audit**: Pre-train check via permutation importance on a naive baseline model.
- [ ] **Data Quality "Quick Fixes"**: Automatically handle noise identified by Cleanlab.
    - *Implementation*: Add a config flag `auto_clean_noise: bool` to drop or re-label rows with low label quality scores during `Trainer.run`.

## Phase 2: Advanced ML & Automated Engineering (Next 6-12 Months)
*Focus: Deepening ML capabilities and autonomous feature evolution.*

### 2.1 Automated Feature Engineering (AFE)
- [x] **Targeted Interaction Discovery**: Search for high-signal polynomial/ratio interactions among top-K features.
- [ ] **Automated Feature Selection (Pruning)**: Remove low-signal or redundant features to improve generalization.
    - *Implementation*: Implement Recursive Feature Elimination (RFE) or Null-Importance checks as a post-AFE step.
- [x] **Target Transformation**: Support for automated scaling and skewness correction (Log1p, Yeo-Johnson).

### 2.2 Deep Learning & Foundation Strategy
- [x] **TabPFN with Guardrails**: Integrated TabPFN with hardware-aware routing and row-count warnings.
- [x] **Hybrid Deep Learning**: Support for TabNet and FT-Transformer as alternatives.
- [x] **Probability Calibration**: Built-in Platt scaling and Isotonic regression via `CalibratedModel`.

### 2.3 Intelligent Observability
- [x] **Lightweight Drift Detection (Tier 1)**: Univariate feature drift via PSI.
- [x] **Domain Classifier Drift Detection (Tier 2)**: Multivariate drift via a domain classifier model.
- [x] **Explainability (SHAP)**: Automated SHAP beeswarm and importance plots integrated into the `StateObserver` report.

### 2.4 LLM Integration (MCP Plugin)
- [x] **TabularAgent Module**: MCP server providing atomic tools for agentic automation and explainability.

## Phase 3: Enterprise Readiness & Portability (12+ Months)
*Focus: High-leverage HPO and Model Portability.*

### 3.1 Jumpstart HPO Tooling
- [x] **Pre-warmed HPO**: Warm-start Optuna search spaces from historical experiment data (`experiments.jsonl`).
- [x] **PedAnova Importance**: identify local hyperparameter importance to refine search spaces.
- [ ] **Optuna Dashboard Integration**: Visualizing HPO searches and trial distributions.
    - *Implementation*: Add a `--view` flag to `tabblueprint hpo` to launch a local `optuna-dashboard` instance.

### 3.2 Model Portability & Deployment
- [ ] **Champion Export (`tabblueprint export`)**: Package the best models for use in external Python environments without infrastructure overhead.
    - *Implementation*: Bundle serialized model, Hamilton DAG functions, and a generated `predictor.py` script into a portable ZIP/directory.
- [ ] **ONNX/TorchScript Export**: Automated conversion for high-performance inference where supported.

### 3.3 Scaling & Data (Single-Node Focus)
- [ ] **Remote Data Loaders**: Native support for S3, GCS, and Snowflake via Polars.
- [ ] **Uncertainty Quantification**: Add prediction intervals to model outputs for production monitoring.
