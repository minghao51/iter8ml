import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Drift Detection & Monitoring

    Compare three drift detection methods on a synthetic dataset with
    engineered distribution shift:

    - **KS / Chi-squared** — univariate per-column tests
    - **PSI** — Population Stability Index with severity classification
    - **Domain Classifier** — multivariate drift via AUC of a classifier
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Create Reference & Shifted Datasets
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from sklearn.datasets import make_classification

    np.random.seed(42)

    X_ref, y_ref = make_classification(
        n_samples=2000, n_features=10, n_informative=6, random_state=42
    )
    df_ref = pl.DataFrame({f"feat_{i}": X_ref[:, i] for i in range(X_ref.shape[1])})
    df_ref = df_ref.with_columns(target=pl.Series(y_ref))

    X_live = X_ref[:800].copy()
    X_live[:, 0] += 1.5
    X_live[:, 3] *= 1.8
    X_live[:, 7] += np.random.normal(0, 3, size=800)
    df_live = pl.DataFrame({f"feat_{i}": X_live[:, i] for i in range(X_live.shape[1])})
    df_live = df_live.with_columns(target=pl.Series(y_ref[:800]))

    f"Reference: {df_ref.shape}  |  Live: {df_live.shape}  |  Shifted columns: feat_0, feat_3, feat_7"
    return df_live, df_ref


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. KS / Chi-squared Drift Detection
    """)
    return


@app.cell
def _(df_live, df_ref):
    from tabular_blueprint.monitoring.drift import DriftDetector

    ks_detector = DriftDetector(df_ref, alpha=0.05)
    ks_report = ks_detector.detect(df_live)

    f"Drift detected: {ks_report.drift_detected}  |  Columns drifted: {ks_report.n_drifted}/{ks_report.n_columns_tested}"
    return (ks_report,)


@app.cell
def _(ks_report):
    ks_lines = []
    for c in ks_report.column_results:
        status = "DRIFT" if c.drift_detected else "  OK"
        ks_lines.append(f"{status}  |  {c.column:10s}  |  p={c.p_value:.6f}  |  test={c.test_used}")
    "\n".join(ks_lines)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. PSI (Population Stability Index)
    """)
    return


@app.cell
def _(df_live, df_ref):
    from tabular_blueprint.monitoring.psi_drift import PSIDriftDetector

    psi_detector = PSIDriftDetector(df_ref, n_bins=10)
    psi_report = psi_detector.detect(df_live)

    f"PSI drift detected: {psi_report.drift_detected}  |  Moderate: {psi_report.n_moderate}  |  Severe: {psi_report.n_severe}"
    return (psi_report,)


@app.cell
def _(psi_report):
    psi_rows = []
    for f in psi_report.feature_psi:
        psi_rows.append(f"| {f.feature:10s} | {f.psi_value:.4f} | {f.drift_level:10s} |")
    table = "| Feature | PSI | Level |\n|----------|------|--------|\n" + "\n".join(psi_rows)
    table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Domain Classifier (Multivariate Drift)
    """)
    return


@app.cell
def _(df_live, df_ref):
    from tabular_blueprint.monitoring.domain_classifier import DomainClassifierDriftDetector

    dc_detector = DomainClassifierDriftDetector(df_ref, threshold=0.7, random_seed=42)
    dc_report = dc_detector.detect(df_live)

    f"AUC: {dc_report.auc_score:.4f}  |  Threshold: {dc_report.threshold}  |  Drift: {dc_report.drift_detected}  |  Ref n={dc_report.n_reference}, Live n={dc_report.n_live}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Hamilton DAG Drift Mode (if available)
    """)
    return


@app.cell
def _(df_live, df_ref):
    from tabular_blueprint.pipelines.executor import PipelineExecutor

    executor = PipelineExecutor()
    if executor.available:
        drift_report = executor.run_drift(df_ref, df_live, drift_method="both")
        drift_report
    else:
        "Hamilton not installed — install with `uv sync --extra hamilton` to use DAG drift mode."
    return


if __name__ == "__main__":
    app.run()
