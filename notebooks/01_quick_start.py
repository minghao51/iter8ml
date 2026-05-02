import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quick Start: End-to-End Classification Experiment

    This notebook walks through the core **tabular-blueprint** workflow:

    1. Generate (or load) a dataset
    2. Configure an experiment
    3. Run multi-model training with cross-validation
    4. Inspect the leaderboard and registry
    5. Export the champion model
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Synthetic Data
    """)
    return


@app.cell
def _(mo):
    import polars as pl
    from sklearn.datasets import make_classification

    @mo.persistent_cache
    def generate_data():
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            n_redundant=3,
            random_state=42,
        )
        df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
        df = df.with_columns(target=pl.Series(y))
        return df

    df = generate_data()
    f"Dataset shape: {df.shape}  |  Target balance: {df['target'].mean():.2%} positive class"
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Configure & Run Experiment
    """)
    return


@app.cell
def _():
    from tabular_blueprint.config import ExperimentConfig

    config = ExperimentConfig(
        name="quickstart_demo",
        task="classification",
        target_col="target",
        data_path="",
        models=["catboost", "lightgbm", "xgboost"],
        cv_folds=5,
        metrics=["roc_auc", "f1_macro", "accuracy"],
        shap_enabled=True,
        max_workers=2,
    )
    config
    return (config,)


@app.cell
def _(config, df, mo):
    from tabular_blueprint.engine.trainer import Trainer

    @mo.persistent_cache
    def run_experiment():
        trainer = Trainer(config)
        return trainer.run(df)

    results = run_experiment()
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inspect the Leaderboard
    """)
    return


@app.cell
def _():
    from tabular_blueprint.services.report_service import ReportService

    report_svc = ReportService()
    console_output = report_svc.format_leaderboard_console(limit=10)
    print(console_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Export the Champion Model
    """)
    return


@app.cell
def _():
    from tabular_blueprint.services.export_service import ExportService

    exporter = ExportService()
    try:
        export_path = exporter.export(
            key="quickstart_demo:classification",
            target_col="target",
        )
        f"Model exported to: {export_path}"
    except (ValueError, FileNotFoundError) as e:
        f"Export skipped: {e}"
    return


if __name__ == "__main__":
    app.run()
