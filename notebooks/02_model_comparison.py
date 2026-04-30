import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Comparison & Evaluation

    Compare multiple models side-by-side using the **Evaluator** API directly,
    with custom cross-validation strategies and per-fold metric breakdowns.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Classification Data
    """)
    return


@app.cell
def _():
    import polars as pl
    from sklearn.datasets import make_classification
    from tabular_blueprint.data.adapter import DataAdapter

    X_raw, y_raw = make_classification(
        n_samples=3000, n_features=25, n_informative=12, random_state=42
    )
    df_cls = pl.DataFrame({f"col_{i}": X_raw[:, i] for i in range(X_raw.shape[1])})
    df_cls = df_cls.with_columns(label=pl.Series(y_raw))

    adapter = DataAdapter(target_format="numpy")
    X, y = adapter.transform(df_cls, "label")

    f"X shape: {X.shape}, y shape: {y.shape}, positive rate: {y.mean():.2%}"
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Configure Evaluator
    """)
    return


@app.cell
def _():
    from tabular_blueprint.config import ExperimentConfig
    from tabular_blueprint.engine.evaluator import Evaluator

    eval_config = ExperimentConfig(
        name="compare",
        task="classification",
        target_col="label",
        data_path="",
        cv_folds=5,
        cv_strategy="stratified",
        metrics=["roc_auc", "f1_macro", "accuracy", "log_loss"],
    )
    evaluator = Evaluator(eval_config)

    f"Task: {evaluator.task}  |  Folds: {evaluator.cv_folds}  |  Strategy: {evaluator.cv_strategy}"
    return Evaluator, evaluator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Evaluate Each Model
    """)
    return


@app.cell
def _(X, evaluator, y):
    from tabular_blueprint.models.factory import get_model_class

    models_to_compare = ["catboost", "lightgbm", "xgboost", "naive_baseline", "linear_baseline"]
    scores = {}

    for model_name in models_to_compare:
        model_cls = get_model_class(model_name)
        scores[model_name] = evaluator.evaluate(model_cls, X, y)

    scores
    return (scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Leaderboard with Lift Over Baseline
    """)
    return


@app.cell
def _(Evaluator, scores):
    baseline = scores["naive_baseline"]

    leaderboard_rows = []
    for model_nm, s in scores.items():
        lift = Evaluator.compute_lift(s, baseline, "roc_auc")
        leaderboard_rows.append(
            f"| {model_nm:20s} | {s['roc_auc']:.4f} | {s['f1_macro']:.4f} | {s['accuracy']:.4f} | {lift:+.2%} |"
        )

    leaderboard_table = (
        "| Model | ROC AUC | F1 Macro | Accuracy | Lift over Naive |\n"
        "|-------|---------|----------|----------|----------------|\n" + "\n".join(leaderboard_rows)
    )
    leaderboard_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Regression Comparison
    """)
    return


@app.cell
def _():
    from sklearn.datasets import make_regression

    from tabular_blueprint.data.adapter import DataAdapter as DA2
    from tabular_blueprint.engine.evaluator import Evaluator as Eval2
    from tabular_blueprint.config import ExperimentConfig as Cfg2
    from tabular_blueprint.models.factory import get_model_class as GMC2

    X_reg_raw, y_reg_raw = make_regression(n_samples=2000, n_features=15, noise=10, random_state=42)

    import polars as pl2

    df_reg = pl2.DataFrame({f"x_{i}": X_reg_raw[:, i] for i in range(X_reg_raw.shape[1])})
    df_reg = df_reg.with_columns(target=pl2.Series(y_reg_raw))

    reg_adapter = DA2(target_format="numpy")
    X_r, y_r = reg_adapter.transform(df_reg, "target")

    reg_config = Cfg2(
        name="reg_compare",
        task="regression",
        target_col="target",
        data_path="",
        cv_folds=5,
        metrics=["rmse", "mae", "r2"],
    )
    reg_eval = Eval2(reg_config)

    reg_scores = {}
    for rnm in ["catboost", "lightgbm", "xgboost", "linear_baseline"]:
        rcl = GMC2(rnm)
        reg_scores[rnm] = reg_eval.evaluate(rcl, X_r, y_r)

    reg_scores
    return


if __name__ == "__main__":
    app.run()
