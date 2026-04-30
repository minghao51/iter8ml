import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameter Optimization with Optuna

    Use the **HPO** module to find optimal hyperparameters for any model.
    Features demonstrated:

    - Warmstarting from previous experiment logs
    - Per-model search spaces
    - Optuna study creation with pruners
    - Param importance tracking
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return


@app.cell
def _():
    import polars as pl
    from sklearn.datasets import make_classification
    from tabular_blueprint.data.adapter import DataAdapter

    X_raw, y_raw = make_classification(
        n_samples=3000, n_features=20, n_informative=12, random_state=42
    )
    df = pl.DataFrame({f"f_{i}": X_raw[:, i] for i in range(X_raw.shape[1])})
    df = df.with_columns(target=pl.Series(y_raw))

    adapter = DataAdapter(target_format="numpy")
    X, y = adapter.transform(df, "target")

    f"X: {X.shape}, y: {y.shape}"
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Create Optuna Study
    """)
    return


@app.cell
def _():
    from tabular_blueprint.engine.hpo import create_study

    study = create_study(
        model_name="catboost",
        direction="maximize",
        n_trials=30,
        pruner="hyperband",
    )
    f"Study: {study.study_name}  |  Direction: {study.direction}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Run HPO for CatBoost
    """)
    return


@app.cell
def _(X, y):
    from tabular_blueprint.engine.hpo import optimize_model as om1
    from tabular_blueprint.engine.evaluator import Evaluator as Ev1
    from tabular_blueprint.config import ExperimentConfig as Ec1
    from tabular_blueprint.models.factory import get_model_class as Gc1

    eval_config = Ec1(
        name="hpo_nb",
        task="classification",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc"],
    )
    evaluator = Ev1(eval_config)
    model_cls = Gc1("catboost")

    hpo_result = om1(
        model_cls,
        X,
        y,
        evaluator,
        "catboost",
        n_trials=20,
        task="classification",
    )

    f"Best ROC AUC: {hpo_result['best_value']:.4f}"
    return (hpo_result,)


@app.cell
def _(hpo_result):
    params = hpo_result["best_params"]
    lines = [f"  {k}: {v}" for k, v in params.items()]
    "Best params:\n" + "\n".join(lines)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Run HPO for LightGBM
    """)
    return


@app.cell
def _(X, y):
    from tabular_blueprint.engine.hpo import optimize_model as om2
    from tabular_blueprint.engine.evaluator import Evaluator as Ev2
    from tabular_blueprint.config import ExperimentConfig as Ec2
    from tabular_blueprint.models.factory import get_model_class as Gc2

    eval_cfg2 = Ec2(
        name="hpo_lgbm",
        task="classification",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc"],
    )
    ev2 = Ev2(eval_cfg2)
    lgbm_cls = Gc2("lightgbm")

    lgbm_result = om2(
        lgbm_cls,
        X,
        y,
        ev2,
        "lightgbm",
        n_trials=20,
        task="classification",
    )

    f"LightGBM Best ROC AUC: {lgbm_result['best_value']:.4f}"
    return (lgbm_result,)


@app.cell
def _(hpo_result, lgbm_result):
    cb = hpo_result["best_value"]
    lb = lgbm_result["best_value"]
    winner = "CatBoost" if cb >= lb else "LightGBM"
    f"| CatBoost: {cb:.4f} | LightGBM: {lb:.4f} | Winner: {winner} |"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Warmstart from Previous Runs
    """)
    return


@app.cell
def _(X, y):
    import tempfile
    from pathlib import Path

    from tabular_blueprint.engine.hpo import optimize_model as om3
    from tabular_blueprint.engine.evaluator import Evaluator as Ev3
    from tabular_blueprint.config import ExperimentConfig as Ec3
    from tabular_blueprint.models.factory import get_model_class as Gc3

    log_path = Path(tempfile.mkdtemp()) / "hpo_warmstart.jsonl"

    eval_cfg3 = Ec3(
        name="warm",
        task="classification",
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc"],
    )
    ev3 = Ev3(eval_cfg3)
    xgb_cls = Gc3("xgboost")

    first_pass = om3(
        xgb_cls,
        X,
        y,
        ev3,
        "xgboost",
        n_trials=10,
        task="classification",
        log_path=str(log_path),
    )

    warmstarted = om3(
        xgb_cls,
        X,
        y,
        ev3,
        "xgboost",
        n_trials=10,
        task="classification",
        log_path=str(log_path),
    )

    f"First pass: {first_pass['best_value']:.4f}  ->  Warmstarted: {warmstarted['best_value']:.4f}  |  Warmstart trials: {warmstarted.get('warmstart_trials', 0)}"
    return


if __name__ == "__main__":
    app.run()
