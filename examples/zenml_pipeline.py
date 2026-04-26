"""ZenML pipeline example — shows how core/ functions drop into ZenML with no changes.

ZenML is not a dependency. This example demonstrates that the plain Python
functions in core/ are already ZenML-compatible as steps.
"""

# from zenml import pipeline, step
#
# import polars as pl
#
# from tabular_blueprint.config import ExperimentConfig
# from tabular_blueprint.data.loaders import load_parquet
# from tabular_blueprint.engine.trainer import Trainer
#
#
# @step
# def load_data(path: str) -> pl.DataFrame:
#     return load_parquet(path)
#
#
# @step
# def train_models(df: pl.DataFrame, config: ExperimentConfig) -> dict:
#     trainer = Trainer(config)
#     return trainer.run(df)
#
#
# @step
# def evaluate_results(results: dict) -> str:
#     best_model = max(results, key=lambda m: results[m].get("roc_auc", 0))
#     return f"Best model: {best_model}"
#
#
# @pipeline
# def retraining_pipeline(data_path: str):
#     df = load_data(data_path)
#     config = ExperimentConfig(
#         name="zenml_run",
#         task="classification",
#         target_col="target",
#         data_path=data_path,
#     )
#     results = train_models(df, config)
#     evaluate_results(results)
