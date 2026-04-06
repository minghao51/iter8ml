"""
Tabular Blueprint - Quick Start Example
========================================

This notebook demonstrates the core workflow of the Tabular Blueprint framework:
1. Initialize a workspace
2. Load data
3. Run experiments with multiple models
4. Inspect results
5. Run hyperparameter optimization
"""

# %% [markdown]
# ## 1. Setup and Data Generation

import polars as pl
from sklearn.datasets import make_classification

# Generate synthetic classification data
X_cls, y_cls = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=8,
    random_state=42,
)

df_cls = pl.DataFrame({f"feat_{i}": X_cls[:, i] for i in range(X_cls.shape[1])})
df_cls = df_cls.with_columns(target=pl.Series(y_cls))

print(f"Classification dataset: {df_cls.shape}")
print(df_cls.describe())

# %% [markdown]
# ## 2. Initialize Workspace

from main import app
from typer.testing import CliRunner

runner = CliRunner()
result = runner.invoke(app, ["init"])
print(result.stdout)

# %% [markdown]
# ## 3. Run a Quick Experiment

from configs.experiment import ExperimentConfig
from core.engine.trainer import Trainer

config = ExperimentConfig(
    name="quick_start",
    task="classification",
    target_col="target",
    data_path="",
    models=["catboost", "lightgbm"],
    cv_folds=3,
    metrics=["roc_auc", "f1_macro", "accuracy"],
)

trainer = Trainer(config)
results = trainer.run(df_cls)

for model, scores in results.items():
    print(f"\n{model}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

# %% [markdown]
# ## 4. Inspect Results

# View leaderboard
result = runner.invoke(app, ["leaderboard"])
print(result.stdout)

# View experiment state
result = runner.invoke(app, ["state"])
print(result.stdout)

# %% [markdown]
# ## 5. Hyperparameter Optimization

from core.engine.hpo import optimize_model
from core.data.adapter import DataAdapter
from core.engine.evaluator import Evaluator
from core.engine.trainer import _get_model_class
from configs.model_configs import ModelConfigs

adapter = DataAdapter(target_format="numpy")
X, y = adapter.transform(df_cls, "target")

evaluator = Evaluator(task="classification", cv_folds=3)
model_cls = _get_model_class("catboost")
model_configs = ModelConfigs()
search_space = model_configs.catboost.hpo_search_space()

hpo_result = optimize_model(
    model_cls,
    X,
    y,
    evaluator,
    "catboost",
    n_trials=10,
    search_space=search_space,
    task="classification",
)

print(f"Best params: {hpo_result['best_params']}")
print(f"Best value: {hpo_result['best_value']:.4f}")

# %% [markdown]
# ## 6. Data Quality Audit

from core.data.quality import audit_data_quality

quality_report = audit_data_quality(df_cls, "target", enabled=True)
print(f"Quality audit: {quality_report['n_issues']} issues found")
print(f"Noise rate: {quality_report['noise_rate']:.2%}")
print(f"Mean quality score: {quality_report['mean_quality_score']:.4f}")

# %% [markdown]
# ## 7. Drift Detection

# Simulate a new batch with shifted distribution
X_shifted, y_shifted = make_classification(
    n_samples=500,
    n_features=15,
    n_informative=8,
    random_state=99,
)
df_shifted = pl.DataFrame({f"feat_{i}": X_shifted[:, i] + 2 for i in range(X_shifted.shape[1])})
df_shifted = df_shifted.with_columns(target=pl.Series(y_shifted))

# Save for drift detection
df_cls.write_parquet("workspace/reference.parquet")
df_shifted.write_parquet("workspace/new_batch.parquet")

from core.monitoring.drift import DriftDetector

detector = DriftDetector(df_cls)
report = detector.detect(df_shifted)

print(f"Drift detected: {report.drift_detected}")
print(f"Columns drifted: {report.n_drifted}/{report.n_columns_tested}")

for col_result in report.column_results:
    status = "DRIFT" if col_result.drift_detected else "OK"
    print(f"  {status} | {col_result.column} | p={col_result.p_value:.6f}")
