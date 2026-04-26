"""Tests for ExperimentConfig validation."""

import pytest
from pydantic import ValidationError

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.constants import CVStrategy


def test_default_config():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    assert config.cv_folds == 5
    assert config.cv_strategy == CVStrategy.STRATIFIED
    assert config.run_hpo is False
    assert config.models == "auto"
    assert config.random_seed == 42


def test_regression_defaults():
    config = ExperimentConfig(
        name="test",
        task="regression",
        target_col="target",
        data_path="data.csv",
    )
    assert config.cv_strategy == CVStrategy.KFOLD
    assert config.metrics == ["rmse", "r2"]


def test_invalid_task():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="test",
            task="invalid_task",
            target_col="target",
            data_path="data.csv",
        )


def test_custom_metrics():
    config = ExperimentConfig(
        name="test",
        task="regression",
        target_col="target",
        data_path="data.csv",
        metrics=["rmse", "mae"],
    )
    assert config.metrics == ["rmse", "mae"]
