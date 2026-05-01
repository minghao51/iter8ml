"""Tests for ExperimentConfig validation."""

from pathlib import Path

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


def test_from_file_blocks_python_config_by_default(tmp_path: Path):
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "from tabular_blueprint.config import ExperimentConfig\n"
        "from tabular_blueprint.constants import TaskType\n"
        "config = ExperimentConfig("
        "name='test', task=TaskType.CLASSIFICATION, target_col='target', data_path='x.csv')\n"
    )

    with pytest.raises(ValueError, match="disabled by default for safety"):
        ExperimentConfig.from_file(config_path)


def test_from_file_allows_python_config_with_opt_in(tmp_path: Path):
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "from tabular_blueprint.config import ExperimentConfig\n"
        "from tabular_blueprint.constants import TaskType\n"
        "config = ExperimentConfig("
        "name='test', task=TaskType.CLASSIFICATION, target_col='target', data_path='x.csv')\n"
    )
    config = ExperimentConfig.from_file(config_path, allow_unsafe_python=True)
    assert config.name == "test"
