"""Tests for ExperimentConfig validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tabular_blueprint.config import DEFAULT_LLM_MODEL, ExperimentConfig
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


# --- Phase D tests ---


def test_llm_model_default_without_env():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    assert config.llm_model == DEFAULT_LLM_MODEL


def test_llm_model_env_var_override():
    with patch.dict(os.environ, {"TABBLUEPRINT_LLM_MODEL": "gpt-4o"}):
        config = ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
        )
        assert config.llm_model == "gpt-4o"


def test_llm_model_explicit_value_overrides_env():
    with patch.dict(os.environ, {"TABBLUEPRINT_LLM_MODEL": "gpt-4o"}):
        config = ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            llm_model="claude-3-opus",
        )
        assert config.llm_model == "claude-3-opus"


def test_data_sample_validation_rejects_zero():
    with pytest.raises(ValidationError, match="data_sample"):
        ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            data_sample=0.0,
        )


def test_data_sample_validation_rejects_negative():
    with pytest.raises(ValidationError, match="data_sample"):
        ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            data_sample=-0.5,
        )


def test_data_sample_validation_rejects_over_one():
    with pytest.raises(ValidationError, match="data_sample"):
        ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            data_sample=1.5,
        )


def test_data_sample_accepts_valid():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
        data_sample=0.5,
    )
    assert config.data_sample == 0.5


def test_hpo_n_trials_must_be_positive():
    with pytest.raises(ValidationError, match="hpo_n_trials"):
        ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            run_hpo=True,
            hpo_n_trials=0,
        )


def test_hpo_n_trials_negative_rejected():
    with pytest.raises(ValidationError, match="hpo_n_trials"):
        ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            run_hpo=True,
            hpo_n_trials=-5,
        )


def test_hpo_n_trials_zero_ok_when_hpo_disabled():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
        run_hpo=False,
        hpo_n_trials=0,
    )
    assert config.hpo_n_trials == 0


def test_invalid_model_name_rejected():
    with pytest.raises(ValidationError, match="Unknown model names"):
        ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            models=["catboost", "nonexistent_model"],
        )


def test_valid_model_names_accepted():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
        models=["catboost", "lightgbm"],
    )
    assert config.models == ["catboost", "lightgbm"]


def test_model_overrides_default_none():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    assert config.model_overrides is None


def test_model_overrides_set():
    overrides = {"catboost": {"depth": 8}, "lightgbm": {"num_leaves": 63}}
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
        model_overrides=overrides,
    )
    assert config.model_overrides == overrides


def test_model_overrides_unknown_model_rejected():
    with pytest.raises(ValidationError, match="Unknown model_overrides keys"):
        ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
            model_overrides={"not_a_model": {"depth": 8}},
        )


def test_section_comments_in_config():
    source = ExperimentConfig.__doc__
    assert source is not None
    assert "Core" in source
    assert "HPO" in source
    assert "Embedding" in source
    assert "LLM" in source
    assert "Model Overrides" in source
