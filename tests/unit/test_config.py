"""Tests for ExperimentConfig validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from iter8ml.config import (
    DEFAULT_LLM_MODEL,
    ExperimentConfig,
    HardwareProfile,
    PipelineSpec,
    PipelineStep,
    StepName,
)
from iter8ml.constants import CVStrategy, EmbeddingMethod, TaskType


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
    assert config.afe_n_jobs == 1
    assert config.afe_max_candidate_pairs == 200
    assert config.leakage_n_jobs == 1


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
        "from iter8ml.config import ExperimentConfig\n"
        "from iter8ml.constants import TaskType\n"
        "config = ExperimentConfig("
        "name='test', task=TaskType.CLASSIFICATION, target_col='target', data_path='x.csv')\n"
    )

    with pytest.raises(ValueError, match="disabled by default for safety"):
        ExperimentConfig.from_file(config_path)


def test_from_file_allows_python_config_with_opt_in(tmp_path: Path):
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "from iter8ml.config import ExperimentConfig\n"
        "from iter8ml.constants import TaskType\n"
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
    with patch.dict(os.environ, {"ITER8ML_LLM_MODEL": "gpt-4o"}):
        config = ExperimentConfig(
            name="test",
            task="classification",
            target_col="target",
            data_path="data.csv",
        )
        assert config.llm_model == "gpt-4o"


def test_llm_model_explicit_value_overrides_env():
    with patch.dict(os.environ, {"ITER8ML_LLM_MODEL": "gpt-4o"}):
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


def test_strict_thread_safety_default_true():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    assert config.strict_thread_safety is True


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


def test_legacy_flat_keys_from_example_config() -> None:
    """The shipped credit_risk.yaml uses legacy flat keys; loading it must
    resolve them onto the nested model (the supported compat layer)."""
    config_path = Path(__file__).resolve().parents[2] / "examples" / "credit_risk.yaml"
    config = ExperimentConfig.from_file(config_path)

    # Flat delegate keys → nested sub-configs.
    assert config.hpo.run is False
    assert config.hpo.n_trials == 100

    # Legacy step-level keys → pipeline step enablement/params.
    quality_step = next(
        s for s in config.pipeline.steps if s.name == StepName.QUALITY_AUDIT
    )
    assert quality_step.enabled is True


# --- HardwareProfile tests ---


def test_hardware_profile_detect_without_torch(monkeypatch):
    import torch as real_torch

    monkeypatch.setattr(real_torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "iter8ml.config.psutil.virtual_memory", lambda: type("mem", (), {"total": 8e9})()
    )
    monkeypatch.setattr("iter8ml.config.psutil.cpu_count", lambda logical=False: 4)
    hp = HardwareProfile.detect()
    assert hp.has_gpu is False
    assert hp.gpu_name is None
    assert hp.vram_gb == 0.0


def test_hardware_profile_detect_with_torch_no_cuda(monkeypatch):
    import torch as real_torch

    monkeypatch.setattr(real_torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "iter8ml.config.psutil.virtual_memory", lambda: type("mem", (), {"total": 16e9})()
    )
    monkeypatch.setattr("iter8ml.config.psutil.cpu_count", lambda logical=False: 8)
    hp = HardwareProfile.detect()
    assert hp.has_gpu is False
    assert hp.system_ram_gb > 0
    assert hp.cpu_cores == 8


def test_hardware_profile_configure_omp_threads_default(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    count = HardwareProfile.configure_omp_threads()
    assert isinstance(count, int)
    assert count > 0


def test_hardware_profile_configure_omp_threads_custom():
    count = HardwareProfile.configure_omp_threads(threads=8)
    assert count == 8
    import os

    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_hardware_profile_configure_omp_threads_reentry(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    count = HardwareProfile.configure_omp_threads()
    assert count == 4


# --- Flat delegate accessors ---


def test_getattr_flat_delegates():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
        embedding_method="autoencoder",
    )
    assert config.embedding_method == EmbeddingMethod.AUTOENCODER


def test_setattr_flat_delegates():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    config.embedding_method = "entity"
    assert config.embedding.method == "entity"


def test_getattr_unknown_raises():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    with pytest.raises(AttributeError, match="no attribute"):
        _ = config.nonexistent_attr


# --- from_file edge cases ---


def test_from_file_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: test\ntask: classification\ntarget_col: target\ndata_path: data.csv\n"
    )
    config = ExperimentConfig.from_file(config_path)
    assert config.name == "test"
    assert config.task == TaskType.CLASSIFICATION


def test_from_file_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"name": "test", "task": "classification",'
        ' "target_col": "target", "data_path": "data.csv"}'
    )
    config = ExperimentConfig.from_file(config_path)
    assert config.name == "test"


def test_from_file_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'name = "test"\ntask = "classification"\ntarget_col = "target"\ndata_path = "data.csv"\n'
    )
    config = ExperimentConfig.from_file(config_path)
    assert config.name == "test"


def test_from_file_unsupported_suffix(tmp_path):
    config_path = tmp_path / "config.ini"
    config_path.write_text("[config]\nname=test\n")
    with pytest.raises(ValueError, match="Unsupported config format"):
        ExperimentConfig.from_file(config_path)


def test_from_file_nonexistent(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        ExperimentConfig.from_file(tmp_path / "missing.yaml")


# --- Nested flat key normalization ---


def test_nest_flat_config_fields():
    data = {
        "name": "test",
        "task": "classification",
        "target_col": "target",
        "data_path": "data.csv",
        "run_hpo": True,
        "hpo_n_trials": 100,
    }
    config = ExperimentConfig.model_validate(data)
    assert config.hpo.run is True
    assert config.hpo.n_trials == 100


def test_legacy_pipeline_flat_keys_are_migrated():
    data = {
        "name": "test",
        "task": "classification",
        "target_col": "target",
        "data_path": "data.csv",
        "run_quality_audit": False,
        "auto_clean_noise": True,
        "noise_quality_threshold": 0.7,
        "run_leakage_audit": False,
        "target_transform": "log1p",
        "target_skewness_threshold": 2.0,
        "feature_strategy": "afe",
        "calibration": "platt",
    }
    config = ExperimentConfig.model_validate(data)

    assert config.pipeline.is_enabled(StepName.QUALITY_AUDIT) is False
    assert config.pipeline.step_params(StepName.QUALITY_AUDIT) == {
        "auto_clean_noise": True,
        "noise_quality_threshold": 0.7,
    }
    assert config.pipeline.is_enabled(StepName.LEAKAGE_AUDIT) is False
    assert config.pipeline.step_params(StepName.TARGET_TRANSFORM) == {
        "method": "log1p",
        "skewness_threshold": 2.0,
    }
    assert config.pipeline.step_params(StepName.FEATURE_ENGINEERING) == {"strategy": "afe"}
    assert config.pipeline.step_params(StepName.CALIBRATION) == {"method": "platt"}


def test_legacy_keys_do_not_override_explicit_pipeline_steps():
    data = {
        "name": "test",
        "task": "classification",
        "target_col": "target",
        "data_path": "data.csv",
        "run_quality_audit": False,
        "pipeline": {
            "steps": [
                {
                    "name": "quality_audit",
                    "enabled": True,
                    "params": {"auto_clean_noise": False},
                }
            ]
        },
    }
    config = ExperimentConfig.model_validate(data)
    assert config.pipeline.is_enabled(StepName.QUALITY_AUDIT) is False
    assert config.pipeline.step_params(StepName.QUALITY_AUDIT) == {"auto_clean_noise": False}


# --- PipelineSpec ---


def test_pipeline_spec_defaults():
    spec = PipelineSpec()
    assert len(spec.steps) == 8
    assert all(s.enabled for s in spec.steps)
    assert spec.is_enabled(StepName.DATA_PREP)
    assert spec.is_enabled(StepName.MODEL_TRAINING)


def test_pipeline_spec_disabled_step():
    spec = PipelineSpec(
        steps=[
            PipelineStep(name=StepName.DATA_PREP),
            PipelineStep(name=StepName.QUALITY_AUDIT, enabled=False),
            PipelineStep(name=StepName.LEAKAGE_AUDIT),
        ]
    )
    assert spec.is_enabled(StepName.DATA_PREP) is True
    assert spec.is_enabled(StepName.QUALITY_AUDIT) is False
    assert spec.is_enabled(StepName.CALIBRATION) is False


def test_pipeline_spec_step_params():
    spec = PipelineSpec(
        steps=[
            PipelineStep(
                name=StepName.TARGET_TRANSFORM,
                params={"method": "log1p", "skewness_threshold": 2.0},
            ),
        ]
    )
    assert spec.step_params(StepName.TARGET_TRANSFORM) == {
        "method": "log1p",
        "skewness_threshold": 2.0,
    }
    assert spec.step_params(StepName.CALIBRATION) == {}


def test_legacy_flat_keys_match_explicit_pipeline_resolution():
    from iter8ml.engine.pipelines.executor import _resolve_hamilton_config

    legacy_cfg = ExperimentConfig.model_validate(
        {
            "name": "legacy",
            "task": "classification",
            "target_col": "target",
            "data_path": "data.csv",
            "run_quality_audit": False,
            "auto_clean_noise": True,
            "noise_quality_threshold": 0.6,
            "run_leakage_audit": False,
            "target_transform": "auto",
            "target_skewness_threshold": 2.0,
            "feature_strategy": "afe",
            "calibration": "isotonic",
        }
    )
    step_cfg = ExperimentConfig.model_validate(
        {
            "name": "step",
            "task": "classification",
            "target_col": "target",
            "data_path": "data.csv",
            "pipeline": {
                "steps": [
                    {
                        "name": "quality_audit",
                        "enabled": False,
                        "params": {"auto_clean_noise": True, "noise_quality_threshold": 0.6},
                    },
                    {"name": "leakage_audit", "enabled": False},
                    {
                        "name": "target_transform",
                        "params": {"method": "auto", "skewness_threshold": 2.0},
                    },
                    {"name": "feature_engineering", "params": {"strategy": "afe"}},
                    {"name": "calibration", "params": {"method": "isotonic"}},
                ]
            },
        }
    )
    assert _resolve_hamilton_config(legacy_cfg) == _resolve_hamilton_config(step_cfg)


def test_pipeline_spec_from_yaml():
    import yaml

    yaml_str = """
    name: test
    task: classification
    target_col: target
    data_path: data.csv
    pipeline:
      steps:
        - name: data_prep
        - name: quality_audit
          enabled: false
        - name: leakage_audit
        - name: target_transform
          params:
            method: auto
        - name: feature_engineering
          params:
            strategy: afe
        - name: model_training
        - name: calibration
        - name: evaluation
    """
    data = yaml.safe_load(yaml_str)
    config = ExperimentConfig.model_validate(data)
    assert config.pipeline.is_enabled(StepName.QUALITY_AUDIT) is False
    assert config.pipeline.step_params(StepName.TARGET_TRANSFORM) == {"method": "auto"}
    assert config.pipeline.step_params(StepName.FEATURE_ENGINEERING) == {"strategy": "afe"}


def test_describe_pipeline():
    from iter8ml.engine.pipelines.executor import PipelineExecutor

    spec = PipelineSpec(
        steps=[
            PipelineStep(name=StepName.DATA_PREP),
            PipelineStep(name=StepName.QUALITY_AUDIT, enabled=False),
            PipelineStep(name=StepName.MODEL_TRAINING),
        ]
    )
    result = PipelineExecutor().describe_pipeline(spec)
    assert len(result) == 3
    assert result[0] == {"step": "data_prep", "enabled": True, "params": {}}
    assert result[1] == {"step": "quality_audit", "enabled": False, "params": {}}
    assert result[2]["step"] == "model_training"


def test_mermaid_annotates_disabled():
    from iter8ml.engine.pipelines.executor import PipelineExecutor

    spec = PipelineSpec(
        steps=[
            PipelineStep(name=StepName.DATA_PREP),
            PipelineStep(name=StepName.QUALITY_AUDIT, enabled=False),
            PipelineStep(name=StepName.MODEL_TRAINING),
        ]
    )
    graph = PipelineExecutor().get_mermaid_graph(spec=spec)
    assert "quality_audit ~disabled~" in graph
    assert "classDef disabled" in graph
    assert "step_0 --> step_1" in graph
    assert "step_1 --> step_2" in graph


def test_mermaid_no_disabled_class_when_all_enabled():
    from iter8ml.engine.pipelines.executor import PipelineExecutor

    spec = PipelineSpec(
        steps=[
            PipelineStep(name=StepName.DATA_PREP),
            PipelineStep(name=StepName.MODEL_TRAINING),
        ]
    )
    graph = PipelineExecutor().get_mermaid_graph(spec=spec)
    assert "classDef disabled" not in graph
    assert "~disabled~" not in graph


# --- Enum serialization ---


def test_serialize_enum():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    dumped = config.model_dump(mode="json")
    assert dumped["task"] == "classification"
    assert dumped["cv_strategy"] == "stratified"


def test_serialize_non_enum_value():
    config = ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )
    result = config.serialize_enum(42)
    assert result == "42"
