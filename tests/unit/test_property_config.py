"""Property-based tests for ExperimentConfig validation invariants."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy, TaskType

pytestmark = pytest.mark.property


class TestPropertyConfig:
    """Property: ExperimentConfig invariants hold across valid parameter space."""

    @settings(max_examples=50)
    @given(
        name=st.text(min_size=1, max_size=20),
        task=st.sampled_from(["classification", "regression"]),
        target_col=st.text(min_size=1, max_size=10),
        data_path=st.text(min_size=1, max_size=50),
        cv_folds=st.integers(2, 10),
        random_seed=st.integers(0, 1000),
        data_sample=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
    )
    def test_config_always_has_required_fields(
        self, name, task, target_col, data_path, cv_folds, random_seed, data_sample
    ):
        config = ExperimentConfig(
            name=name,
            task=TaskType(task),
            target_col=target_col,
            data_path=data_path,
            cv_folds=cv_folds,
            random_seed=random_seed,
            data_sample=data_sample,
        )
        assert config.name == name
        assert config.task.value == task
        assert config.target_col == target_col
        assert config.data_path == data_path
        assert config.cv_folds == cv_folds
        assert config.random_seed == random_seed
        assert config.data_sample == data_sample
        assert config.metrics is not None
        assert len(config.metrics) > 0

    @settings(max_examples=50)
    @given(
        name=st.text(min_size=1, max_size=20),
        target_col=st.text(min_size=1, max_size=10),
        data_path=st.text(min_size=1, max_size=50),
        cv_folds=st.integers(2, 5),
    )
    def test_regression_defaults(self, name, target_col, data_path, cv_folds):
        config = ExperimentConfig(
            name=name,
            task=TaskType.REGRESSION,
            target_col=target_col,
            data_path=data_path,
            cv_folds=cv_folds,
        )
        assert config.cv_strategy == CVStrategy.KFOLD

    @settings(max_examples=50)
    @given(
        name=st.text(min_size=1, max_size=20),
        target_col=st.text(min_size=1, max_size=10),
        data_path=st.text(min_size=1, max_size=50),
        cv_folds=st.integers(2, 5),
    )
    def test_classification_defaults(self, name, target_col, data_path, cv_folds):
        config = ExperimentConfig(
            name=name,
            task=TaskType.CLASSIFICATION,
            target_col=target_col,
            data_path=data_path,
            cv_folds=cv_folds,
        )
        assert config.cv_strategy == CVStrategy.STRATIFIED

    @settings(max_examples=30)
    @given(
        name=st.text(min_size=1, max_size=20),
        target_col=st.text(min_size=1, max_size=10),
        data_path=st.text(min_size=1, max_size=50),
    )
    def test_config_serializable_to_dict(self, name, target_col, data_path):
        config = ExperimentConfig(
            name=name,
            task=TaskType.CLASSIFICATION,
            target_col=target_col,
            data_path=data_path,
        )
        d = config.model_dump(mode="json")
        assert isinstance(d, dict)
        assert d["name"] == name
        assert d["target_col"] == target_col
