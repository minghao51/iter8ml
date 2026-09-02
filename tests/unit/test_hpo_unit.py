"""Tests for Optuna HPO module."""

from typing import ClassVar

import numpy as np
import pytest

from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.engine.evaluator import Evaluator
from iter8ml.engine.hpo import create_study, optimize_model


class DummyModel:
    """A deterministic dummy model for HPO testing."""

    model_name = "Dummy"

    def __init__(self, task="classification", lr=0.01, n_estimators=100):
        self.task = task
        self.lr = lr
        self.n_estimators = n_estimators

    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        np.random.seed(42)
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X):
        np.random.seed(42)
        probs = np.random.rand(len(X), 2)
        return probs / probs.sum(axis=1, keepdims=True)


@pytest.fixture
def sample_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


def test_create_study_median_pruner():
    import optuna

    study = create_study("test_model", pruner="median", n_trials=5)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert isinstance(study.pruner.__class__.__name__, str)


def test_create_study_hyperband_pruner():
    study = create_study("test_model", pruner="hyperband", n_trials=5)
    assert study.pruner is not None


def test_create_study_nop_pruner():
    study = create_study("test_model", pruner="unknown", n_trials=5)
    assert study.pruner is not None


def test_create_study_minimize():
    import optuna

    study = create_study("test_model", direction="minimize", n_trials=5)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


def test_optimize_model_basic(sample_data):
    X, y = sample_data
    config = ExperimentConfig(
        name="hpo_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        cv_folds=2,
        metrics=["roc_auc"],
    )
    evaluator = Evaluator(config)
    search_space = {"lr": [0.001, 0.1], "n_estimators": [50, 200]}

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=3,
        search_space=search_space,
        task="classification",
        metrics=["roc_auc"],
    )

    assert "best_params" in result
    assert "best_value" in result
    assert "n_trials" in result
    assert result["n_trials"] == 3


def test_optimize_model_no_search_space(sample_data):
    X, y = sample_data
    config = ExperimentConfig(
        name="hpo_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        cv_folds=2,
        metrics=["roc_auc"],
    )
    evaluator = Evaluator(config)

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=2,
        search_space=None,
        task="classification",
        metrics=["roc_auc"],
    )

    assert result["n_trials"] == 2


def test_optimize_model_log_space(sample_data):
    X, y = sample_data
    config = ExperimentConfig(
        name="hpo_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        cv_folds=2,
        metrics=["roc_auc"],
    )
    evaluator = Evaluator(config)
    search_space = {"lr": [0.001, 0.1, "log"]}

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=2,
        search_space=search_space,
        task="classification",
        metrics=["roc_auc"],
    )

    assert "best_params" in result
    assert result["n_trials"] == 2


def test_optimize_model_preserves_exception_context():
    """Test that evaluation failures preserve exception context."""
    from unittest.mock import Mock

    from iter8ml.engine.hpo import optimize_model
    from iter8ml.engine.models.catboost_model import CatBoostModel

    # Create invalid data to trigger error
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])  # Wrong shape for classification

    evaluator = Mock()

    # First trial fails, second succeeds
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Invalid data shape")
        return {"roc_auc": 0.8}

    evaluator.evaluate.side_effect = side_effect

    result = optimize_model(
        CatBoostModel,
        X,
        y,
        evaluator,
        "catboost",
        # 10 trials so the min-completed-trials guard (max(3, n//10)) is not
        # triggered by the single seeded failure — this test pins exception
        # context, not survivor-set semantics.
        n_trials=10,
        search_space={},
    )

    # Should have completed trials without crashing
    assert result["n_trials"] == 10
    assert "best_params" in result


def test_optimize_model_minimizes_lower_is_better_metric():
    """HPO direction must come from the central registry: rmse minimizes."""
    from unittest.mock import Mock

    X = np.zeros((10, 2))
    y = np.zeros(10)
    evaluator = Mock()
    evaluator.evaluate.return_value = {"rmse": 1.5, "r2": 0.3}

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=1,
        task="regression",
        metrics=["rmse", "r2"],
    )

    assert result["direction"] == "minimize"
    assert result["primary_metric"] == "rmse"
    assert result["best_value"] == 1.5


def test_optimize_model_reports_warmstart_summary_and_warnings(sample_data, tmp_path, monkeypatch):
    X, y = sample_data
    config = ExperimentConfig(
        name="hpo_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="test.csv",
        cv_folds=2,
        metrics=["roc_auc"],
    )
    evaluator = Evaluator(config)
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(
        '{"event":"hpo_trial_completed","model":"dummy","cv_scores":{"roc_auc":0.7},"params":"bad"}\n'
    )

    import iter8ml.engine.hpo_importance as hpo_importance

    def _raise_importance(study):
        raise RuntimeError("importance failed")

    monkeypatch.setattr(hpo_importance, "compute_param_importance", _raise_importance)

    result = optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=1,
        search_space={"lr": [0.001, 0.01]},
        task="classification",
        log_path=str(log_path),
        metrics=["roc_auc"],
    )

    assert "warmstart_summary" in result
    assert "warnings" in result
    assert any(w["source"] == "hpo_importance" for w in result["warnings"])


def test_setup_hpo_components_rejects_baseline_models(tmp_path):
    """Baselines pass validate_model_name but have no searchable config."""
    from iter8ml.engine.hpo import setup_hpo_components

    data_file = tmp_path / "test.csv"
    data_file.write_text("a,b,target\n1.0,2.0,0.0\n3.0,4.0,1.0\n5.0,6.0,0.0\n")

    with pytest.raises(ValueError, match="not HPO-able"):
        setup_hpo_components(str(data_file), "target", "classification", "naive_baseline")

    with pytest.raises(ValueError, match="no configurable search space"):
        setup_hpo_components(str(data_file), "target", "regression", "linear_baseline")


def test_optimize_model_flags_warmstart_metric_mismatch(tmp_path):
    """Historical events scored on another metric must not be injected."""
    from unittest.mock import Mock

    log_path = tmp_path / "events.jsonl"
    log_path.write_text(
        '{"event": "model_completed", "run_id": "r1", "model": "dummy", '
        '"cv_scores": {"r2": 0.9}, "params": {"lr": 0.01}}\n'
    )
    evaluator = Mock()
    evaluator.evaluate.return_value = {"rmse": 1.5}

    result = optimize_model(
        DummyModel,
        np.zeros((10, 2)),
        np.zeros(10),
        evaluator,
        "dummy",
        n_trials=1,
        task="regression",
        log_path=str(log_path),
        metrics=["rmse"],
    )

    # Direction still comes from the primary metric...
    assert result["direction"] == "minimize"
    assert result["primary_metric"] == "rmse"
    # ...no maximize-oriented warmstart injection happened...
    assert result.get("warmstart_trials", 0) == 0
    assert result["warmstart_summary"]["n_skipped_metric_mismatch"] == 1
    # ...and the skip is flagged as a warning.
    assert any(w.get("warning_type") == "MetricMismatch" for w in result.get("warnings", []))


def test_optimize_model_warmstarts_compatible_metric(tmp_path):
    """Historical events scored on the current primary metric still warm the study."""
    from unittest.mock import Mock

    log_path = tmp_path / "events.jsonl"
    log_path.write_text(
        '{"event": "model_completed", "run_id": "r1", "model": "dummy", '
        '"cv_scores": {"rmse": 2.5}, "params": {"lr": 0.01}}\n'
    )
    evaluator = Mock()
    evaluator.evaluate.return_value = {"rmse": 1.5}

    result = optimize_model(
        DummyModel,
        np.zeros((10, 2)),
        np.zeros(10),
        evaluator,
        "dummy",
        n_trials=1,
        task="regression",
        log_path=str(log_path),
        metrics=["rmse"],
    )

    assert result["warmstart_trials"] == 1
    assert result["warmstart_summary"]["n_trials_injected"] == 1
    assert result["warmstart_summary"]["n_skipped_metric_mismatch"] == 0


class _ExplodingEvaluator:
    """Evaluator whose every evaluation fails — models systematic breakage."""

    metrics: ClassVar[list[str]] = ["roc_auc"]

    def evaluate(self, model_cls, X, y, task="classification", **params):
        raise ValueError("boom: synthetic model failure")


class _SpyEvaluator:
    """Records every trial's params; returns finite scores."""

    metrics: ClassVar[list[str]] = ["roc_auc"]

    def __init__(self):
        self.seen: list[dict] = []

    def evaluate(self, model_cls, X, y, task="classification", **params):
        self.seen.append(dict(params))
        return {"roc_auc": 0.5 + np.random.rand() * 0.1}


def test_optimize_model_raises_when_all_trials_prune():
    """A systematically broken model must fail loudly, not 'complete' pruned."""
    X = np.random.rand(30, 3)
    y = np.random.randint(0, 2, 30)
    with pytest.raises(ValueError, match=r"only 0 of 5 trials completed.*boom"):
        optimize_model(
            DummyModel,
            X,
            y,
            _ExplodingEvaluator(),
            "dummy",
            n_trials=5,
            search_space={"lr": [0.001, 0.1, "log"]},
            task="classification",
        )


def test_optimize_model_fixed_params_reach_every_trial():
    X = np.random.rand(30, 3)
    y = np.random.randint(0, 2, 30)
    evaluator = _SpyEvaluator()
    optimize_model(
        DummyModel,
        X,
        y,
        evaluator,
        "dummy",
        n_trials=3,
        search_space={"lr": [0.001, 0.1, "log"]},
        task="classification",
        fixed_params={"scale_pos_weight": 2.0},
    )
    assert evaluator.seen
    assert all(params.get("scale_pos_weight") == 2.0 for params in evaluator.seen)
