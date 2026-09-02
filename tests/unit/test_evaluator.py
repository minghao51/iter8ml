"""Tests for evaluator metric routing and probability handling."""

import numpy as np
import pytest

from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy, TaskType
from iter8ml.engine.evaluator import Evaluator, get_cv_split


class ProbaDrivenModel:
    """Model where label predictions are weak but probabilities are informative."""

    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Intentionally bad thresholded predictions.
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        # Probability is strongly correlated with X[:, 0].
        proba_pos = np.clip(X[:, 0], 1e-6, 1 - 1e-6)
        return np.column_stack([1 - proba_pos, proba_pos])


class NoProbaModel:
    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return None


def test_roc_auc_uses_probabilities():
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=200)
    X = np.column_stack([0.05 + 0.9 * y + 0.02 * rng.randn(200), rng.randn(200)])
    X[:, 0] = np.clip(X[:, 0], 0.0, 1.0)

    config = ExperimentConfig(
        name="eval_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc", "accuracy"],
    )
    scores = Evaluator(config).evaluate(ProbaDrivenModel, X, y)
    assert scores["roc_auc"] > 0.9


class MulticlassModel:
    """Model for 3+ class classification."""

    def __init__(self, task: str = "classification"):
        self.task = task

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(X[:, :3], axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        raw = np.abs(X[:, :3])
        return raw / raw.sum(axis=1, keepdims=True)


def test_multiclass_roc_auc():
    rng = np.random.RandomState(42)
    y = rng.randint(0, 3, size=150)
    X = np.column_stack(
        [(y == c).astype(float) + 0.1 * rng.randn(150) for c in range(3)] + [rng.randn(150)]
    )
    config = ExperimentConfig(
        name="eval_mc",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        cv_folds=3,
        metrics=["roc_auc"],
    )
    scores = Evaluator(config).evaluate(MulticlassModel, X, y)
    assert scores["roc_auc"] > 0.5


def test_roc_auc_requires_predict_proba():
    X = np.random.RandomState(0).rand(40, 3)
    y = np.random.RandomState(1).randint(0, 2, size=40)

    config = ExperimentConfig(
        name="eval_test",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        cv_folds=2,
        metrics=["roc_auc"],
    )

    with pytest.raises(ValueError, match="Metric 'roc_auc' requires predict_proba"):
        Evaluator(config).evaluate(NoProbaModel, X, y)


def test_log_loss_metric():
    X = np.random.RandomState(0).rand(60, 3)
    y = np.random.RandomState(1).randint(0, 2, size=60)

    config = ExperimentConfig(
        name="eval_ll",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        cv_folds=2,
        metrics=["log_loss"],
    )
    scores = Evaluator(config).evaluate(ProbaDrivenModel, X, y)
    assert "log_loss" in scores
    assert scores["log_loss"] > 0


def test_log_loss_requires_proba():
    X = np.random.RandomState(0).rand(20, 3)
    y = np.random.RandomState(1).randint(0, 2, size=20)

    config = ExperimentConfig(
        name="eval_ll",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        cv_folds=2,
        metrics=["log_loss"],
    )
    with pytest.raises(ValueError, match="Metric 'log_loss'"):
        Evaluator(config).evaluate(NoProbaModel, X, y)


def test_timeseries_cv():
    splitter = get_cv_split(CVStrategy.TIMESERIES, n_splits=3)
    from sklearn.model_selection import TimeSeriesSplit

    assert isinstance(splitter, TimeSeriesSplit)
    assert splitter.n_splits == 3


def test_get_cv_split_unknown():
    with pytest.raises(ValueError, match="Unknown CV strategy"):
        get_cv_split("unknown_strategy", n_splits=5)


def test_get_cv_split_kfold():
    splitter = get_cv_split(CVStrategy.KFOLD, n_splits=5)
    from sklearn.model_selection import KFold

    assert isinstance(splitter, KFold)


def test_compute_lift_higher_is_better():
    model_scores = {"roc_auc": 0.90}
    baseline_scores = {"roc_auc": 0.80}
    lift = Evaluator.compute_lift(model_scores, baseline_scores, "roc_auc")
    assert lift == pytest.approx(0.125, rel=1e-3)


def test_compute_lift_lower_is_better():
    model_scores = {"rmse": 0.5}
    baseline_scores = {"rmse": 1.0}
    lift = Evaluator.compute_lift(model_scores, baseline_scores, "rmse")
    assert lift == 0.5


def test_compute_lift_zero_baseline_returns_none():
    """Lift over a 0 baseline is undefined — None, not a fabricated 0.0."""
    lift = Evaluator.compute_lift({"acc": 0.9}, {"acc": 0.0}, "acc")
    assert lift is None


def test_compute_lift_missing_metric_returns_none():
    assert Evaluator.compute_lift({"roc_auc": 0.9}, {"f1": 0.8}, "roc_auc") is None
    assert Evaluator.compute_lift({"f1": 0.9}, {"roc_auc": 0.8}, "roc_auc") is None
    assert Evaluator.compute_lift({}, {"roc_auc": 0.8}, "roc_auc") is None


def test_compute_lift_happy_path_unchanged():
    assert Evaluator.compute_lift({"roc_auc": 0.9}, {"roc_auc": 0.8}, "roc_auc") == pytest.approx(
        0.125, rel=1e-3
    )


def test_get_cv_split_seed_changes_fold_assignment():
    import numpy as np

    X = np.arange(60).reshape(30, 2).astype(float)
    y = np.array([0, 1] * 15)
    folds_a = list(get_cv_split(CVStrategy.STRATIFIED, n_splits=3, random_seed=1).split(X, y))
    folds_b = list(get_cv_split(CVStrategy.STRATIFIED, n_splits=3, random_seed=2).split(X, y))
    assert [tuple(a[1]) for a in folds_a] != [tuple(b[1]) for b in folds_b]


def test_evaluate_with_std_returns_mean_and_fold_std():
    import numpy as np

    from iter8ml.config import ExperimentConfig
    from iter8ml.constants import TaskType
    from iter8ml.engine.models.baselines import NaiveBaseline

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + rng.normal(scale=0.1, size=60) > 0).astype(float)
    config = ExperimentConfig(
        name="t",
        task=TaskType.CLASSIFICATION,
        target_col="y",
        data_path="",
        cv_folds=4,
        metrics=["accuracy"],
    )
    means, stds = Evaluator(config).evaluate_with_std(NaiveBaseline, X, y)
    assert set(means) == {"accuracy"}
    assert set(stds) == {"accuracy"}
    assert all(np.isfinite(v) for v in means.values())
    assert all(v >= 0 for v in stds.values())
