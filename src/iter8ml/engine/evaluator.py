from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from iter8ml.config import ExperimentConfig
from iter8ml.constants import CVStrategy
from iter8ml.services.reporting import LOWER_IS_BETTER_METRICS

METRICS_REGISTRY = {
    "classification": {
        "roc_auc": roc_auc_score,
        "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score,
        "log_loss": log_loss,
    },
    "regression": {
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error,
        "r2": r2_score,
    },
}


def get_cv_split(strategy: CVStrategy, n_splits: int = 5) -> Any:
    """Return a cross-validation splitter from a CVStrategy enum."""
    if strategy == CVStrategy.KFOLD:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif strategy == CVStrategy.STRATIFIED:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif strategy == CVStrategy.TIMESERIES:
        return TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")


class Evaluator:
    """Runs cross-validation and computes metrics."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    @property
    def task(self) -> str:
        return self.config.task.value

    @property
    def metrics(self) -> list[str]:
        return self.config.metrics

    @property
    def cv_folds(self) -> int:
        return self.config.cv_folds

    @property
    def cv_strategy(self) -> str:
        return self.config.cv_strategy.value

    def _run_cv(
        self,
        model_cls: Any,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
        **model_kwargs: Any,
    ) -> dict[str, list[float]]:
        """Run cross-validation and return per-fold scores keyed by metric.

        A fresh model instance is created for each fold to prevent state leakage.
        When ``fold_indices`` is provided (explicit train/validation index pairs),
        it is used instead of the configured CV splitter.
        """
        if fold_indices is None:
            cv = get_cv_split(self.config.cv_strategy, self.cv_folds)
            fold_indices = list(cv.split(X, y))
        fold_scores: dict[str, list[float]] = {m: [] for m in self.metrics}

        metric_fns = {
            metric_name: METRICS_REGISTRY[task][metric_name] for metric_name in self.metrics
        }

        all_classes = np.unique(y)

        for train_idx, val_idx in fold_indices:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_cls(task=task, **model_kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = None
            requires_proba = any(m in {"roc_auc", "log_loss"} for m in self.metrics)
            if requires_proba:
                y_proba = model.predict_proba(X_val)

            for metric_name, metric_fn in metric_fns.items():
                if metric_name == "log_loss":
                    if y_proba is not None:
                        fold_scores[metric_name].append(metric_fn(y_val, y_proba))
                    else:
                        raise ValueError(
                            "Metric 'log_loss' requires predict_proba(), but model "
                            f"'{getattr(model, 'model_name', type(model).__name__)}' "
                            "returned None."
                        )
                elif metric_name == "roc_auc":
                    if y_proba is None:
                        raise ValueError(
                            "Metric 'roc_auc' requires predict_proba(), but model "
                            f"'{getattr(model, 'model_name', type(model).__name__)}' "
                            "returned None."
                        )
                    if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                        score = roc_auc_score(y_val, y_proba, multi_class="ovr", labels=all_classes)
                    elif y_proba.ndim == 2 and y_proba.shape[1] > 1:
                        score = metric_fn(y_val, y_proba[:, 1])
                    else:
                        score = metric_fn(y_val, y_proba)
                    fold_scores[metric_name].append(score)
                else:
                    fold_scores[metric_name].append(metric_fn(y_val, y_pred))

        return fold_scores

    def evaluate_with_folds(
        self,
        model_cls: Any,
        X: np.ndarray,
        y: np.ndarray,
        fold_indices: list[tuple[np.ndarray, np.ndarray]],
        task: str | None = None,
        **model_kwargs: Any,
    ) -> dict[str, float]:
        """Cross-validate using explicit fold index pairs (from an external split)."""
        fold_scores = self._run_cv(
            model_cls, X, y, task or self.task, fold_indices=fold_indices, **model_kwargs
        )
        return {m: float(np.mean(scores)) for m, scores in fold_scores.items()}

    def evaluate(
        self,
        model_cls: Any,
        X: np.ndarray,
        y: np.ndarray,
        task: str | None = None,
        **model_kwargs: Any,
    ) -> dict[str, float]:
        """Run cross-validation and return the mean of each metric.

        Args:
            model_cls: Model class (not instance). A fresh instance is created
                for each fold to prevent state leakage across folds.
            X: Feature matrix.
            y: Target vector.
            task: Task type override. Falls back to self.task.
            **model_kwargs: Passed to model_cls constructor.
        """
        fold_scores = self._run_cv(model_cls, X, y, task or self.task, **model_kwargs)
        return {m: float(np.mean(scores)) for m, scores in fold_scores.items()}

    @staticmethod
    def compute_lift(
        model_scores: dict[str, float],
        baseline_scores: dict[str, float],
        metric_name: str,
    ) -> float:
        """Compute lift of model over baseline for a given metric.

        Returns the relative improvement as a fraction (e.g. 0.15 = 15% lift).
        For metrics where lower is better (rmse, mae), positive lift means improvement.
        """
        model_val = model_scores.get(metric_name, 0.0)
        baseline_val = baseline_scores.get(metric_name, 0.0)
        if baseline_val == 0:
            return 0.0
        if metric_name in LOWER_IS_BETTER_METRICS:
            return (baseline_val - model_val) / abs(baseline_val)
        return (model_val - baseline_val) / abs(baseline_val)
