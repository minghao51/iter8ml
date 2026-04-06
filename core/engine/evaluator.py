"""Cross-validation strategies and metrics registry."""

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


def get_cv_split(strategy: str, n_splits: int = 5):
    """Return a cross-validation splitter."""
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif strategy == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")


class Evaluator:
    """Runs cross-validation and computes metrics."""

    def __init__(
        self,
        task: str,
        metrics: list[str] | None = None,
        cv_folds: int = 5,
        cv_strategy: str = "stratified",
    ):
        self.task = task
        default_metrics = ["roc_auc", "f1_macro"] if task == "classification" else ["rmse", "r2"]
        self.metrics = metrics or default_metrics
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy

    def evaluate(
        self,
        model_cls,
        X: np.ndarray,
        y: np.ndarray,
        task: str | None = None,
        **model_kwargs,
    ) -> dict[str, float]:
        """Run cross-validation and return aggregated metrics.

        Args:
            model_cls: Model class (not instance). A fresh instance is created
                for each fold to prevent state leakage across folds.
            X: Feature matrix.
            y: Target vector.
            task: Task type override. Falls back to self.task.
            **model_kwargs: Passed to model_cls constructor.
        """
        cv = get_cv_split(self.cv_strategy, self.cv_folds)
        fold_scores = {m: [] for m in self.metrics}
        model_task = task or self.task

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_cls(task=model_task, **model_kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            for metric_name in self.metrics:
                metric_fn = METRICS_REGISTRY[model_task][metric_name]
                if metric_name == "log_loss":
                    y_proba = model.predict_proba(X_val)
                    if y_proba is not None:
                        fold_scores[metric_name].append(metric_fn(y_val, y_proba))
                else:
                    fold_scores[metric_name].append(metric_fn(y_val, y_pred))

        return {m: float(np.mean(scores)) for m, scores in fold_scores.items()}
