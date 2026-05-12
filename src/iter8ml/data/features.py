"""Automated Feature Engineering: target transformation and interaction discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel
from scipy import stats as sp_stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score


class TargetTransformResult(BaseModel):
    """Result of applying a target transformation (log1p, box-cox, yeo-johnson)."""

    original_skewness: float
    transformed_skewness: float
    method: str
    applied: bool


class InteractionCandidate(BaseModel):
    """A single candidate interaction feature discovered during AFE."""

    feature_names: tuple[str, ...]
    operation: str
    lift: float
    kept: bool


class InteractionDiscoveryResult(BaseModel):
    """Summary of AFE interaction search."""

    n_candidates_tested: int
    n_candidates_kept: int
    candidates: list[InteractionCandidate]
    new_feature_names: list[str]


class PruningResult(BaseModel):
    """Result of pruning low-importance features."""

    n_original: int
    n_kept: int
    n_dropped: int
    dropped_features: list[str]
    kept_indices: list[int]


def detect_target_skewness(y: np.ndarray) -> float:
    return float(sp_stats.skew(y))


def transform_target(
    y: np.ndarray,
    method: Literal["auto", "log1p", "yeo-johnson", "box-cox", "none"] = "auto",
    skewness_threshold: float = 1.0,
) -> tuple[np.ndarray, TargetTransformResult, _TargetTransformer | None]:
    if method == "none":
        return (
            y,
            TargetTransformResult(
                original_skewness=detect_target_skewness(y),
                transformed_skewness=detect_target_skewness(y),
                method="none",
                applied=False,
            ),
            None,
        )

    original_skewness = detect_target_skewness(y)

    if method == "auto":
        if abs(original_skewness) <= skewness_threshold:
            return (
                y,
                TargetTransformResult(
                    original_skewness=original_skewness,
                    transformed_skewness=original_skewness,
                    method="none",
                    applied=False,
                ),
                None,
            )
        all_positive = np.all(y > 0)
        method = "box-cox" if all_positive else "yeo-johnson"

    transformer = _TargetTransformer(method=method)
    y_transformed = transformer.fit_transform(y)
    transformed_skewness = detect_target_skewness(y_transformed)

    return (
        y_transformed,
        TargetTransformResult(
            original_skewness=original_skewness,
            transformed_skewness=transformed_skewness,
            method=method,
            applied=True,
        ),
        transformer,
    )


class _TargetTransformer:
    def __init__(self, method: str):
        self.method = method
        self._scaler: Any = None

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        if self.method == "log1p":
            return np.log1p(y)  # type: ignore[no-any-return]
        elif self.method == "yeo-johnson":
            from sklearn.preprocessing import PowerTransformer

            self._scaler = PowerTransformer(method="yeo-johnson")
            return self._scaler.fit_transform(y.reshape(-1, 1)).ravel()  # type: ignore[no-any-return]
        elif self.method == "box-cox":
            from sklearn.preprocessing import PowerTransformer

            self._scaler = PowerTransformer(method="box-cox")
            return self._scaler.fit_transform(y.reshape(-1, 1)).ravel()  # type: ignore[no-any-return]
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.inverse_transform(y.reshape(-1, 1)).ravel()  # type: ignore[no-any-return]
        elif self.method == "log1p":
            return np.expm1(y)  # type: ignore[no-any-return]
        return y


def extract_top_k_features(
    model_or_predictions: Any,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    feature_names: list[str] | None = None,
    task: str = "classification",
    random_seed: int = 42,
) -> list[int]:
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(n_features)]

    if hasattr(model_or_predictions, "predict"):
        result = permutation_importance(
            model_or_predictions,
            X,
            y,
            n_repeats=10,
            random_state=random_seed,
            scoring="roc_auc" if task == "classification" else "r2",
        )
        top_indices = np.argsort(result.importances_mean)[::-1][:k]
    else:
        top_indices = np.arange(min(k, n_features))

    return [int(i) for i in top_indices]


def discover_interactions(
    X: np.ndarray,
    y: np.ndarray,
    top_k_indices: list[int],
    feature_names: list[str] | None = None,
    task: str = "classification",
    lift_threshold: float = 0.01,
    cv_folds: int = 3,
    random_seed: int = 42,
) -> tuple[np.ndarray, InteractionDiscoveryResult]:
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(n_features)]

    baseline_model_cls = LogisticRegression if task == "classification" else Ridge
    scoring = "roc_auc" if task == "classification" else "r2"

    baseline_scores = cross_val_score(
        baseline_model_cls(random_state=random_seed),
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
    )
    baseline_mean = float(np.mean(baseline_scores))

    candidates: list[InteractionCandidate] = []
    kept_features: list[np.ndarray] = []
    kept_names: list[str] = []

    for i_idx, i in enumerate(top_k_indices):
        for j in top_k_indices[i_idx + 1 :]:
            x_i = X[:, i]
            x_j = X[:, j]

            op_list: list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray | None]]] = [
                ("multiply", np.multiply),
                ("ratio", _safe_ratio),
            ]

            for op_name, op_fn in op_list:
                interaction_feature = op_fn(x_i, x_j)
                if interaction_feature is None:
                    continue

                X_augmented = np.column_stack([X, interaction_feature.reshape(-1, 1)])

                try:
                    aug_scores = cross_val_score(
                        baseline_model_cls(random_state=random_seed),
                        X_augmented,
                        y,
                        cv=cv_folds,
                        scoring=scoring,
                    )
                    aug_mean = float(np.mean(aug_scores))
                except (ValueError, RuntimeError):
                    continue

                lift = (
                    aug_mean - baseline_mean
                    if scoring != "neg_mean_squared_error"
                    else baseline_mean - aug_mean
                )
                kept = lift > lift_threshold

                candidate = InteractionCandidate(
                    feature_names=(feature_names[i], feature_names[j]),
                    operation=op_name,
                    lift=round(lift, 6),
                    kept=kept,
                )
                candidates.append(candidate)

                if kept:
                    kept_features.append(interaction_feature.reshape(-1, 1))
                    kept_names.append(f"{feature_names[i]}_{op_name}_{feature_names[j]}")

    n_tested = len(candidates)
    n_kept = len(kept_features)

    X_new = np.hstack(kept_features) if kept_features else np.empty((X.shape[0], 0))

    result = InteractionDiscoveryResult(
        n_candidates_tested=n_tested,
        n_candidates_kept=n_kept,
        candidates=candidates,
        new_feature_names=kept_names,
    )

    return X_new, result


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-10, a / b, 0.0)
    if np.all(np.isnan(result)) or np.all(np.isinf(result)):
        return None
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def prune_features(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    min_importance: float = 0.001,
    task: str = "classification",
    random_seed: int = 42,
) -> tuple[np.ndarray, PruningResult]:
    """Remove features with permutation importance below a threshold.

    Uses the same GBDT model already fitted during AFE to avoid retraining.

    Args:
        model: A fitted model with a ``predict`` method.
        X: Feature matrix (may include AFE-generated features).
        y: Target vector.
        feature_names: Optional names for each column in X.
        min_importance: Drop features with mean importance below this value.
        task: "classification" or "regression".
        random_seed: Random state for permutation importance.

    Returns:
        Tuple of (pruned X, PruningResult).
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(n_features)]

    scoring = "roc_auc" if task == "classification" else "r2"

    if not hasattr(model, "predict"):
        result = PruningResult(
            n_original=n_features,
            n_kept=n_features,
            n_dropped=0,
            dropped_features=[],
            kept_indices=list(range(n_features)),
        )
        return X, result

    perm_result = permutation_importance(
        model, X, y, n_repeats=10, random_state=random_seed, scoring=scoring
    )

    importances = perm_result.importances_mean
    kept_mask = importances >= min_importance
    kept_indices = [int(i) for i in np.where(kept_mask)[0]]
    dropped_indices = [int(i) for i in np.where(~kept_mask)[0]]
    dropped_names = [feature_names[i] for i in dropped_indices]

    X_pruned = X[:, kept_mask]

    result = PruningResult(
        n_original=n_features,
        n_kept=len(kept_indices),
        n_dropped=len(dropped_indices),
        dropped_features=dropped_names,
        kept_indices=kept_indices,
    )
    return X_pruned, result
