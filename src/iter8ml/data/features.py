"""Automated Feature Engineering: target transformation and interaction discovery."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from pydantic import BaseModel
from scipy import stats as sp_stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


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
    effective_n_jobs: int = 1
    duration_seconds: float = 0.0


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


def _to_sklearn_estimator(model: Any, task: str) -> Any:
    if hasattr(model, "_estimator_type"):
        return model
    if hasattr(model, "_model") and hasattr(model._model, "fit"):
        inner = model._model
        if hasattr(inner, "_estimator_type"):
            return inner
    classes = getattr(model, "_class_labels", None)
    return _SKLearnAdapter(model, task, classes=classes)


class _SKLearnAdapter:
    def __init__(self, model: Any, task: str, *, classes: np.ndarray | None = None) -> None:
        self._model = model
        self.classes_ = classes
        self._estimator_type = "classifier" if task == "classification" else "regressor"

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SKLearnAdapter:
        self._model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score, r2_score

        if self._estimator_type == "classifier":
            return float(accuracy_score(y, self.predict(X)))
        return float(r2_score(y, self.predict(X)))


def extract_top_k_features(
    model_or_predictions: Any,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    feature_names: list[str] | None = None,
    task: str = "classification",
    random_seed: int = 42,
    n_repeats: int = 5,
) -> tuple[list[int], Any | None]:
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(n_features)]

    perm_result = None
    if hasattr(model_or_predictions, "predict"):
        sk_model = _to_sklearn_estimator(model_or_predictions, task)
        perm_result = permutation_importance(
            sk_model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_seed,
            scoring="roc_auc" if task == "classification" else "r2",
        )
        top_indices = np.argsort(perm_result.importances_mean)[::-1][:k]
    else:
        top_indices = np.arange(min(k, n_features))

    return [int(i) for i in top_indices], perm_result


def discover_interactions(
    X: np.ndarray,
    y: np.ndarray,
    top_k_indices: list[int],
    feature_names: list[str] | None = None,
    task: str = "classification",
    lift_threshold: float = 0.01,
    cv_folds: int = 3,
    random_seed: int = 42,
    n_jobs: int = 1,
    max_candidate_pairs: int = 200,
) -> tuple[np.ndarray, InteractionDiscoveryResult]:
    start = time.perf_counter()
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(n_features)]

    baseline_model = (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_seed),
        )
        if task == "classification"
        else make_pipeline(StandardScaler(), Ridge(random_state=random_seed))
    )
    scoring = "roc_auc" if task == "classification" else "r2"

    baseline_scores = cross_val_score(
        baseline_model,
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
    )
    baseline_mean = float(np.mean(baseline_scores))

    candidates: list[InteractionCandidate] = []
    kept_features: list[np.ndarray] = []
    kept_names: list[str] = []

    pair_list: list[tuple[int, int, str]] = []
    for i_idx, i in enumerate(top_k_indices):
        for j in top_k_indices[i_idx + 1 :]:
            for op_name in ("multiply", "ratio"):
                pair_list.append((i, j, op_name))
    if max_candidate_pairs > 0 and len(pair_list) > max_candidate_pairs:
        pair_list = pair_list[:max_candidate_pairs]

    def _evaluate_pair(
        i: int, j: int, op_name: str
    ) -> tuple[InteractionCandidate, np.ndarray | None, str | None] | None:
        x_i = X[:, i]
        x_j = X[:, j]
        op_fn = np.multiply if op_name == "multiply" else _safe_ratio
        interaction_feature = op_fn(x_i, x_j)
        if interaction_feature is None:
            return None
        X_augmented = np.column_stack([X, interaction_feature.reshape(-1, 1)])
        try:
            aug_scores = cross_val_score(
                baseline_model,
                X_augmented,
                y,
                cv=cv_folds,
                scoring=scoring,
            )
            aug_mean = float(np.mean(aug_scores))
        except (ValueError, RuntimeError):
            return None
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
        feature_data = interaction_feature.reshape(-1, 1) if kept else None
        name = f"{feature_names[i]}_{op_name}_{feature_names[j]}" if kept else None
        return candidate, feature_data, name

    effective_jobs = _effective_parallel_jobs(
        requested_jobs=n_jobs,
        n_tasks=len(pair_list),
        n_samples=X.shape[0],
        n_features=X.shape[1],
    )
    if effective_jobs == 1:
        pair_results = [_evaluate_pair(i, j, op_name) for i, j, op_name in pair_list]
    else:
        pair_results = Parallel(n_jobs=effective_jobs)(
            delayed(_evaluate_pair)(i, j, op_name) for i, j, op_name in pair_list
        )

    for pair_result in pair_results:
        if pair_result is None:
            continue
        cand, feature_data, name = pair_result
        candidates.append(cand)
        if feature_data is not None and name is not None:
            kept_features.append(feature_data)
            kept_names.append(name)

    n_tested = len(candidates)
    n_kept = len(kept_features)

    X_new = np.hstack(kept_features) if kept_features else np.empty((X.shape[0], 0))

    discovery_result: InteractionDiscoveryResult = InteractionDiscoveryResult(
        n_candidates_tested=n_tested,
        n_candidates_kept=n_kept,
        candidates=candidates,
        new_feature_names=kept_names,
        effective_n_jobs=effective_jobs,
        duration_seconds=round(time.perf_counter() - start, 4),
    )
    logger.info(
        "afe_interactions tested=%d kept=%d jobs=%d elapsed_seconds=%.3f",
        n_tested,
        n_kept,
        effective_jobs,
        time.perf_counter() - start,
    )

    return X_new, discovery_result


def _effective_parallel_jobs(
    *, requested_jobs: int, n_tasks: int, n_samples: int, n_features: int
) -> int:
    if requested_jobs <= 1 or n_tasks <= 1:
        return 1
    cpu_cap = max(1, os.cpu_count() or 1)
    requested_cap = min(int(requested_jobs), n_tasks, cpu_cap)
    matrix_size = n_samples * n_features
    if matrix_size >= 1_000_000:
        return min(requested_cap, 2)
    return requested_cap


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.divide(
            a,
            b,
            out=np.zeros_like(a, dtype=float),
            where=np.abs(b) > 1e-10,
        )
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
    n_repeats: int = 5,
    precomputed_importances: Any | None = None,
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

    if precomputed_importances is not None:
        importances = precomputed_importances.importances_mean
    else:
        sk_model = _to_sklearn_estimator(model, task)
        perm_result = permutation_importance(
            sk_model, X, y, n_repeats=n_repeats, random_state=random_seed, scoring=scoring
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
