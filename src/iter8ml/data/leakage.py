"""Leakage detection audit: flags features where permutation destroys performance."""

import logging
import time
from typing import Any

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iter8ml.utils.parallel import _effective_parallel_jobs

logger = logging.getLogger(__name__)


class LeakageReport(BaseModel):
    """Report of features flagged for potential data leakage."""

    flagged_features: list[dict[str, Any]]
    n_features_tested: int
    n_flagged: int
    baseline_score: float
    effective_n_jobs: int = 1
    duration_seconds: float = 0.0


def detect_leakage(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    threshold: float = 0.15,
    cv_folds: int = 3,
    n_jobs: int = 1,
) -> LeakageReport:
    """Run permutation importance on a naive baseline to detect leaky features.

    For each feature, permute its values and measure the drop in CV score.
    Features whose permutation causes a large performance drop relative to
    baseline are flagged for manual review.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector.
        task: "classification" or "regression".
        threshold: Minimum relative score drop to flag a feature.
        cv_folds: Number of CV folds for scoring.
        n_jobs: Number of parallel jobs (default 1 for safe resource usage).

    Returns:
        LeakageReport with flagged features and summary stats.
    """
    start = time.perf_counter()
    scoring = "roc_auc" if task == "classification" else "r2"

    model = (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
        if task == "classification"
        else make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    )

    baseline_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
    baseline_score = float(np.mean(baseline_scores))

    n_features = X.shape[1]

    def _score_drop(permuted_score: float) -> float:
        drop = baseline_score - permuted_score
        if scoring in ("neg_mean_squared_error",):
            drop = -drop
        return drop

    effective_jobs = _effective_parallel_jobs(
        requested_jobs=n_jobs,
        n_tasks=n_features,
        n_samples=X.shape[0],
        n_features=X.shape[1],
    )
    if effective_jobs == 1:
        flagged: list[dict[str, Any]] = []
        X_working = X.copy()
        for col_idx in range(n_features):
            original_col = X_working[:, col_idx].copy()
            rng_local = np.random.default_rng(42 + col_idx)
            rng_local.shuffle(X_working[:, col_idx])
            permuted_scores = cross_val_score(model, X_working, y, cv=cv_folds, scoring=scoring)
            X_working[:, col_idx] = original_col
            permuted_score = float(np.mean(permuted_scores))
            drop = _score_drop(permuted_score)
            if drop > threshold:
                flagged.append(
                    {
                        "feature_index": col_idx,
                        "baseline_score": round(baseline_score, 4),
                        "permuted_score": round(permuted_score, 4),
                        "score_drop": round(drop, 4),
                    }
                )
    else:

        def _score_one_col(col_idx: int) -> dict[str, Any] | None:
            X_col = X.copy()
            rng_local = np.random.default_rng(42 + col_idx)
            rng_local.shuffle(X_col[:, col_idx])
            permuted_scores = cross_val_score(model, X_col, y, cv=cv_folds, scoring=scoring)
            permuted_score = float(np.mean(permuted_scores))
            drop = _score_drop(permuted_score)
            if drop > threshold:
                return {
                    "feature_index": col_idx,
                    "baseline_score": round(baseline_score, 4),
                    "permuted_score": round(permuted_score, 4),
                    "score_drop": round(drop, 4),
                }
            return None

        results = Parallel(n_jobs=effective_jobs)(
            delayed(_score_one_col)(col_idx) for col_idx in range(n_features)
        )
        flagged = [r for r in results if r is not None]

    flagged.sort(key=lambda f: f["score_drop"], reverse=True)
    logger.info(
        "leakage_audit tested=%d flagged=%d jobs=%d elapsed_seconds=%.3f",
        n_features,
        len(flagged),
        effective_jobs,
        time.perf_counter() - start,
    )

    return LeakageReport(
        flagged_features=flagged,
        n_features_tested=n_features,
        n_flagged=len(flagged),
        baseline_score=round(baseline_score, 4),
        effective_n_jobs=effective_jobs,
        duration_seconds=round(time.perf_counter() - start, 4),
    )
