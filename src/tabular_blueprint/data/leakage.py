"""Leakage detection audit: flags features where permutation destroys performance."""

from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score


class LeakageReport(BaseModel):
    """Report of features flagged for potential data leakage."""

    flagged_features: list[dict[str, Any]]
    n_features_tested: int
    n_flagged: int
    baseline_score: float


def detect_leakage(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    threshold: float = 0.15,
    cv_folds: int = 3,
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

    Returns:
        LeakageReport with flagged features and summary stats.
    """
    scoring = "roc_auc" if task == "classification" else "r2"

    model = (
        LogisticRegression(max_iter=1000, random_state=42)
        if task == "classification"
        else Ridge(alpha=1.0, random_state=42)
    )

    baseline_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
    baseline_score = float(np.mean(baseline_scores))

    flagged = []
    n_features = X.shape[1]

    rng = np.random.RandomState(42)

    X_working = X.copy()

    for col_idx in range(n_features):
        original_col = X_working[:, col_idx].copy()
        rng.shuffle(X_working[:, col_idx])

        permuted_scores = cross_val_score(model, X_working, y, cv=cv_folds, scoring=scoring)
        permuted_score = float(np.mean(permuted_scores))

        X_working[:, col_idx] = original_col

        drop = baseline_score - permuted_score
        if scoring in ("neg_mean_squared_error",):
            drop = -drop

        if drop > threshold:
            flagged.append(
                {
                    "feature_index": col_idx,
                    "baseline_score": round(baseline_score, 4),
                    "permuted_score": round(permuted_score, 4),
                    "score_drop": round(drop, 4),
                }
            )

    flagged.sort(key=lambda f: f["score_drop"], reverse=True)

    return LeakageReport(
        flagged_features=flagged,
        n_features_tested=n_features,
        n_flagged=len(flagged),
        baseline_score=round(baseline_score, 4),
    )
