"""Data quality audit using Cleanlab for label noise detection."""

from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def audit_data_quality(
    df: pl.DataFrame,
    target_col: str,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """
    Run Cleanlab-based label noise detection.

    Returns a quality report with flagged indices and noise estimates.
    """
    if not enabled:
        return {"enabled": False, "message": "Quality audit skipped per config"}

    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        return {"enabled": False, "message": "cleanlab not installed"}

    # Use .select() to get feature columns (may create view instead of copy)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    if len(np.unique(y)) < 2:
        return {"enabled": False, "message": "Need at least 2 classes for quality audit"}

    model = LogisticRegression(max_iter=1000, random_state=42)
    pred_probs = cross_val_predict(model, X, y, cv=3, method="predict_proba")

    scores = get_label_quality_scores(y, pred_probs)
    issue_indices = find_label_issues(y, pred_probs, return_indices_ranked_by="self_confidence")

    report = {
        "enabled": True,
        "n_rows": len(df),
        "n_issues": len(issue_indices),
        "noise_rate": round(len(issue_indices) / len(df), 4),
        "flagged_indices": issue_indices.tolist()[:100],
        "quality_scores": np.asarray(scores, dtype=float).tolist(),
        "mean_quality_score": round(float(np.mean(scores)), 4),
    }

    return report


def clean_noise(
    df: pl.DataFrame,
    report: dict[str, Any],
    target_col: str,
    quality_threshold: float = 0.5,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Drop rows with label quality scores below a threshold.

    Args:
        df: Input DataFrame.
        report: Quality report dict from ``audit_data_quality``.
        target_col: Name of the target column.
        quality_threshold: Rows with quality score below this value are dropped.

    Returns:
        Tuple of (cleaned DataFrame, summary dict).
    """
    if not report.get("enabled"):
        return df, {"n_before": len(df), "n_after": len(df), "n_dropped": 0}

    quality_scores = report.get("quality_scores")
    drop_mask = None

    if isinstance(quality_scores, list) and len(quality_scores) == len(df):
        score_array = np.asarray(quality_scores, dtype=float)
        drop_mask = score_array < quality_threshold

    if drop_mask is None:
        return df, {"n_before": len(df), "n_after": len(df), "n_dropped": 0}

    mask = pl.Series("drop_mask", drop_mask)
    cleaned = df.filter(~mask)

    summary = {
        "n_before": len(df),
        "n_after": len(cleaned),
        "n_dropped": len(df) - len(cleaned),
        "threshold": quality_threshold,
    }
    return cleaned, summary
