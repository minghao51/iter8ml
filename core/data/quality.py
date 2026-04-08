"""Data quality audit using Cleanlab for label noise detection."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def audit_data_quality(
    df: pl.DataFrame,
    target_col: str,
    *,
    output_path: str | Path | None = None,
    enabled: bool = True,
) -> dict:
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
        "mean_quality_score": round(float(np.mean(scores)), 4),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
