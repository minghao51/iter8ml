"""Promotion gate helpers based on explicit run evidence."""

from __future__ import annotations

from typing import Any


def promotion_eligibility(
    *,
    quality_ok: bool,
    leakage_ok: bool,
    oof_predictions: bool,
    metrics: dict[str, Any],
    minimum_score: float | None = None,
    primary_metric: str | None = None,
) -> dict[str, object]:
    score = metrics.get(primary_metric) if primary_metric else None
    score_ok = minimum_score is None or (isinstance(score, int | float) and score >= minimum_score)
    eligible = quality_ok and leakage_ok and oof_predictions and score_ok
    return {
        "eligible": eligible,
        "quality_ok": quality_ok,
        "leakage_ok": leakage_ok,
        "oof_predictions": oof_predictions,
        "score_ok": score_ok,
        "primary_metric": primary_metric,
        "score": score,
    }
