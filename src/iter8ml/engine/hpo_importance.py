"""Hyperparameter importance analysis and search space refinement using Optuna."""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
from pydantic import BaseModel, ConfigDict


class ParamImportance(BaseModel):
    """A single hyperparameter's importance score."""

    model_config = ConfigDict(frozen=True)

    param_name: str
    importance: float


class ImportanceReport(BaseModel):
    """Complete hyperparameter importance analysis result."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    n_trials: int
    importances: list[ParamImportance]
    evaluator: str
    timestamp: str


def compute_param_importance(
    study: optuna.Study,
    evaluator_class: type | None = None,
) -> ImportanceReport:
    """Compute hyperparameter importance using Optuna's PedAnovaImportanceEvaluator.

    Args:
        study: Completed or partially-completed Optuna study
        evaluator_class: Importance evaluator class. Defaults to
            optuna.importance.PedAnovaImportanceEvaluator.

    Returns:
        ImportanceReport with ranked parameter importances.
    """
    if evaluator_class is None:
        evaluator_class = optuna.importance.PedAnovaImportanceEvaluator

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return ImportanceReport(
            model_name=study.study_name or "unknown",
            n_trials=0,
            importances=[],
            evaluator=evaluator_class.__name__,
            timestamp=datetime.now(UTC).isoformat(),
        )

    evaluator = evaluator_class()
    importances = evaluator.evaluate(study)

    result_list = [
        ParamImportance(param_name=name, importance=importance)
        for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)
    ]

    return ImportanceReport(
        model_name=study.study_name or "unknown",
        n_trials=len(study.trials),
        importances=result_list,
        evaluator=evaluator_class.__name__,
        timestamp=datetime.now(UTC).isoformat(),
    )


def suggest_refined_space(
    study: optuna.Study,
    original_space: dict[str, Any],
    top_k: int | None = None,
    importance_threshold: float = 0.01,
    expansion_factor: float = 1.3,
) -> dict[str, Any]:
    """Suggest refined search space bounds based on high-performing regions.

    Uses trial results to narrow bounds toward regions that performed well,
    while preserving exploration for low-importance params.

    Args:
        study: Optuna study with completed trials
        original_space: Original search space dict (param_name -> (low, high) or (low, high, kind))
        top_k: Only refine top K most important params. None = all above threshold.
        importance_threshold: Minimum importance to trigger refinement.
        expansion_factor: How much to expand high-performing bounds (1.3 = 30% wider).

    Returns:
        Refined search space dict with adjusted bounds for important params.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return original_space

    evaluator = optuna.importance.PedAnovaImportanceEvaluator()
    importances = evaluator.evaluate(study)

    refined: dict[str, Any] = {}
    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    for param_name, importance in sorted_params:
        if importance < importance_threshold:
            continue
        if top_k is not None and len(refined) >= top_k:
            break

        if param_name not in original_space:
            continue

        original = original_space[param_name]
        if not isinstance(original, (tuple, list)) or len(original) not in (2, 3):
            refined[param_name] = original
            continue

        trial_values = [t.params.get(param_name) for t in study.trials if param_name in t.params]
        if not trial_values:
            refined[param_name] = original
            continue

        if isinstance(trial_values[0], float):
            sorted_vals = sorted([v for v in trial_values if v is not None])
            q25, q75 = np.percentile(sorted_vals, [25, 75])
            span = q75 - q25
            new_low = max(original[0], q25 - expansion_factor * span)
            new_high = min(original[1], q75 + expansion_factor * span)
            kind = original[2] if len(original) == 3 else "linear"
            if kind != "linear":
                refined[param_name] = (new_low, new_high, kind)
            else:
                refined[param_name] = (new_low, new_high)
        elif isinstance(trial_values[0], int):
            sorted_vals = sorted([v for v in trial_values if v is not None])
            q25, q75 = (int(v) for v in np.percentile(sorted_vals, [25, 75]))
            span = max(1, q75 - q25)
            new_low = max(original[0], int(q25 - expansion_factor * span))
            new_high = min(original[1], int(q75 + expansion_factor * span))
            refined[param_name] = (new_low, new_high)
        else:
            refined[param_name] = original

    for param_name, val in original_space.items():
        if param_name not in refined:
            refined[param_name] = val

    return refined
