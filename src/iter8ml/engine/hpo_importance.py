"""Hyperparameter importance analysis and search space refinement using Optuna."""

from datetime import UTC, datetime

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
