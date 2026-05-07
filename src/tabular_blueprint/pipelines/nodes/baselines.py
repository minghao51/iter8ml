from __future__ import annotations

from typing import Any


def baseline_models() -> dict[str, type[Any]]:
    from tabular_blueprint.models.baselines import LinearBaseline, NaiveBaseline

    return {
        "naive_baseline": NaiveBaseline,
        "linear_baseline": LinearBaseline,
    }


def baseline_scores(
    data_prep_result: Any,
    baseline_models: dict[str, type[Any]],
    task: str,
    cv_folds: int,
    cv_strategy: str,
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    from tabular_blueprint.config import CVStrategy, ExperimentConfig
    from tabular_blueprint.constants import TaskType
    from tabular_blueprint.engine.evaluator import Evaluator

    X = data_prep_result.X
    y = data_prep_result.y

    config = ExperimentConfig(
        name="_baseline_eval",
        task=TaskType(task),
        target_col="_target",
        data_path="",
        cv_folds=cv_folds,
        cv_strategy=CVStrategy(cv_strategy),
        metrics=metrics,
    )
    evaluator = Evaluator(config)

    scores: dict[str, dict[str, float]] = {}
    for name, cls in baseline_models.items():
        try:
            cv_scores = evaluator.evaluate(cls, X, y, task=task)
            scores[name] = cv_scores
        except Exception:
            continue
    return scores
