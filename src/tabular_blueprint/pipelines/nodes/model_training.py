from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelResult:
    model_name: str
    input_name: str
    cv_scores: dict[str, float]
    artifact_path: str
    duration_seconds: float
    lift_over_baselines: dict[str, float] | None = None
    params: dict | None = None
    error: str | None = None


def _evaluate_model(
    model_cls: type,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv_folds: int,
    cv_strategy: str,
    metrics: list[str],
) -> dict[str, float]:
    from tabular_blueprint.config import CVStrategy, ExperimentConfig
    from tabular_blueprint.constants import TaskType
    from tabular_blueprint.engine.evaluator import Evaluator

    config = ExperimentConfig(
        name="_eval",
        task=TaskType(task),
        target_col="_t",
        data_path="",
        cv_folds=cv_folds,
        cv_strategy=CVStrategy(cv_strategy),
        metrics=metrics,
    )
    evaluator = Evaluator(config)
    return evaluator.evaluate(model_cls, X, y, task=task)


def _extract_params(model: object) -> dict:
    params = getattr(model, "params", None)
    if isinstance(params, dict):
        return params
    base_model = getattr(model, "base_model", None)
    base_params = getattr(base_model, "params", None)
    if isinstance(base_params, dict):
        extracted = dict(base_params)
        calibration_method = getattr(model, "method", None)
        if isinstance(calibration_method, str):
            extracted["calibration"] = calibration_method
        return extracted
    return {}


def _train_one(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv_folds: int,
    cv_strategy: str,
    metrics: list[str],
    calibration: str,
    workspace_dir: str,
    run_id: str,
    baseline_scores: dict[str, dict[str, float]],
    model_overrides: dict[str, dict[str, Any]] | None = None,
) -> ModelResult:
    start = time.time()
    n_features = X.shape[1]
    overrides = (model_overrides or {}).get(name)

    try:
        from tabular_blueprint.models.factory import get_model_class

        model_cls = get_model_class(name)
        cv_scores = _evaluate_model(model_cls, X, y, task, cv_folds, cv_strategy, metrics)

        if name == "ft_transformer":
            n_classes = len(np.unique(y)) if task == "classification" else 1
            model = model_cls(task=task, n_features=n_features, n_classes=n_classes)
        else:
            model = model_cls(task=task)

        if overrides and hasattr(model, "apply_overrides"):
            model.apply_overrides(overrides)

        if calibration != "none" and task == "classification":
            from tabular_blueprint.engine.calibration import CalibratedModel

            model = CalibratedModel(model, method=calibration)  # type: ignore[arg-type]
            model.fit(X, y)
        else:
            model.fit(X, y)

        artifact_path = f"{workspace_dir}/artifacts/{name}_{run_id}"
        model.save(artifact_path)
        duration = time.time() - start

        lift: dict[str, float] = {}
        primary_metric = metrics[0] if metrics else None
        if baseline_scores and primary_metric:
            model_score = cv_scores.get(primary_metric, 0)
            for bl_name, bl_scores in baseline_scores.items():
                bl_score = bl_scores.get(primary_metric, 0)
                if bl_score != 0:
                    if primary_metric in {"rmse", "mae", "log_loss"}:
                        lift[f"lift_over_{bl_name}"] = round(
                            (bl_score - model_score) / abs(bl_score), 4
                        )
                    else:
                        lift[f"lift_over_{bl_name}"] = round(
                            (model_score - bl_score) / abs(bl_score), 4
                        )

        return ModelResult(
            model_name=model.model_name,
            input_name=name,
            cv_scores=cv_scores,
            artifact_path=artifact_path,
            duration_seconds=round(duration, 2),
            lift_over_baselines=lift if lift else None,
            params=_extract_params(model),
        )

    except Exception as e:
        return ModelResult(
            model_name=name,
            input_name=name,
            cv_scores={},
            artifact_path="",
            duration_seconds=round(time.time() - start, 2),
            error=str(e),
        )


def training_results(
    training_features: tuple[np.ndarray, list[str]],
    data_prep_result: Any,
    models_to_run: list[str],
    baseline_scores: dict[str, dict[str, float]],
    task: str,
    cv_folds: int,
    cv_strategy: str,
    metrics: list[str],
    calibration: str,
    workspace_dir: str,
    run_id: str,
    model_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[ModelResult]:
    X, _ = training_features
    y = data_prep_result.y

    results: list[ModelResult] = []
    for name in models_to_run:
        if name in ("naive_baseline", "linear_baseline"):
            continue
        result = _train_one(
            name,
            X,
            y,
            task,
            cv_folds,
            cv_strategy,
            metrics,
            calibration,
            workspace_dir,
            run_id,
            baseline_scores,
            model_overrides=model_overrides,
        )
        results.append(result)
    return results
