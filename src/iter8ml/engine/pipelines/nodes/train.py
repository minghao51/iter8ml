from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from iter8ml.services.registry import RegistryService
from iter8ml.services.reporting import metric_sort_value, metric_value_is_better
from iter8ml.workspace import Workspace

_GBDT_MODEL_NAMES = frozenset({"catboost", "lightgbm", "xgboost"})


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


@dataclass
class TrainingState:
    results: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    best_model: str | None
    best_score: float | None
    best_metric: str | None


# ── model selection node ─────────────────────────────────────────────────


def models_to_run(
    data_prep_result: Any,
    task: str,
    vram_gb: float,
    config_models: Any,
    completed_models: list[str] | None = None,
    include_baselines: bool = True,
) -> list[str]:
    completed = set(completed_models or [])
    if isinstance(config_models, list):
        return [m for m in config_models if m not in completed]
    if config_models != "auto":
        return [config_models] if config_models not in completed else []

    from iter8ml.engine.models.selector import ModelSelector

    return [
        m
        for m in ModelSelector().select(
            n_rows=data_prep_result.n_rows,
            task=task,  # type: ignore[arg-type]
            vram_gb=vram_gb,
            include_baselines=include_baselines,
        )
        if m not in completed
    ]


# ── baseline nodes ───────────────────────────────────────────────────────


def baseline_models() -> dict[str, type[Any]]:
    from iter8ml.engine.models.baselines import LinearBaseline, NaiveBaseline

    return {"naive_baseline": NaiveBaseline, "linear_baseline": LinearBaseline}


def baseline_scores(
    data_prep_result: Any,
    baseline_models: dict[str, type[Any]],
    task: str,
    cv_folds: int,
    cv_strategy: str,
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    from iter8ml.config import CVStrategy, ExperimentConfig
    from iter8ml.constants import TaskType
    from iter8ml.engine.evaluator import Evaluator

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
            scores[name] = evaluator.evaluate(
                cls, data_prep_result.X, data_prep_result.y, task=task
            )
        except (ValueError, RuntimeError):
            continue
    return scores


# ── model training nodes ─────────────────────────────────────────────────


def _evaluate_model(
    model_cls: type,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv_folds: int,
    cv_strategy: str,
    metrics: list[str],
) -> dict[str, float]:
    from iter8ml.config import CVStrategy, ExperimentConfig
    from iter8ml.constants import TaskType
    from iter8ml.engine.evaluator import Evaluator

    config = ExperimentConfig(
        name="_eval",
        task=TaskType(task),
        target_col="_t",
        data_path="",
        cv_folds=cv_folds,
        cv_strategy=CVStrategy(cv_strategy),
        metrics=metrics,
    )
    return Evaluator(config).evaluate(model_cls, X, y, task=task)


def _extract_params(model: object) -> dict:
    params = getattr(model, "params", None)
    if isinstance(params, dict):
        return params
    base_params = getattr(getattr(model, "base_model", None), "params", None)
    if isinstance(base_params, dict):
        extracted = dict(base_params)
        method = getattr(model, "method", None)
        if isinstance(method, str):
            extracted["calibration"] = method
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
    workspace: Workspace,
    run_id: str,
    baseline_scores: dict[str, dict[str, float]],
    model_overrides: dict[str, dict[str, Any]] | None = None,
) -> ModelResult:
    start = time.time()
    n_features = X.shape[1]
    overrides = (model_overrides or {}).get(name)

    try:
        from iter8ml.engine.models.factory import get_model_class

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
            from iter8ml.engine.calibration import CalibratedModel

            model = CalibratedModel(model, method=calibration)  # type: ignore[arg-type]
        model.fit(X, y)

        artifact_path = str(workspace.artifacts_dir / f"{name}_{run_id}")
        model.save(artifact_path)
        duration = time.time() - start

        lift: dict[str, float] = {}
        primary_metric = metrics[0] if metrics else None
        if baseline_scores and primary_metric:
            from iter8ml.engine.evaluator import Evaluator

            lift = {
                f"lift_over_{bl_name}": round(
                    Evaluator.compute_lift(cv_scores, bl_scores, primary_metric), 4
                )
                for bl_name, bl_scores in baseline_scores.items()
            }
        return ModelResult(
            model_name=model.model_name,
            input_name=name,
            cv_scores=cv_scores,
            artifact_path=artifact_path,
            duration_seconds=round(duration, 2),
            lift_over_baselines=lift or None,
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
    workspace: Workspace,
    run_id: str,
    model_overrides: dict[str, dict[str, Any]] | None = None,
    max_workers: int = 1,
    strict_thread_safety: bool = True,
) -> list[ModelResult]:
    X, _ = training_features
    y = data_prep_result.y
    non_baseline = [
        name for name in models_to_run if name not in ("naive_baseline", "linear_baseline")
    ]

    effective_workers = _effective_training_workers(
        max_workers,
        non_baseline,
        strict_thread_safety=strict_thread_safety,
    )

    if effective_workers <= 1 or len(non_baseline) <= 1:
        results: list[ModelResult] = []
        for name in non_baseline:
            results.append(
                _train_one(
                    name,
                    X,
                    y,
                    task,
                    cv_folds,
                    cv_strategy,
                    metrics,
                    calibration,
                    workspace,
                    run_id,
                    baseline_scores,
                    model_overrides=model_overrides,
                )
            )
        return results

    results_dict: dict[str, ModelResult] = {}
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _train_one,
                name,
                X,
                y,
                task,
                cv_folds,
                cv_strategy,
                metrics,
                calibration,
                workspace,
                run_id,
                baseline_scores,
                model_overrides=model_overrides,
            ): name
            for name in non_baseline
        }
        for future in as_completed(futures):
            results_dict[futures[future]] = future.result()

    return [results_dict[name] for name in non_baseline]


def _effective_training_workers(
    max_workers: int,
    model_names: list[str],
    *,
    strict_thread_safety: bool = True,
) -> int:
    if max_workers <= 1 or len(model_names) <= 1:
        return 1
    if strict_thread_safety and any(name in _GBDT_MODEL_NAMES for name in model_names):
        return 1
    return min(int(max_workers), len(model_names))


# ── state generation node ────────────────────────────────────────────────


def training_state(
    training_results: list[ModelResult],
    baseline_scores: dict[str, dict[str, float]],
    metrics: list[str],
    run_id: str,
    experiment_name: str,
    task: str,
    workspace: Workspace,
) -> TrainingState:
    results: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    best_model: str | None = None
    best_score: float | None = None
    primary_metric = metrics[0] if metrics else None

    for r in training_results:
        key = r.input_name
        if r.error is not None:
            results[key] = {"error": r.error}
            continue

        entry = {
            "model_name": r.model_name,
            "cv_scores": r.cv_scores,
            "artifact_path": r.artifact_path,
            "duration_seconds": r.duration_seconds,
            "lift_over_baselines": r.lift_over_baselines,
            "params": r.params or {},
        }
        results[key] = entry
        leaderboard.append(
            {
                "model": r.model_name,
                "score": r.cv_scores.get(primary_metric, 0) if primary_metric else 0,
                "metric": primary_metric,
            }
        )
        if primary_metric:
            score = r.cv_scores.get(primary_metric, 0)
            if best_score is None or metric_value_is_better(primary_metric, score, best_score):
                best_score = score
                best_model = key

    for bl_name, bl_scores in baseline_scores.items():
        results[bl_name] = {"cv_scores": bl_scores, "is_baseline": True}

    if best_model and primary_metric:
        registry = RegistryService(workspace)
        artifact = results.get(best_model, {}).get("artifact_path", "")
        registry.update_if_better(
            f"{experiment_name}:{task}",
            best_model,
            run_id,
            best_score if best_score is not None else 0.0,
            artifact,
            metric_name=primary_metric,
        )

    leaderboard.sort(
        key=lambda x: (
            metric_sort_value(primary_metric, x.get("score", 0))
            if primary_metric
            else x.get("score", 0)
        ),
        reverse=True,
    )
    return TrainingState(
        results=results,
        leaderboard=leaderboard,
        best_model=best_model,
        best_score=best_score,
        best_metric=primary_metric,
    )
