from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from iter8ml.exceptions import ModelFitError
from iter8ml.services.registry import RegistryService
from iter8ml.services.reporting import metric_sort_value, metric_value_is_better
from iter8ml.workspace import Workspace

_GBDT_MODEL_NAMES = frozenset({"catboost", "lightgbm", "xgboost"})
logger = logging.getLogger(__name__)


def _fold_indices_from_split(
    split_frame: pl.DataFrame, row_ids: list[str]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map surviving ``row_ids`` to train/validation index pairs from a split frame.

    The split frame carries one row per ``(row_id, fold)`` membership, so a single
    row_id appears in multiple folds. Group by fold, then translate surviving
    row_ids into positional indices of the engineered feature matrix.
    """
    index_by_row_id = {rid: i for i, rid in enumerate(row_ids)}
    train_idx_by_fold: dict[int, list[int]] = {}
    val_idx_by_fold: dict[int, list[int]] = {}
    for row in split_frame.iter_rows(named=True):
        idx = index_by_row_id.get(row["row_id"])
        if idx is None:
            continue
        if row["role"] == "train":
            train_idx_by_fold.setdefault(row["fold"], []).append(idx)
        elif row["role"] == "validation":
            val_idx_by_fold.setdefault(row["fold"], []).append(idx)
    folds = sorted(set(train_idx_by_fold) | set(val_idx_by_fold))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        out.append(
            (
                np.array(train_idx_by_fold.get(fold, []), dtype=int),
                np.array(val_idx_by_fold.get(fold, []), dtype=int),
            )
        )
    return out


@dataclass
class ModelResult:
    model_name: str
    input_name: str
    cv_scores: dict[str, float]
    artifact_path: str
    duration_seconds: float
    cv_std: dict[str, float] | None = None
    calibration_method: str | None = None
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
    split_frame: pl.DataFrame | None = None,
    random_seed: int = 42,
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
        random_seed=random_seed,
    )
    evaluator = Evaluator(config)
    fold_indices = None
    if split_frame is not None:
        fold_indices = _fold_indices_from_split(split_frame, data_prep_result.row_ids)
    scores: dict[str, dict[str, float]] = {}
    for name, cls in baseline_models.items():
        try:
            if fold_indices is not None:
                scores[name] = evaluator.evaluate_with_folds(
                    cls, data_prep_result.X, data_prep_result.y, fold_indices, task=task
                )
            else:
                scores[name] = evaluator.evaluate(
                    cls, data_prep_result.X, data_prep_result.y, task=task
                )
        except (ValueError, RuntimeError) as e:
            logger.warning("baseline %s skipped: %s", name, e)
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
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
    random_seed: int = 42,
    model_ctor_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
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
        random_seed=random_seed,
    )
    evaluator = Evaluator(config)
    ctor_kwargs = model_ctor_kwargs or {}
    if fold_indices is not None:
        return evaluator.evaluate_with_folds_and_std(
            model_cls, X, y, fold_indices, task=task, **ctor_kwargs
        )
    return evaluator.evaluate_with_std(model_cls, X, y, task=task, **ctor_kwargs)


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
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
    random_seed: int = 42,
    primary_metric: str | None = None,
) -> ModelResult:
    start = time.time()
    n_features = X.shape[1]
    overrides = (model_overrides or {}).get(name)
    # User-provided overrides win over the config seed.
    effective_overrides = {"random_seed": random_seed, **(overrides or {})}
    # FT-Transformer's ctor takes an explicit signature (no **kwargs); its seed
    # flows through apply_overrides onto the full-data fit instead. Every other
    # model accepts **kwargs and reads random_seed from self.params.
    model_ctor_kwargs: dict[str, Any] = (
        {} if name == "ft_transformer" else {"random_seed": random_seed}
    )

    try:
        from iter8ml.engine.models.factory import get_model_class

        model_cls = get_model_class(name)
        cv_scores, cv_std = _evaluate_model(
            model_cls,
            X,
            y,
            task,
            cv_folds,
            cv_strategy,
            metrics,
            fold_indices=fold_indices,
            random_seed=random_seed,
            model_ctor_kwargs=model_ctor_kwargs,
        )

        if name == "ft_transformer":
            n_classes = len(np.unique(y)) if task == "classification" else 1
            model = model_cls(task=task, n_features=n_features, n_classes=n_classes)
        else:
            model = model_cls(task=task, random_seed=random_seed)

        if hasattr(model, "apply_overrides"):
            model.apply_overrides(effective_overrides)

        calibration_method: str | None = None
        if calibration != "none" and task == "classification":
            from iter8ml.engine.calibration import CalibratedModel

            model = CalibratedModel(model, method=calibration, random_seed=random_seed)  # type: ignore[arg-type]
        fit_result = model.fit(X, y)
        if calibration != "none" and task == "classification":
            if fit_result is not None and getattr(fit_result, "applied", False):
                calibration_method = fit_result.method
            else:
                logger.warning(
                    "Calibration method '%s' was requested but not applied for '%s' "
                    "(model may lack predict_proba); scores and artifact are uncalibrated.",
                    calibration,
                    name,
                )
        duration = time.time() - start

        artifact_path = str(workspace.artifacts_dir / f"{name}_{run_id}")
        model.save(artifact_path)

        lift: dict[str, float] = {}
        primary = primary_metric or (metrics[0] if metrics else None)
        if baseline_scores and primary:
            from iter8ml.engine.evaluator import Evaluator

            for bl_name, bl_scores in baseline_scores.items():
                value = Evaluator.compute_lift(cv_scores, bl_scores, primary)
                if value is not None:
                    lift[f"lift_over_{bl_name}"] = round(value, 4)
        return ModelResult(
            model_name=model.model_name,
            input_name=name,
            cv_scores=cv_scores,
            artifact_path=artifact_path,
            duration_seconds=round(duration, 2),
            cv_std=cv_std,
            calibration_method=calibration_method,
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
    split_frame: pl.DataFrame | None = None,
    random_seed: int = 42,
    primary_metric: str | None = None,
) -> list[ModelResult]:
    X, _ = training_features
    y = data_prep_result.y
    non_baseline = [
        name for name in models_to_run if name not in ("naive_baseline", "linear_baseline")
    ]
    fold_indices = None
    if split_frame is not None:
        fold_indices = _fold_indices_from_split(split_frame, data_prep_result.row_ids)

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
                    fold_indices=fold_indices,
                    random_seed=random_seed,
                    primary_metric=primary_metric,
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
                fold_indices=fold_indices,
                random_seed=random_seed,
                primary_metric=primary_metric,
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
    models_to_run: list[str] | None = None,
    primary_metric: str | None = None,
) -> TrainingState:
    results: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    best_model: str | None = None
    best_score: float | None = None
    # Single ranking metric shared by lift, leaderboard, and registry promotion
    # (config.primary_metric, defaulted to metrics[0] at parse time).
    primary = primary_metric or (metrics[0] if metrics else None)

    for r in training_results:
        key = r.input_name
        if r.error is not None:
            results[key] = {"error": r.error}
            continue

        entry = {
            "model_name": r.model_name,
            "cv_scores": r.cv_scores,
            "cv_std": r.cv_std or {},
            "calibration": r.calibration_method,
            "artifact_path": r.artifact_path,
            "duration_seconds": r.duration_seconds,
            "lift_over_baselines": r.lift_over_baselines,
            "params": r.params or {},
        }
        results[key] = entry
        leaderboard.append(
            {
                "model": r.model_name,
                "score": r.cv_scores.get(primary, 0) if primary else 0,
                "std": (r.cv_std or {}).get(primary) if primary else None,
                "metric": primary,
                "calibration": r.calibration_method,
            }
        )
        if primary:
            score = r.cv_scores.get(primary, 0)
            if best_score is None or metric_value_is_better(primary, score, best_score):
                best_score = score
                best_model = key

    for bl_name, bl_scores in baseline_scores.items():
        results[bl_name] = {"cv_scores": bl_scores, "is_baseline": True}

    # Fail the run when every requested model failed: an empty leaderboard that
    # still exits 0 is the worst failure mode a reporting harness can have.
    requested = [m for m in (models_to_run or []) if m not in ("naive_baseline", "linear_baseline")]
    if requested:
        succeeded = [r for r in training_results if r.error is None]
        if not succeeded:
            first_error = next((r.error for r in training_results if r.error), "unknown error")
            raise ModelFitError(
                f"All {len(requested)} requested models failed to train. "
                f"First error: {first_error}",
                context={"run_id": run_id, "requested_models": requested},
            )

    if best_model and primary:
        registry = RegistryService(workspace)
        artifact = results.get(best_model, {}).get("artifact_path", "")
        registry.update_if_better(
            f"{experiment_name}:{task}",
            best_model,
            run_id,
            best_score if best_score is not None else 0.0,
            artifact,
            metric_name=primary,
        )

    leaderboard.sort(
        key=lambda x: (
            metric_sort_value(primary, x.get("score", 0)) if primary else x.get("score", 0)
        ),
        reverse=True,
    )
    return TrainingState(
        results=results,
        leaderboard=leaderboard,
        best_model=best_model,
        best_score=best_score,
        best_metric=primary,
    )
