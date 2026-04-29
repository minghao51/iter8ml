from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tabular_blueprint.pipelines.nodes.model_training import ModelResult
from tabular_blueprint.services.registry_service import RegistryService
from tabular_blueprint.services.report_service import metric_value_is_better


@dataclass
class TrainingState:
    results: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    best_model: str | None
    best_score: float | None
    best_metric: str | None


def training_state(
    training_results: list[ModelResult],
    baseline_scores: dict[str, dict[str, float]],
    metrics: list[str],
    run_id: str,
    experiment_name: str,
    task: str,
    workspace_dir: str,
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
        registry = RegistryService(f"{workspace_dir}/registry.json")
        artifact = results.get(best_model, {}).get("artifact_path", "")
        registry.update_if_better(
            f"{experiment_name}:{task}",
            best_model,
            run_id,
            best_score,
            artifact,
            metric_name=primary_metric,
        )

    leaderboard.sort(
        key=lambda x: x.get("score", 0),
        reverse=(primary_metric not in {"rmse", "mae", "log_loss"}) if primary_metric else True,
    )

    return TrainingState(
        results=results,
        leaderboard=leaderboard,
        best_model=best_model,
        best_score=best_score,
        best_metric=primary_metric,
    )
