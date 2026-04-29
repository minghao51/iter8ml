"""Model training orchestration: sequential, concurrent, and single model training."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from tabular_blueprint.config import ExperimentConfig, HardwareProfile
from tabular_blueprint.engine.calibration import CalibratedModel
from tabular_blueprint.engine.evaluator import Evaluator
from tabular_blueprint.engine.tracker import Tracker
from tabular_blueprint.exceptions import ModelFitError, track_errors
from tabular_blueprint.models.factory import get_model_class
from tabular_blueprint.services.registry_service import RegistryService
from tabular_blueprint.services.report_service import metric_value_is_better


class ModelTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        tracker: Tracker,
        hardware: HardwareProfile,
    ):
        self.config = config
        self.tracker = tracker
        self.hardware = hardware

    def run_baselines(
        self,
        X: np.ndarray,
        y: np.ndarray,
        evaluator: Evaluator,
        run_id: str,
        data_hash: str,
        n_rows: int,
        n_features: int,
        baseline_models: dict[str, type[Any]],
    ) -> dict[str, dict[str, float]]:
        baseline_scores: dict[str, dict[str, float]] = {}
        for baseline_name, baseline_cls in baseline_models.items():
            start = time.time()
            try:
                cv_scores = evaluator.evaluate(baseline_cls, X, y, task=self.config.task.value)
                baseline_scores[baseline_name] = cv_scores
                self.tracker.log_event(
                    {
                        "event": "baseline_completed",
                        "run_id": run_id,
                        "model": baseline_name,
                        "task": self.config.task.value,
                        "data_hash": data_hash,
                        "cv_scores": cv_scores,
                        "duration_seconds": round(time.time() - start, 2),
                    }
                )
            except Exception as e:
                self.tracker.log_event(
                    {
                        "event": "baseline_failed",
                        "model": baseline_name,
                        "error": str(e),
                    }
                )
        return baseline_scores

    def train_all(
        self,
        models_to_run: list[str],
        X: np.ndarray,
        y: np.ndarray,
        evaluator: Evaluator,
        run_id: str,
        data_hash: str,
        n_rows: int,
        n_features: int,
        max_workers: int = 1,
        baseline_scores: dict[str, dict[str, float]] | None = None,
        feature_names: list[str] | None = None,
    ) -> dict:
        if max_workers == 1:
            return self._train_sequential(
                models_to_run,
                X,
                y,
                evaluator,
                run_id,
                data_hash,
                n_rows,
                n_features,
                baseline_scores or {},
                feature_names or [],
            )
        return self._train_concurrent(
            models_to_run,
            X,
            y,
            evaluator,
            run_id,
            data_hash,
            n_rows,
            n_features,
            max_workers,
            baseline_scores or {},
            feature_names or [],
        )

    def _train_sequential(
        self,
        models_to_run: list[str],
        X: np.ndarray,
        y: np.ndarray,
        evaluator: Evaluator,
        run_id: str,
        data_hash: str,
        n_rows: int,
        n_features: int,
        baseline_scores: dict[str, dict[str, float]],
        feature_names: list[str],
    ) -> dict:
        results = {}
        best_score = None
        primary_metric = self.config.metrics[0]

        for model_name in models_to_run:
            try:
                result = self._train_single_model(
                    model_name,
                    X,
                    y,
                    evaluator,
                    run_id,
                    data_hash,
                    n_rows,
                    n_features,
                    baseline_scores,
                    feature_names,
                )
                results[model_name] = result
            except ModelFitError as e:
                results[model_name] = {"error": str(e)}
                continue

            if "error" not in result:
                score = result.get("cv_scores", {}).get(primary_metric, 0)
                if best_score is None or metric_value_is_better(primary_metric, score, best_score):
                    best_score = score
                    self._update_champion(
                        result["model_name"],
                        run_id,
                        score,
                        result["artifact_path"],
                        primary_metric,
                    )

        return results

    def _train_concurrent(
        self,
        models_to_run: list[str],
        X: np.ndarray,
        y: np.ndarray,
        evaluator: Evaluator,
        run_id: str,
        data_hash: str,
        n_rows: int,
        n_features: int,
        max_workers: int,
        baseline_scores: dict[str, dict[str, float]],
        feature_names: list[str],
    ) -> dict:
        results = {}
        best_score = None
        primary_metric = self.config.metrics[0]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._train_single_model,
                    model_name,
                    X,
                    y,
                    evaluator,
                    run_id,
                    data_hash,
                    n_rows,
                    n_features,
                    baseline_scores,
                    feature_names,
                ): model_name
                for model_name in models_to_run
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result

                    if "error" not in result:
                        score = result.get("cv_scores", {}).get(primary_metric, 0)
                        if best_score is None or metric_value_is_better(
                            primary_metric, score, best_score
                        ):
                            best_score = score
                            self._update_champion(
                                result["model_name"],
                                run_id,
                                score,
                                result["artifact_path"],
                                primary_metric,
                            )
                except ModelFitError as e:
                    results[model_name] = {"error": str(e)}
                except Exception as e:
                    results[model_name] = {"error": str(e)}

        return results

    @track_errors()
    def _train_single_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        evaluator: Evaluator,
        run_id: str,
        data_hash: str,
        n_rows: int,
        n_features: int,
        baseline_scores: dict[str, dict[str, float]],
        feature_names: list[str],
    ) -> dict:
        start = time.time()
        try:
            model_cls = get_model_class(model_name)

            cv_scores = evaluator.evaluate(model_cls, X, y, task=self.config.task.value)

            if model_name == "ft_transformer":
                n_classes_val = (
                    len(np.unique(y)) if self.config.task.value == "classification" else 1
                )
                model = model_cls(
                    task=self.config.task.value,
                    n_features=n_features,
                    n_classes=n_classes_val,
                )
            else:
                model = model_cls(task=self.config.task.value)

            if self.config.calibration != "none" and self.config.task.value == "classification":
                model = CalibratedModel(model, method=self.config.calibration)
                cal_result = model.fit(X, y)

                self.tracker.log_event(
                    {
                        "event": "calibration_applied",
                        "run_id": run_id,
                        "model": model_name,
                        "method": cal_result.method,
                        "applied": cal_result.applied,
                    }
                )
            else:
                model.fit(X, y)

            artifact_path = str(self.config.workspace_dir / "artifacts" / f"{model_name}_{run_id}")
            model.save(artifact_path)

            duration = time.time() - start

            lift_over_baselines = {}
            if baseline_scores:
                primary_metric = self.config.metrics[0]
                model_score = cv_scores.get(primary_metric, 0)
                for bl_name, bl_scores in baseline_scores.items():
                    bl_score = bl_scores.get(primary_metric, 0)
                    if bl_score != 0:
                        if primary_metric in {"rmse", "mae", "log_loss"}:
                            lift_over_baselines[f"lift_over_{bl_name}"] = round(
                                (bl_score - model_score) / abs(bl_score), 4
                            )
                        else:
                            lift_over_baselines[f"lift_over_{bl_name}"] = round(
                                (model_score - bl_score) / abs(bl_score), 4
                            )

            event: dict[str, Any] = {
                "event": "model_completed",
                "run_id": run_id,
                "model": model.model_name,
                "task": self.config.task.value,
                "params": self._extract_model_params(model),
                "data_hash": data_hash,
                "n_rows": n_rows,
                "n_features": n_features,
                "cv_scores": cv_scores,
                "duration_seconds": round(duration, 2),
                "artifact_path": artifact_path,
                "hardware": {
                    "device": "cuda" if self.hardware.has_gpu else "cpu",
                    "vram_used_gb": 0.0,
                },
            }

            if lift_over_baselines:
                event["lift_over_baselines"] = lift_over_baselines

            self.tracker.log_event(event)
            self.tracker.log_metrics(cv_scores)

            return {
                "model_name": model.model_name,
                "cv_scores": cv_scores,
                "artifact_path": artifact_path,
                "duration_seconds": round(duration, 2),
                "lift_over_baselines": lift_over_baselines if lift_over_baselines else None,
            }

        except Exception as e:
            self.tracker.log_event(
                {
                    "event": "model_failed",
                    "model": model_name,
                    "error": str(e),
                }
            )
            raise ModelFitError(
                f"Model {model_name} failed during training",
                context={"model": model_name, "error": str(e)},
            ) from e

    def _extract_model_params(self, model: Any) -> dict[str, Any]:
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

    def _update_champion(
        self,
        model_name: str,
        run_id: str,
        score: float,
        artifact_path: str,
        metric_name: str,
    ) -> bool:
        registry = RegistryService(str(self.config.workspace_dir / "registry.json"))
        return registry.update_if_better(
            f"{self.config.name}:{self.config.task.value}",
            model_name,
            run_id,
            score,
            artifact_path,
            metric_name=metric_name,
        )
