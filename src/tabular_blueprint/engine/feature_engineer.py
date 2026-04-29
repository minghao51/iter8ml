"""Automated Feature Engineering as a standalone service."""

from typing import Any, ClassVar

import numpy as np

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.data.feature_engine import (
    discover_interactions,
    extract_top_k_features,
    prune_features,
)
from tabular_blueprint.engine.tracker import Tracker
from tabular_blueprint.models.baselines import LinearBaseline
from tabular_blueprint.models.factory import get_model_class


class FeatureEngineer:
    _GBDT_PRIORITY: ClassVar[list[str]] = ["catboost", "lightgbm", "xgboost"]

    def __init__(self, config: ExperimentConfig, tracker: Tracker):
        self.config = config
        self.tracker = tracker

    def run_afe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        run_id: str,
        data_hash: str,
        models_to_run: list[str],
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        importance_model = self._fit_importance_model(X, y, models_to_run)
        top_k_indices = extract_top_k_features(
            importance_model,
            X,
            y,
            k=self.config.afe_top_k,
            feature_names=feature_names,
            task=self.config.task.value,
            random_seed=self.config.random_seed,
        )

        X_new, afe_result = discover_interactions(
            X,
            y,
            top_k_indices=top_k_indices,
            feature_names=feature_names,
            task=self.config.task.value,
            lift_threshold=self.config.afe_lift_threshold,
            random_seed=self.config.random_seed,
        )

        self.tracker.log_event(
            {
                "event": "afe_completed",
                "run_id": run_id,
                "data_hash": data_hash,
                "n_candidates_tested": afe_result.n_candidates_tested,
                "n_candidates_kept": afe_result.n_candidates_kept,
                "new_feature_names": afe_result.new_feature_names,
            }
        )

        if X_new.shape[1] > 0:
            feature_names = feature_names + afe_result.new_feature_names
            X_augmented = np.hstack([X, X_new])
        else:
            X_augmented = X

        if self.config.afe_pruning:
            importance_model = self._fit_importance_model(X_augmented, y, models_to_run)
            X_augmented, pruning_result = prune_features(
                importance_model,
                X_augmented,
                y,
                feature_names=feature_names,
                min_importance=self.config.afe_prune_min_importance,
                task=self.config.task.value,
                random_seed=self.config.random_seed,
            )
            feature_names = [feature_names[i] for i in pruning_result.kept_indices]
            self.tracker.log_event(
                {
                    "event": "feature_pruning",
                    "run_id": run_id,
                    "data_hash": data_hash,
                    "n_original": pruning_result.n_original,
                    "n_kept": pruning_result.n_kept,
                    "n_dropped": pruning_result.n_dropped,
                    "dropped_features": pruning_result.dropped_features,
                }
            )

        return X_augmented, feature_names

    def _fit_importance_model(self, X: np.ndarray, y: np.ndarray, models_to_run: list[str]) -> Any:
        gbdt_name = next(
            (m for m in self._GBDT_PRIORITY if m in models_to_run),
            None,
        )
        if gbdt_name is not None:
            cls = get_model_class(gbdt_name)
            model = cls(task=self.config.task.value)
            model.fit(X, y)
            return model
        return self._get_best_baseline_model(X, y)

    def _get_best_baseline_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        model = LinearBaseline(task=self.config.task.value)
        model.fit(X, y)
        return model
