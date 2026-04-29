from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

with contextlib.suppress(ImportError):
    from hamilton.function_modifiers import config


def _passthrough(data_prep_result: object) -> tuple[np.ndarray, list[str]]:
    return data_prep_result.X, data_prep_result.feature_names


def _fit_importance_model(
    X: np.ndarray,
    y: np.ndarray,
    models_to_run: list[str],
    task: str,
) -> Any:
    from tabular_blueprint.models.factory import get_model_class

    gbdt_priority = ["catboost", "lightgbm", "xgboost"]
    gbdt_name = next((m for m in gbdt_priority if m in models_to_run), None)
    if gbdt_name is not None:
        cls = get_model_class(gbdt_name)
        return cls(task=task).fit(X, y) or cls(task=task)

    from tabular_blueprint.models.baselines import LinearBaseline

    model = LinearBaseline(task=task)
    model.fit(X, y)
    return model


def _run_afe(
    X: np.ndarray,
    y: np.ndarray,
    models_to_run: list[str],
    feature_names: list[str],
    afe_top_k: int,
    afe_lift_threshold: float,
    afe_pruning: bool,
    afe_prune_min_importance: float,
    task: str,
    random_seed: int,
) -> tuple[np.ndarray, list[str]]:
    from tabular_blueprint.data.feature_engine import (
        discover_interactions,
        extract_top_k_features,
        prune_features,
    )

    imp_model = _fit_importance_model(X, y, models_to_run, task)
    top_k_indices = extract_top_k_features(
        imp_model,
        X,
        y,
        k=afe_top_k,
        feature_names=feature_names,
        task=task,
        random_seed=random_seed,
    )
    X_new, afe_result = discover_interactions(
        X,
        y,
        top_k_indices=top_k_indices,
        feature_names=feature_names,
        task=task,
        lift_threshold=afe_lift_threshold,
        random_seed=random_seed,
    )

    if X_new.shape[1] > 0:
        augmented_names = feature_names + afe_result.new_feature_names
        X_aug = np.hstack([X, X_new])
    else:
        augmented_names = feature_names
        X_aug = X

    if afe_pruning:
        imp_model2 = _fit_importance_model(X_aug, y, models_to_run, task)
        X_aug, pruning_result = prune_features(
            imp_model2,
            X_aug,
            y,
            feature_names=augmented_names,
            min_importance=afe_prune_min_importance,
            task=task,
            random_seed=random_seed,
        )
        augmented_names = [augmented_names[i] for i in pruning_result.kept_indices]

    return X_aug, augmented_names


@config.when_not(afe_enabled=True)
def training_features__default(
    data_prep_result: object,
) -> tuple[np.ndarray, list[str]]:
    return _passthrough(data_prep_result)


@config.when(afe_enabled=True)
def training_features__afe_enabled(
    data_prep_result: object,
    models_to_run: list[str],
    afe_top_k: int,
    afe_lift_threshold: float,
    afe_pruning: bool,
    afe_prune_min_importance: float,
    task: str,
    random_seed: int,
) -> tuple[np.ndarray, list[str]]:
    return _run_afe(
        data_prep_result.X,
        data_prep_result.y,
        models_to_run,
        data_prep_result.feature_names,
        afe_top_k,
        afe_lift_threshold,
        afe_pruning,
        afe_prune_min_importance,
        task,
        random_seed,
    )
