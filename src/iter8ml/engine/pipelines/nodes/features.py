from __future__ import annotations

from typing import Any

import numpy as np

try:
    from hamilton.function_modifiers import config

    _HAS_HAMILTON = True
except ImportError:
    _HAS_HAMILTON = False

if not _HAS_HAMILTON:
    from unittest.mock import MagicMock

    config = MagicMock()


def _passthrough(data_prep_result: Any) -> tuple[np.ndarray, list[str]]:
    return data_prep_result.X, data_prep_result.feature_names


def _fit_importance_model(
    X: np.ndarray,
    y: np.ndarray,
    models_to_run: list[str],
    task: str,
) -> Any:
    from iter8ml.engine.models.factory import get_model_class

    for name in ("catboost", "lightgbm", "xgboost"):
        if name in models_to_run:
            cls = get_model_class(name)
            return cls(task=task).fit(X, y) or cls(task=task)

    from iter8ml.engine.models.baselines import LinearBaseline

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
    from iter8ml.data.features import (
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


def _run_embedding(
    data_prep_result: Any,
    task: str,
    random_seed: int,
    run_id: str,
    workspace_dir: str,
    embedding_method: str = "entity",
    embedding_dim: int = 16,
    embedding_max_categories: int = 50,
    embedding_epochs: int = 10,
    embedding_lr: float = 1e-3,
    embedding_mlp_width: int = 128,
    embedding_mlp_depth: int = 2,
    embedding_ae_latent_dim: int = 32,
    embedding_ae_dropout: float = 0.2,
) -> tuple[np.ndarray, list[str]]:
    from iter8ml.data.embedding import EmbeddingEngine

    engine = EmbeddingEngine(
        task=task,
        workspace_dir=workspace_dir,
        embedding_method=embedding_method,
        embedding_dim=embedding_dim,
        embedding_max_categories=embedding_max_categories,
        embedding_epochs=embedding_epochs,
        embedding_lr=embedding_lr,
        embedding_mlp_width=embedding_mlp_width,
        embedding_mlp_depth=embedding_mlp_depth,
        embedding_ae_latent_dim=embedding_ae_latent_dim,
        embedding_ae_dropout=embedding_ae_dropout,
        random_seed=random_seed,
    )
    return engine.fit_transform(
        df=getattr(data_prep_result, "_df", None),
        X=data_prep_result.X,
        y=data_prep_result.y,
        feature_names=data_prep_result.feature_names,
        target_col="_target_",
        run_id=run_id,
    )


if _HAS_HAMILTON:

    @config.when(feature_strategy="none")
    def training_features__none(data_prep_result: Any) -> tuple[np.ndarray, list[str]]:
        return _passthrough(data_prep_result)

    @config.when(feature_strategy="afe")
    def training_features__afe(
        data_prep_result: Any,
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

    @config.when(feature_strategy="embedding")
    def training_features__embedding(
        data_prep_result: Any,
        task: str,
        random_seed: int,
        run_id: str,
        workspace_dir: str,
        embedding_method: str = "entity",
        embedding_dim: int = 16,
        embedding_max_categories: int = 50,
        embedding_epochs: int = 10,
        embedding_lr: float = 1e-3,
        embedding_mlp_width: int = 128,
        embedding_mlp_depth: int = 2,
        embedding_ae_latent_dim: int = 32,
        embedding_ae_dropout: float = 0.2,
    ) -> tuple[np.ndarray, list[str]]:
        return _run_embedding(
            data_prep_result,
            task,
            random_seed,
            run_id,
            workspace_dir,
            embedding_method,
            embedding_dim,
            embedding_max_categories,
            embedding_epochs,
            embedding_lr,
            embedding_mlp_width,
            embedding_mlp_depth,
            embedding_ae_latent_dim,
            embedding_ae_dropout,
        )
