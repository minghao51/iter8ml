"""Per-model configuration and default hyperparameter search spaces."""

from typing import Any

from pydantic import BaseModel, Field


class CatBoostConfig(BaseModel):
    """CatBoost hyperparameter configuration and HPO search space.

    Training defaults live in the wrapper (``catboost_model.py``); this config
    only carries ``random_seed`` and the HPO search space consumed by
    ``engine/hpo.py``.
    """

    random_seed: int = 42

    def hpo_search_space(self) -> dict[str, Any]:
        return {
            "depth": (4, 10),
            "learning_rate": (0.01, 0.2, "log"),
            "l2_leaf_reg": (1.0, 10.0, "log"),
            "iterations": (500, 3000),
        }


class LightGBMConfig(BaseModel):
    """LightGBM hyperparameter configuration and HPO search space.

    Training defaults live in the wrapper (``lightgbm_model.py``); this config
    only carries ``random_seed`` and the HPO search space consumed by
    ``engine/hpo.py``.
    """

    random_seed: int = 42

    def hpo_search_space(self) -> dict[str, Any]:
        return {
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.2, "log"),
            "num_leaves": (15, 127),
            "min_child_samples": (5, 50),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
        }


class XGBoostConfig(BaseModel):
    """XGBoost hyperparameter configuration and HPO search space.

    Training defaults live in the wrapper (``xgboost_model.py``); this config
    only carries ``random_seed`` and the HPO search space consumed by
    ``engine/hpo.py``.
    """

    random_seed: int = 42

    def hpo_search_space(self) -> dict[str, Any]:
        return {
            "max_depth": (3, 12),
            "learning_rate": (0.01, 0.2, "log"),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0.0, 5.0),
        }


class TabPFNConfig(BaseModel):
    """TabPFN hyperparameter configuration."""

    n_estimators: int = 4
    device: str = "cpu"
    random_seed: int = 42
    max_rows: int = 50_000

    def hpo_search_space(self) -> dict[str, Any]:
        return {}


class TabNetConfig(BaseModel):
    """TabNet hyperparameter configuration and HPO search space."""

    n_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    random_seed: int = 42

    def hpo_search_space(self) -> dict[str, Any]:
        return {
            "learning_rate": (1e-4, 1e-2, "log"),
            "batch_size": (64, 512),
            "n_epochs": (20, 100),
        }


class FTTransformerConfig(BaseModel):
    """FT-Transformer hyperparameter configuration and HPO search space."""

    n_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    n_heads: int = 4
    d_hidden: int = 128
    n_layers: int = 3
    dropout: float = 0.1
    random_seed: int = 42

    def hpo_search_space(self) -> dict[str, Any]:
        return {
            "learning_rate": (1e-5, 1e-3, "log"),
            "d_hidden": (64, 256),
            "n_heads": (2, 8),
            "n_layers": (2, 6),
            "dropout": (0.0, 0.3),
        }


class ModelConfigs(BaseModel):
    """Container for all per-model configuration objects."""

    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    tabpfn: TabPFNConfig = Field(default_factory=TabPFNConfig)
    ft_transformer: FTTransformerConfig = Field(default_factory=FTTransformerConfig)
    tabnet: TabNetConfig = Field(default_factory=TabNetConfig)
