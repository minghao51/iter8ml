"""Per-model configuration and default hyperparameter search spaces."""

from typing import Any

from pydantic import BaseModel, Field


class CatBoostConfig(BaseModel):
    iterations: int = 1000
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 50
    task_type: str = "CPU"
    random_seed: int = 42

    def hpo_search_space(self) -> dict[str, Any]:
        return {
            "depth": (4, 10),
            "learning_rate": (0.01, 0.2, "log"),
            "l2_leaf_reg": (1.0, 10.0, "log"),
            "iterations": (500, 3000),
        }


class LightGBMConfig(BaseModel):
    n_estimators: int = 1000
    max_depth: int = -1
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
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
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
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
    n_estimators: int = 4
    device: str = "cpu"
    random_seed: int = 42

    def hpo_search_space(self) -> dict[str, Any]:
        return {}


class FTTransformerConfig(BaseModel):
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
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    tabpfn: TabPFNConfig = Field(default_factory=TabPFNConfig)
    ft_transformer: FTTransformerConfig = Field(default_factory=FTTransformerConfig)
