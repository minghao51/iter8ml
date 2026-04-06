"""LightGBM model wrapper."""

from pathlib import Path

import lightgbm as lgb
import numpy as np


class LightGBMModel:
    def __init__(self, task: str = "classification", **kwargs):
        self.task = task
        self.params = kwargs
        self.model = None

    def _build_params(self) -> dict:
        base = {
            "objective": "binary" if self.task == "classification" else "regression",
            "metric": "auc" if self.task == "classification" else "rmse",
            "verbose": -1,
            "seed": self.params.get("random_seed", 42),
        }
        base.update(self.params)
        return base

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        params = self._build_params()
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, train_data, num_boost_round=params.get("n_estimators", 1000))

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        if self.task == "classification":
            return (preds >= 0.5).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task == "classification":
            preds = self.model.predict(X)
            return np.column_stack([1 - preds, preds])
        return None

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model = lgb.Booster(model_file=path)

    @property
    def model_name(self) -> str:
        return "LightGBM"
