"""XGBoost model wrapper."""

from pathlib import Path

import numpy as np
import xgboost as xgb


class XGBoostModel:
    def __init__(self, task: str = "classification", **kwargs):
        self.task = task
        self.params = kwargs
        self.model = None

    def _build_params(self) -> dict:
        base = {
            "objective": "binary:logistic" if self.task == "classification" else "reg:squarederror",
            "eval_metric": "auc" if self.task == "classification" else "rmse",
            "verbosity": 0,
            "seed": self.params.pop("random_seed", 42),
            "tree_method": "hist",
        }
        base.update(self.params)
        return base

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        params = self._build_params()
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(params, dtrain, num_boost_round=params.get("n_estimators", 1000))

    def predict(self, X: np.ndarray) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        if self.task == "classification":
            return (preds >= 0.5).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task == "classification":
            dtest = xgb.DMatrix(X)
            preds = self.model.predict(dtest)
            return np.column_stack([1 - preds, preds])
        return None

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        self.model = xgb.Booster()
        self.model.load_model(path)

    @property
    def model_name(self) -> str:
        return "XGBoost"
