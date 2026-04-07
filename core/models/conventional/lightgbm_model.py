import lightgbm as lgb
import numpy as np

from core.models.gbdt_base import BaseGBDTModel


class LightGBMModel(BaseGBDTModel):
    def _build_params(self) -> dict:
        base = {
            "objective": "binary" if self.task == "classification" else "regression",
            "metric": "auc" if self.task == "classification" else "rmse",
            "verbose": -1,
            "seed": self.params.get("random_seed", 42),
        }
        base.update(self.params)
        return base

    def _create_model(self, params: dict):
        return (
            lgb.LGBMClassifier(**params)
            if self.task == "classification"
            else lgb.LGBMRegressor(**params)
        )

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        params = self._build_params()
        train_data = lgb.Dataset(X, label=y)
        self._model = lgb.train(
            params, train_data, num_boost_round=params.get("n_estimators", 1000)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._model.predict(X)
        if self.task == "classification":
            return (preds >= 0.5).astype(int)
        return preds

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        preds = self._model.predict(X)
        return np.column_stack([1 - preds, preds])

    def load(self, path: str) -> None:
        self._model = lgb.Booster(model_file=path)

    @property
    def model_name(self) -> str:
        return "LightGBM"
