import numpy as np
import xgboost as xgb

from tabular_blueprint.models.gbdt_base import BaseGBDTModel


class XGBoostModel(BaseGBDTModel):
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

    def _create_model(self, params: dict) -> xgb.Booster:
        return xgb.Booster(params=params)

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        params = self._build_params()
        dtrain = xgb.DMatrix(X, label=y)
        self._model = xgb.train(params, dtrain, num_boost_round=params.get("n_estimators", 1000))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
        dtest = xgb.DMatrix(X)
        preds = self._model.predict(dtest)
        if self.task == "classification":
            return (preds >= 0.5).astype(int)
        return preds

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
        dtest = xgb.DMatrix(X)
        preds = self._model.predict(dtest)
        return np.column_stack([1 - preds, preds])

    def load(self, path: str) -> None:
        self._model = xgb.Booster()
        self._model.load_model(path)

    @property
    def model_name(self) -> str:
        return "XGBoost"
