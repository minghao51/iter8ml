from typing import Any

import lightgbm as lgb
import numpy as np

from iter8ml.engine.models.gbdt_base import BaseGBDTModel


class LightGBMModel(BaseGBDTModel):
    def _build_params(self) -> dict[str, Any]:
        if self.task == "classification":
            n_cls = getattr(self, "_n_classes", 2)
            if n_cls > 2:
                objective = "multiclass"
                metric = "multi_logloss"
            else:
                objective = "binary"
                metric = "auc"
        else:
            objective = "regression"
            metric = "rmse"
        base = {
            "objective": objective,
            "metric": metric,
            "verbose": -1,
            "seed": self.params.get("random_seed", 42),
        }
        if objective == "multiclass":
            base["num_class"] = n_cls
        base.update(self.params)
        return base

    def _create_model(self, params: dict[str, Any]) -> Any:
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
        if self._model is None:
            raise ValueError("Model not fitted")
        preds = self._model.predict(X)
        if self.task == "classification":
            return self._classify_predictions(preds)
        return preds  # type: ignore[no-any-return]

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
        preds = self._model.predict(X)
        return self._format_proba(preds)

    def load(self, path: str) -> None:
        self._model = lgb.Booster(model_file=path)

    @property
    def model_name(self) -> str:
        return "LightGBM"
