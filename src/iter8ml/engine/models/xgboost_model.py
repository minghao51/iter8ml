from typing import Any

import numpy as np
import xgboost as xgb

from iter8ml.engine.models.gbdt_base import BaseGBDTModel


class XGBoostModel(BaseGBDTModel):
    def _build_params(self) -> dict[str, Any]:
        if self.task == "classification":
            n_cls = getattr(self, "_n_classes", 2)
            if n_cls > 2:
                objective = "multi:softprob"
                eval_metric = "mlogloss"
            else:
                objective = "binary:logistic"
                eval_metric = "auc"
        else:
            objective = "reg:squarederror"
            eval_metric = "rmse"
        base = {
            "objective": objective,
            "eval_metric": eval_metric,
            "verbosity": 0,
            "seed": self.params.get("random_seed", 42),
            "tree_method": "hist",
        }
        if objective == "multi:softprob":
            base["num_class"] = n_cls
        # Pin threads to the OMP cap before user overrides can take effect.
        base.setdefault("nthread", self._default_thread_count())
        base.update(self.params)
        return base

    def _create_model(self, params: dict[str, Any]) -> xgb.Booster:
        return xgb.Booster(params=params)

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        params = self._build_params()
        dtrain = xgb.DMatrix(X, label=y)
        self._model = xgb.train(params, dtrain, num_boost_round=params.get("n_estimators", 1000))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        dtest = xgb.DMatrix(X)
        preds = self._model.predict(dtest)
        if self.task == "classification":
            return self._classify_predictions(preds)
        return preds  # type: ignore[no-any-return]

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        preds = self._model.predict(dtest)
        return self._format_proba(preds)

    def load(self, path: str) -> None:
        self._model = xgb.Booster()
        self._model.load_model(path)
        self._fitted = True

    @property
    def model_name(self) -> str:
        return "XGBoost"
