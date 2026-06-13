"""CatBoost model wrapper."""

import functools
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

from iter8ml.engine.models.gbdt_base import BaseGBDTModel


class CatBoostModel(BaseGBDTModel):
    def __init__(
        self,
        task: str = "classification",
        cat_features: list[int] | None = None,
        n_classes: int = 0,
        **kwargs: Any,
    ):
        super().__init__(task=task, n_classes=n_classes, **kwargs)
        self.cat_features = cat_features

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _detect_gpu() -> bool:
        try:
            from catboost.utils import get_gpu_count

            return get_gpu_count() > 0  # type: ignore[no-any-return]
        except Exception:
            return False

    def _build_params(self) -> dict[str, Any]:
        seed = self.params.get("random_seed", 42)
        kwargs = {k: v for k, v in self.params.items() if k != "random_seed"}
        task_type = kwargs.pop("task_type", "auto")
        if task_type == "auto":
            task_type = "GPU" if self._detect_gpu() else "CPU"
        kwargs.setdefault("task_type", task_type)
        if self.task == "classification" and self._n_classes and self._n_classes > 2:
            kwargs.setdefault("classes_count", self._n_classes)
        kwargs["verbose"] = False
        kwargs["random_seed"] = seed
        return kwargs

    def _create_model(self, params: dict[str, Any]) -> CatBoostClassifier | CatBoostRegressor:
        if self.task == "classification":
            return CatBoostClassifier(
                cat_features=self.cat_features,
                **params,
            )
        return CatBoostRegressor(
            cat_features=self.cat_features,
            **params,
        )

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y, cat_features=self.cat_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return self._model.predict(X).flatten()  # type: ignore[no-any-return]

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)  # type: ignore[no-any-return]

    def load(self, path: str) -> None:
        if self.task == "classification":
            self._model = CatBoostClassifier()
        else:
            self._model = CatBoostRegressor()
        self._model.load_model(path)
        self._fitted = True

    @property
    def model_name(self) -> str:
        return "CatBoost"
