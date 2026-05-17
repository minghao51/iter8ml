"""CatBoost model wrapper."""

from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor


class CatBoostModel:
    def __init__(
        self,
        task: str = "classification",
        cat_features: list[int] | None = None,
        n_classes: int = 0,
        **kwargs: Any,
    ):
        self.task = task
        self.cat_features = cat_features
        self._n_classes = n_classes
        self.params = kwargs
        self.model: CatBoostClassifier | CatBoostRegressor | None = None

    @staticmethod
    def _detect_gpu() -> bool:
        try:
            from catboost.utils import get_gpu_count

            return get_gpu_count() > 0  # type: ignore[no-any-return]
        except Exception:
            return False

    def _build_model(self) -> CatBoostClassifier | CatBoostRegressor:
        seed = self.params.get("random_seed", 42)
        kwargs = {k: v for k, v in self.params.items() if k != "random_seed"}
        task_type = kwargs.pop("task_type", "auto")
        if task_type == "auto":
            task_type = "GPU" if self._detect_gpu() else "CPU"
        kwargs.setdefault("task_type", task_type)
        if self.task == "classification":
            if self._n_classes and self._n_classes > 2:
                kwargs.setdefault("classes_count", self._n_classes)
            return CatBoostClassifier(
                cat_features=self.cat_features,
                verbose=False,
                random_seed=seed,
                **kwargs,
            )
        return CatBoostRegressor(
            cat_features=self.cat_features,
            verbose=False,
            random_seed=seed,
            **kwargs,
        )

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge per-model hyperparameter overrides into self.params."""
        self.params.update(overrides)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        self.model = self._build_model()
        self.model.fit(X, y, cat_features=self.cat_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task == "classification" and self.model is not None:
            return self.model.predict_proba(X)  # type: ignore[no-any-return]
        return None

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        if self.task == "classification":
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRegressor()
        self.model.load_model(path)

    @property
    def model_name(self) -> str:
        return "CatBoost"
