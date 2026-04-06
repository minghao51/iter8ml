"""CatBoost model wrapper with native categorical support."""

from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor


class CatBoostModel:
    def __init__(
        self,
        task: str = "classification",
        cat_features: list[int] | None = None,
        **kwargs,
    ):
        self.task = task
        self.cat_features = cat_features
        self.params = kwargs
        self.model = None

    def _build_model(self):
        seed = self.params.get("random_seed", 42)
        kwargs = {k: v for k, v in self.params.items() if k != "random_seed"}
        if self.task == "classification":
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

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model = self._build_model()
        self.model.fit(X, y, cat_features=self.cat_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task == "classification":
            return self.model.predict_proba(X)
        return None

    def save(self, path: str) -> None:
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
