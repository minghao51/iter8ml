"""TabPFN v2 wrapper with row-count guardrail."""

from pathlib import Path

import numpy as np


class DataSizeError(ValueError):
    """Raised when dataset exceeds model size limits."""


class TabPFNModel:
    """
    TabPFN v2 wrapper with configurable row-count guardrail.

    Args:
        task: "classification" or "regression"
        max_rows: Maximum number of training rows. Default: 10_000.
        **kwargs: Additional parameters passed to TabPFN model.
    """

    DEFAULT_MAX_ROWS = 10_000

    def __init__(self, task: str = "classification", max_rows: int | None = None, **kwargs):
        self.task = task
        self.max_rows = max_rows or self.DEFAULT_MAX_ROWS
        self.params = kwargs
        self.model = None

    def _build_model(self):
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        if self.task == "classification":
            return TabPFNClassifier(
                device=self.params.get("device", "cpu"),
                random_state=self.params.get("random_seed", 42),
            )
        return TabPFNRegressor(
            device=self.params.get("device", "cpu"),
            random_state=self.params.get("random_seed", 42),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if len(X) > self.max_rows:
            raise DataSizeError(
                f"TabPFN supports max {self.max_rows} rows for this instance, got {len(X)}. "
                f"Increase max_rows parameter or use CatBoost/LightGBM for larger datasets."
            )
        self.model = self._build_model()
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, path: str) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        import torch

        self.model = self._build_model()
        self.model.load_state_dict(torch.load(path, weights_only=True))

    @property
    def model_name(self) -> str:
        return "TabPFN"
