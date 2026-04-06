"""TabPFN v2 wrapper with row-count guardrail."""

from pathlib import Path

import numpy as np


class DataSizeError(ValueError):
    """Raised when dataset exceeds model size limits."""


class TabPFNModel:
    """
    TabPFN v2 wrapper with hard row-count guardrail.
    Raises DataSizeError if n_rows > 10,000.
    """

    MAX_ROWS = 10_000

    def __init__(self, task: str = "classification", **kwargs):
        self.task = task
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
        if len(X) > self.MAX_ROWS:
            raise DataSizeError(
                f"TabPFN supports max {self.MAX_ROWS} rows, got {len(X)}. "
                "Use CatBoost or LightGBM for larger datasets."
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
