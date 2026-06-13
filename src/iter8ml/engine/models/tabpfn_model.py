"""TabPFN v2 wrapper with row-count guardrails and CPU fallback."""

import logging
from typing import Any

import numpy as np

from iter8ml.exceptions import ModelNotFittedError

logger = logging.getLogger(__name__)


class DataSizeError(ValueError):
    pass


class TabPFNModel:
    DEFAULT_MAX_ROWS = 50_000

    def __init__(self, task: str = "classification", max_rows: int | None = None, **kwargs: Any):
        self.task = task
        self.max_rows = max_rows or self.DEFAULT_MAX_ROWS
        self.params = kwargs
        self.model: Any = None
        self._fitted: bool = False

    def _resolve_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        logger.warning("No CUDA GPU detected — TabPFN will run on CPU (expect slower performance).")
        return "cpu"

    def _build_model(self) -> Any:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        device = self.params.get("device") or self._resolve_device()

        if self.task == "classification":
            return TabPFNClassifier(
                device=device,
                random_state=self.params.get("random_seed", 42),
            )
        return TabPFNRegressor(
            device=device,
            random_state=self.params.get("random_seed", 42),
        )

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge per-model hyperparameter overrides into self.params."""
        self.params.update(overrides)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        if len(X) > self.max_rows:
            raise DataSizeError(
                f"TabPFN supports max {self.max_rows} rows, got {len(X)}. "
                f"Use CatBoost/LightGBM for larger datasets."
            )
        self.model = self._build_model()
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise ModelNotFittedError("Model not fitted")
        return self.model.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self._fitted or self.model is None:
            return None
        if self.task != "classification":
            return None
        if not hasattr(self.model, "predict_proba"):
            return None
        return self.model.predict_proba(X)  # type: ignore[no-any-return]

    def save(self, path: str) -> None:
        from iter8ml.utils.io import safe_dump

        if not self._fitted or self.model is None:
            raise ModelNotFittedError("Model not fitted")
        safe_dump(
            {"model": self.model, "task": self.task, "params": self.params},
            path,
        )

    def load(self, path: str) -> None:
        from iter8ml.utils.io import safe_load_file

        data = safe_load_file(path)
        self.model = data["model"]
        self.task = data.get("task", self.task)
        self.params = data.get("params", self.params)
        self._fitted = True

    @property
    def model_name(self) -> str:
        return "TabPFN"
