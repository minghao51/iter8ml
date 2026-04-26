"""TabPFN v2 wrapper with row-count and GPU guardrails."""

from typing import Any

import numpy as np


class DataSizeError(ValueError):
    pass


class GPUUnavailableError(RuntimeError):
    pass


class TabPFNModel:
    DEFAULT_MAX_ROWS = 50_000

    def __init__(self, task: str = "classification", max_rows: int | None = None, **kwargs: Any):
        self.task = task
        self.max_rows = max_rows or self.DEFAULT_MAX_ROWS
        self.params = kwargs
        self.model: Any = None

    def _check_gpu(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                raise GPUUnavailableError(
                    "TabPFN requires a CUDA GPU but none was detected. "
                    "Use CatBoost or LightGBM instead (auto-fallback recommended)."
                )
        except ImportError:
            raise GPUUnavailableError(
                "PyTorch is not installed. TabPFN requires PyTorch with CUDA support."
            ) from None

    def _build_model(self) -> Any:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        if self.task == "classification":
            return TabPFNClassifier(
                device=self.params.get("device", "cuda"),
                random_state=self.params.get("random_seed", 42),
            )
        return TabPFNRegressor(
            device=self.params.get("device", "cuda"),
            random_state=self.params.get("random_seed", 42),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        if len(X) > self.max_rows:
            raise DataSizeError(
                f"TabPFN supports max {self.max_rows} rows, got {len(X)}. "
                f"Use CatBoost/LightGBM for larger datasets."
            )
        self._check_gpu()
        self.model = self._build_model()
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if (
            self.task == "classification"
            and self.model is not None
            and hasattr(self.model, "predict_proba")
        ):
            return self.model.predict_proba(X)
        return None

    def save(self, path: str) -> None:
        from tabular_blueprint.utils.safe_pickle import safe_dump

        if self.model is None:
            raise ValueError("Model not fitted")
        safe_dump(
            {"model": self.model, "task": self.task, "params": self.params},
            path,
        )

    def load(self, path: str) -> None:
        from tabular_blueprint.utils.safe_pickle import safe_load_file

        data = safe_load_file(path)
        self.model = data["model"]
        self.task = data.get("task", self.task)
        self.params = data.get("params", self.params)

    @property
    def model_name(self) -> str:
        return "TabPFN"
