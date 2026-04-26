"""Base class for GBDT models with common functionality."""

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseGBDTModel:
    """Base class for gradient boosting decision tree models.

    Provides common initialization, save/load, and prediction methods.
    Subclasses must implement _build_params and _create_model.
    """

    def __init__(self, task: str = "classification", **kwargs: Any):
        self.task = task
        self.params = kwargs
        self._model: Any = None

    @abstractmethod
    def _build_params(self) -> dict:
        """Build model-specific parameters."""
        pass

    @abstractmethod
    def _create_model(self, params: dict) -> Any:
        """Create the underlying model instance."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        params = self._build_params()
        self._model = self._create_model(params)
        self._train_model(X, y)

    @abstractmethod
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task == "classification" and self._model is not None:
            return self._predict_proba_impl(X)
        return None

    @abstractmethod
    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        """Implementation of probability prediction."""
        pass

    def save(self, path: str) -> None:
        if self._model is None:
            raise ValueError("Model has not been trained yet.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(path)

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from path."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
