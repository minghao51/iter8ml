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
        self._n_classes: int = kwargs.pop("n_classes", 0)
        self._class_labels: np.ndarray | None = None

    @abstractmethod
    def _build_params(self) -> dict[str, Any]:
        """Build model-specific parameters."""
        pass

    @abstractmethod
    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create the underlying model instance."""
        pass

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge per-model hyperparameter overrides into self.params."""
        self.params.update(overrides)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        if self.task == "classification":
            labels, y_encoded = np.unique(y, return_inverse=True)
            self._class_labels = labels
            self._n_classes = self._n_classes or int(labels.size)
            y_train = y_encoded
        else:
            self._n_classes = 1
            y_train = y
        params = self._build_params()
        self._model = self._create_model(params)
        self._train_model(X, y_train)

    def _decode_class_indices(self, y_pred: np.ndarray) -> np.ndarray:
        if self.task != "classification" or self._class_labels is None:
            return y_pred
        idx = y_pred.astype(int)
        return self._class_labels[idx]

    def _classify_predictions(self, preds: np.ndarray) -> np.ndarray:
        n_cls = self._n_classes
        if n_cls > 2:
            return self._decode_class_indices(np.argmax(preds, axis=1))
        return self._decode_class_indices((preds >= 0.5).astype(int))

    def _format_proba(self, preds: np.ndarray) -> np.ndarray:
        n_cls = self._n_classes
        if n_cls > 2:
            return preds
        return np.column_stack([1 - preds, preds])

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
