"""AbstractModel Protocol for structural subtyping."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AbstractModel(Protocol):
    """Contract for all model implementations.

    Error-handling guarantees:
    - ``predict()``  raises :class:`ModelNotFittedError` when called before ``fit()``.
    - ``predict_proba()`` returns ``None`` when the model is not fitted *or* when
      the task/mode does not support probability outputs (e.g. regression).
      It must **never** raise for an unfitted model.
    - ``save()``    raises :class:`ModelNotFittedError` when called before ``fit()``.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

    @property
    def model_name(self) -> str: ...
