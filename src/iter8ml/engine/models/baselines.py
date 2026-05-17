"""Smart baseline models: Naive (Mean/Mode) and Linear (Logistic/Ridge)."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


class NaiveBaseline:
    """Predicts the mean (regression) or mode (classification) for all samples."""

    def __init__(self, task: str = "classification", **kwargs: Any):
        self.task = task
        self._value: float | Any | None = None
        self._classes: list[int] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        if self.task == "classification":
            classes, counts = np.unique(y, return_counts=True)
            self._classes = classes.tolist()
            self._value = classes[np.argmax(counts)]
        else:
            self._value = float(np.mean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._value is None:
            raise ValueError("Model not fitted")
        return np.full(X.shape[0], self._value)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task != "classification" or self._value is None:
            return None
        classes = self._classes or [0, 1]
        n_classes = len(classes)
        proba = np.zeros((X.shape[0], n_classes))
        try:
            idx = classes.index(int(self._value))
            proba[:, idx] = 1.0
        except (ValueError, TypeError):
            proba[:, 0] = 1.0
        return proba

    def save(self, path: str) -> None:
        np.savez(
            path,
            value=np.array([self._value]),
            task=np.array([self.task]),
            classes=np.array(self._classes) if self._classes else np.array([]),
        )

    def load(self, path: str) -> None:
        normalized = path if path.endswith(".npz") else path + ".npz"
        data = np.load(normalized, allow_pickle=False)
        self._value = data["value"][0]
        self.task = str(data["task"][0])
        cls_arr = data.get("classes")
        self._classes = cls_arr.tolist() if cls_arr is not None and cls_arr.size > 0 else None

    @property
    def model_name(self) -> str:
        return "NaiveBaseline"


class LinearBaseline:
    """LogisticRegression for classification, Ridge for regression."""

    def __init__(self, task: str = "classification", **kwargs: Any):
        self.task = task
        self._model: LogisticRegression | Ridge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        if self.task == "classification":
            self._model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if (
            self.task == "classification"
            and self._model is not None
            and hasattr(self._model, "predict_proba")
        ):
            return self._model.predict_proba(X)  # type: ignore[no-any-return]
        return None

    def save(self, path: str) -> None:
        from iter8ml.utils.io import safe_dump

        safe_dump(self._model, path + ".pkl")

    def load(self, path: str) -> None:
        from iter8ml.utils.io import safe_load_file

        self._model = safe_load_file(path + ".pkl")

    @property
    def model_name(self) -> str:
        return "LinearBaseline"
