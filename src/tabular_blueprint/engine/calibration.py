"""Probability calibration: Platt scaling and Isotonic regression."""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


class CalibrationResult(BaseModel):
    """Result of applying probability calibration (Platt/Isotonic)."""

    method: str
    n_classes: int
    applied: bool


class CalibratedModel:
    def __init__(
        self,
        base_model: Any,
        method: Literal["platt", "isotonic", "none"] = "none",
        cv_folds: int = 3,
    ):
        self.base_model = base_model
        self.method = method
        self.cv_folds = cv_folds
        self._calibrated: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> CalibrationResult:
        if self.method == "none":
            self.base_model.fit(X, y)
            return CalibrationResult(
                method="none",
                n_classes=len(np.unique(y)),
                applied=False,
            )

        if not hasattr(self.base_model, "predict_proba"):
            self.base_model.fit(X, y)
            return CalibrationResult(
                method="none",
                n_classes=len(np.unique(y)),
                applied=False,
            )

        sk_method = "sigmoid" if self.method == "platt" else "isotonic"
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        self._calibrated = CalibratedClassifierCV(
            estimator=self.base_model,
            method=sk_method,
            cv=cv,
        )
        self._calibrated.fit(X, y)

        return CalibrationResult(
            method=self.method,
            n_classes=len(np.unique(y)),
            applied=True,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._calibrated is not None:
            return self._calibrated.predict(X)  # type: ignore[no-any-return]
        return self.base_model.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self._calibrated is not None:
            return self._calibrated.predict_proba(X)  # type: ignore[no-any-return]
        if hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict_proba(X)  # type: ignore[no-any-return]
        return None

    def save(self, path: str) -> None:
        from tabular_blueprint.utils.safe_pickle import safe_dump

        safe_dump(
            {
                "calibrated": self._calibrated,
                "method": self.method,
                "base_model": self.base_model,
            },
            path,
        )

    def load(self, path: str) -> None:
        from tabular_blueprint.utils.safe_pickle import safe_load_file

        data = safe_load_file(path)
        self._calibrated = data.get("calibrated")
        self.method = data.get("method", self.method)
        self.base_model = data.get("base_model", self.base_model)

    @property
    def model_name(self) -> str:
        base = getattr(self.base_model, "model_name", "unknown")
        if self.method != "none" and self._calibrated is not None:
            return f"{base}_calibrated_{self.method}"
        return base
