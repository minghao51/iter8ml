"""Tests for probability calibration module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from iter8ml.engine.calibration import CalibratedModel


class TestCalibratedModel:
    @pytest.fixture
    def classification_data(self):
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        return X, y

    def test_none_calibration_passes_through(self, classification_data):
        X, y = classification_data
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(random_state=42)
        calibrated = CalibratedModel(base, method="none")
        result = calibrated.fit(X, y)

        assert result.method == "none"
        assert result.applied is False
        preds = calibrated.predict(X)
        assert len(preds) == len(y)

    def test_platt_calibration_applied(self, classification_data):
        X, y = classification_data
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(random_state=42)
        calibrated = CalibratedModel(base, method="platt")
        result = calibrated.fit(X, y)

        assert result.method == "platt"
        assert result.applied is True
        proba = calibrated.predict_proba(X)
        assert proba is not None
        assert proba.shape[0] == len(y)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_isotonic_calibration_applied(self, classification_data):
        X, y = classification_data
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(random_state=42)
        calibrated = CalibratedModel(base, method="isotonic")
        result = calibrated.fit(X, y)

        assert result.method == "isotonic"
        assert result.applied is True
        preds = calibrated.predict(X)
        assert len(preds) == len(y)

    def test_model_name_reflects_calibration(self, classification_data):
        X, y = classification_data
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(random_state=42)
        calibrated = CalibratedModel(base, method="platt")
        calibrated.fit(X, y)

        name = calibrated.model_name
        assert "platt" in name

    def test_none_calibration_keeps_base_name(self, classification_data):
        X, y = classification_data
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(random_state=42)
        calibrated = CalibratedModel(base, method="none")
        calibrated.fit(X, y)

        name = calibrated.model_name
        assert "calibrated" not in name
