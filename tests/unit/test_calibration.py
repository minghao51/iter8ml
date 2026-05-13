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

    def test_save_and_load_roundtrip(self, classification_data, tmp_path):
        X, y = classification_data
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(random_state=42)
        calibrated = CalibratedModel(base, method="platt")
        calibrated.fit(X, y)

        save_path = str(tmp_path / "calibrated_model.pkl")
        calibrated.save(save_path)

        loaded = CalibratedModel(LogisticRegression(), method="none")
        loaded.load(save_path)

        assert loaded.method == "platt"
        preds = loaded.predict(X)
        assert len(preds) == len(y)
        proba = loaded.predict_proba(X)
        assert proba is not None

    def test_no_predict_proba_falls_back(self, classification_data):
        X, y = classification_data

        class NoProbaEstimator:
            def fit(self, X, y):
                return None

            def predict(self, X):
                return np.zeros(len(X))

        base = NoProbaEstimator()
        calibrated = CalibratedModel(base, method="platt")
        result = calibrated.fit(X, y)
        assert result.applied is False
        assert result.method == "none"
        assert calibrated.predict_proba(X) is None

    def test_predict_proba_returns_none_when_no_proba_on_base(self, classification_data):
        X, _y = classification_data

        class NoProbaEstimator:
            model_name = "NoProba"

            def predict(self, X):
                return np.zeros(len(X))

        base = NoProbaEstimator()
        calibrated = CalibratedModel(base, method="platt")
        assert calibrated.predict_proba(X) is None

    def test_model_name_without_fit(self):
        base = type("MockModel", (), {"model_name": "Mock"})()
        calibrated = CalibratedModel(base, method="none")
        assert calibrated.model_name == "Mock"
