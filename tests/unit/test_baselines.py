"""Tests for smart baseline models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from tabular_blueprint.models.baselines import LinearBaseline, NaiveBaseline


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    return X, y


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    return X, y


class TestNaiveBaseline:
    def test_classification_fit_predict(self, classification_data):
        X, y = classification_data
        model = NaiveBaseline(task="classification")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        assert len(np.unique(y)) >= 2
        assert all(p in np.unique(y) for p in preds)

    def test_regression_fit_predict(self, regression_data):
        X, y = regression_data
        model = NaiveBaseline(task="regression")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        assert np.allclose(preds, np.mean(y))

    def test_predict_proba(self, classification_data):
        X, y = classification_data
        model = NaiveBaseline(task="classification")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba is not None
        assert proba.shape[0] == X.shape[0]
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_regression(self, regression_data):
        X, y = regression_data
        model = NaiveBaseline(task="regression")
        model.fit(X, y)
        assert model.predict_proba(X) is None

    def test_model_name(self):
        model = NaiveBaseline()
        assert model.model_name == "NaiveBaseline"


class TestLinearBaseline:
    def test_classification_fit_predict(self, classification_data):
        X, y = classification_data
        model = LinearBaseline(task="classification")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_regression_fit_predict(self, regression_data):
        X, y = regression_data
        model = LinearBaseline(task="regression")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_proba(self, classification_data):
        X, y = classification_data
        model = LinearBaseline(task="classification")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba is not None
        assert proba.shape[0] == X.shape[0]

    def test_model_name(self):
        model = LinearBaseline()
        assert model.model_name == "LinearBaseline"
