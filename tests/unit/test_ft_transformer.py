"""Tests for FT-Transformer model."""

import os
import warnings

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.models.deep.ft_transformer import FTTransformerModel


@pytest.fixture
def sample_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


def _make_model(**kwargs):
    """Create model with explicit device handling."""
    defaults = dict(task="classification", n_features=10, n_classes=2, n_epochs=2, batch_size=32)
    defaults.update(kwargs)
    return FTTransformerModel(**defaults)


def test_ft_transformer_fit(sample_data):
    X, y = sample_data
    model = _make_model()
    model.fit(X, y)
    assert model.model is not None


def test_ft_transformer_predict(sample_data):
    X, y = sample_data
    model = _make_model()
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


def test_ft_transformer_predict_proba(sample_data):
    X, y = sample_data
    model = _make_model()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba is not None
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_ft_transformer_save_load(sample_data):
    X, y = sample_data
    model = _make_model()
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "ft_transformer_model.pt")
        torch.save(
            {
                "model_state": model.model.state_dict(),
                "n_features": model.n_features,
                "n_classes": model.n_classes,
            },
            path,
        )

        new_model = FTTransformerModel(task="classification", n_features=10, n_classes=2)
        new_model.load(path)
        assert new_model.model is not None


def test_ft_transformer_model_name():
    model = FTTransformerModel()
    assert model.model_name == "FT-Transformer"


def test_ft_transformer_regression():
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    model = FTTransformerModel(
        task="regression",
        n_features=10,
        n_classes=1,
        n_epochs=2,
        batch_size=32,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)
    assert model.predict_proba(X) is None


def test_ft_transformer_regression_does_not_broadcast_targets():
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=64, n_features=10, random_state=42)
    model = FTTransformerModel(
        task="regression",
        n_features=10,
        n_classes=1,
        n_epochs=1,
        batch_size=16,
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(X, y)

    broadcast_warnings = [
        warning for warning in record if "different to the input size" in str(warning.message)
    ]
    assert not broadcast_warnings
