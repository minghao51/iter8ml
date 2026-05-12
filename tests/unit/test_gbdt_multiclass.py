"""Regression tests for multiclass handling in GBDT wrappers."""

import numpy as np
from sklearn.datasets import make_classification

from iter8ml.engine.models.lightgbm_model import LightGBMModel
from iter8ml.engine.models.xgboost_model import XGBoostModel


def _multiclass_dataset_with_noncontiguous_labels() -> tuple[np.ndarray, np.ndarray]:
    X, y0 = make_classification(
        n_samples=160,
        n_features=12,
        n_classes=3,
        n_informative=8,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    # Non-contiguous and non-zero-based labels: {10, 20, 30}
    mapping = np.array([10, 20, 30])
    y = mapping[y0]
    return X, y


def _assert_multiclass_behavior(model_class: type[LightGBMModel] | type[XGBoostModel]) -> None:
    X, y = _multiclass_dataset_with_noncontiguous_labels()
    model = model_class(task="classification")
    model.fit(X, y)

    preds = model.predict(X[:20])
    proba = model.predict_proba(X[:20])

    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({10, 20, 30})
    assert proba is not None
    assert proba.shape == (20, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lightgbm_multiclass_noncontiguous_labels() -> None:
    _assert_multiclass_behavior(LightGBMModel)


def test_xgboost_multiclass_noncontiguous_labels() -> None:
    _assert_multiclass_behavior(XGBoostModel)
