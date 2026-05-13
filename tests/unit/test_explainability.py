"""Tests for SHAP explainability module."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from iter8ml.analysis.explainability import Explainer, FeatureImportance, SHAPExplanationResult


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    model.model_name = "RandomForest"
    return model


def test_explainer_init_with_feature_names():
    names = ["a", "b", "c"]
    explainer = Explainer(model=object(), feature_names=names, output_dir="/tmp")
    assert explainer.feature_names == names


def test_explainer_init_without_feature_names():
    explainer = Explainer(model=object(), output_dir="/tmp")
    assert explainer.feature_names is None


def test_explainer_init_default_output_dir():
    explainer = Explainer(model=object())
    assert str(explainer.output_dir) == "workspace/artifacts"


def test_explain_returns_result(trained_model, sample_data):
    X, _ = sample_data
    explainer = Explainer(model=trained_model, output_dir="/tmp")
    result = explainer.explain(X, run_id="test_001", max_display=5, generate_plots=False)
    assert isinstance(result, SHAPExplanationResult)
    assert result.model_name == "RandomForest"
    assert result.n_features == 5
    assert len(result.top_features) == 5
    assert result.plot_paths == []


def test_explain_ranks_by_importance(trained_model, sample_data):
    X, _ = sample_data
    explainer = Explainer(model=trained_model, output_dir="/tmp")
    result = explainer.explain(X, run_id="test_002", max_display=10, generate_plots=False)
    scores = [f.importance for f in result.top_features]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_explain_generates_plots(trained_model, sample_data, tmp_path):
    X, _ = sample_data
    explainer = Explainer(model=trained_model, output_dir=str(tmp_path))
    result = explainer.explain(X, run_id="plot_test", max_display=5, generate_plots=True)
    plot_dir = tmp_path / "shap_plot_test"
    assert plot_dir.exists()
    if result.plot_paths:
        for path in result.plot_paths:
            assert Path(path).exists()


def test_explain_auto_feature_names(trained_model, sample_data):
    X, _ = sample_data
    explainer = Explainer(model=trained_model, output_dir="/tmp")
    result = explainer.explain(X, run_id="test_003", max_display=5, generate_plots=False)
    assert all(f.feature_name.startswith("feature_") for f in result.top_features)


def test_create_explainer_tree_for_gbdt_models():
    from sklearn.ensemble import RandomForestClassifier

    inner = RandomForestClassifier(n_estimators=5, random_state=42)
    model = type("DummyLightGBM", (), {"model_name": "LightGBM", "model": inner})()
    X = np.random.RandomState(0).rand(20, 4)
    y = (X[:, 0] > 0.5).astype(int)
    inner.fit(X, y)
    explainer = Explainer(model=model, output_dir="/tmp")
    internal = explainer._create_explainer(X)
    import shap

    assert isinstance(internal, shap.TreeExplainer)


def test_create_explainer_kernel(sample_data):
    X, _ = sample_data
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(X, (X[:, 0] > 0.5).astype(int))
    explainer = Explainer(model=model, output_dir="/tmp")
    internal = explainer._create_explainer(X)
    import shap

    assert isinstance(internal, shap.KernelExplainer)


def test_feature_importance_model():
    fi = FeatureImportance(feature_name="age", importance=0.42)
    assert fi.feature_name == "age"
    assert fi.importance == 0.42


def test_shap_explanation_result_model():
    fi = FeatureImportance(feature_name="age", importance=0.42)
    result = SHAPExplanationResult(
        model_name="RF",
        n_features=5,
        top_features=[fi],
        plot_paths=["/tmp/plot.png"],
    )
    assert result.model_name == "RF"
    assert result.n_features == 5
    assert len(result.top_features) == 1
    assert result.plot_paths == ["/tmp/plot.png"]
