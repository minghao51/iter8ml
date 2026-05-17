"""Metamorphic tests for explainability invariants."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from iter8ml.analysis.explainability import Explainer

pytestmark = pytest.mark.metamorphic


class _DummyModelForExplain:
    def __init__(self):
        self.model_name = "DummyExplainModel"

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, 2))
        proba[:, 1] = 0.6
        proba[:, 0] = 0.4
        return proba


class TestExplainabilityMetamorphic:
    """Metamorphic: importance scores are consistent under transformations."""

    def test_importance_non_increasing_with_max_display(self, tmp_path):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = Explainer(model, output_dir=str(tmp_path))
        result_full = explainer.explain(X, run_id="test_full", max_display=5, generate_plots=False)
        result_truncated = explainer.explain(
            X, run_id="test_trunc", max_display=2, generate_plots=False
        )

        assert len(result_full.top_features) >= len(result_truncated.top_features)
        assert result_full.n_features == result_truncated.n_features

    def test_importance_contains_feature_names(self, tmp_path):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        feature_names = [f"feat_{i}" for i in range(5)]
        explainer = Explainer(model, feature_names=feature_names, output_dir=str(tmp_path))
        result = explainer.explain(X, run_id="test_names", max_display=5, generate_plots=False)

        feature_names_in_result = {f.feature_name for f in result.top_features}
        assert feature_names_in_result.issubset(set(feature_names))

    def test_importance_descending_order(self, tmp_path):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = Explainer(model, output_dir=str(tmp_path))
        result = explainer.explain(X, run_id="test_order", max_display=5, generate_plots=False)

        importances = [f.importance for f in result.top_features]
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1], "Importances not in descending order"


class TestExplainabilityEdgeCases:
    """Edge-case handling for explainer."""

    def test_single_feature(self, tmp_path):
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=100,
            n_features=3,
            n_informative=2,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = X[:, :1]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = Explainer(model, output_dir=str(tmp_path))
        result = explainer.explain(X, run_id="test_single", max_display=5, generate_plots=False)
        assert result.n_features == 1
        assert len(result.top_features) == 1

    def test_kernel_explainer_for_non_tree(self, tmp_path):
        from sklearn.datasets import make_classification

        X, _ = make_classification(
            n_samples=50,
            n_features=3,
            n_informative=2,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42,
        )
        model = _DummyModelForExplain()

        explainer = Explainer(model, output_dir=str(tmp_path))
        result = explainer.explain(X, run_id="test_kernel", max_display=3, generate_plots=False)
        assert result.n_features == 3
        assert len(result.top_features) <= 3

    def test_no_feature_names_generates_defaults(self, tmp_path):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = Explainer(model, output_dir=str(tmp_path))
        result = explainer.explain(
            X, run_id="test_default_names", max_display=5, generate_plots=False
        )
        feature_names = {f.feature_name for f in result.top_features}
        expected = {f"feature_{i}" for i in range(5)}
        assert feature_names == expected
