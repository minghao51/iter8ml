"""Tests for automated feature engineering module."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from iter8ml.data.features import (
    _effective_parallel_jobs,
    _safe_ratio,
    detect_target_skewness,
    discover_interactions,
    extract_top_k_features,
    prune_features,
    transform_target,
)
from iter8ml.engine.pipelines.nodes.features import _run_embedding
from iter8ml.workspace import Workspace


class TestDetectTargetSkewness:
    def test_symmetric_data_near_zero(self):
        y = np.random.normal(0, 1, 10_000)
        skew = detect_target_skewness(y)
        assert abs(skew) < 0.1

    def test_right_skewed_data(self):
        y = np.random.exponential(2, 10_000)
        skew = detect_target_skewness(y)
        assert skew > 1.0


class TestTransformTarget:
    def test_none_method_returns_original(self):
        y = np.random.exponential(2, 1000)
        y_out, result, _ = transform_target(y, method="none")
        np.testing.assert_array_equal(y_out, y)
        assert result.applied is False

    def test_auto_no_transform_when_not_skewed(self):
        y = np.random.normal(0, 1, 10_000)
        _, result, _ = transform_target(y, method="auto", skewness_threshold=1.0)
        assert result.applied is False
        assert result.method == "none"

    def test_log1p_transform(self):
        y = np.random.exponential(2, 1000)
        _, result, _ = transform_target(y, method="log1p")
        assert result.applied is True
        assert result.method == "log1p"
        assert abs(result.transformed_skewness) < abs(result.original_skewness)

    def test_yeo_johnson_transform(self):
        y = np.random.exponential(2, 1000) - 1
        y_out, result, transformer = transform_target(y, method="yeo-johnson")
        assert result.applied is True
        assert result.method == "yeo-johnson"
        assert transformer is not None
        y_inv = transformer.inverse_transform(y_out)
        np.testing.assert_allclose(y_inv, y, rtol=1e-5)

    def test_box_cox_transform_positive_only(self):
        y = np.random.exponential(2, 1000)
        y_out, result, transformer = transform_target(y, method="box-cox")
        assert result.applied is True
        assert result.method == "box-cox"
        y_inv = transformer.inverse_transform(y_out)
        np.testing.assert_allclose(y_inv, y, rtol=1e-5)

    def test_auto_selects_yeo_johnson_for_negative_values(self):
        y = np.random.normal(0, 2, 10_000)
        _, result, _ = transform_target(y, method="auto", skewness_threshold=0.5)
        if result.applied:
            assert result.method == "yeo-johnson"

    def test_inverse_transform_roundtrip_log1p(self):
        y = np.abs(np.random.normal(5, 2, 100))
        y_out, _, transformer = transform_target(y, method="log1p")
        y_inv = transformer.inverse_transform(y_out)
        np.testing.assert_allclose(y_inv, y, rtol=1e-10)


class TestExtractTopKFeatures:
    def test_returns_correct_number_of_indices(self):
        X = np.random.randn(200, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        indices, _ = extract_top_k_features(None, X, y, k=5, task="classification")
        assert len(indices) == 5
        assert all(isinstance(i, int) for i in indices)

    def test_returns_at_most_n_features(self):
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        indices, _ = extract_top_k_features(None, X, y, k=10, task="classification")
        assert len(indices) == 3


class TestDiscoverInteractions:
    def test_discovers_interaction_in_synthetic_data(self):
        np.random.seed(42)
        X = np.random.randn(300, 5)
        y = (X[:, 0] * X[:, 1] > 0).astype(float)
        top_k = [0, 1]
        X_new, result = discover_interactions(
            X, y, top_k_indices=top_k, task="classification", lift_threshold=0.0
        )
        assert result.n_candidates_tested > 0
        assert isinstance(X_new, np.ndarray)
        assert result.effective_n_jobs >= 1
        assert result.duration_seconds >= 0.0

    def test_no_interactions_kept_with_high_threshold(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, 200)
        X_new, result = discover_interactions(
            X, y, top_k_indices=[0, 1], task="classification", lift_threshold=10.0
        )
        assert result.n_candidates_kept == 0
        assert X_new.shape[1] == 0

    def test_default_parallel_jobs_is_one(self):
        jobs = _effective_parallel_jobs(
            requested_jobs=1,
            n_tasks=10,
            n_samples=120,
            n_features=5,
        )
        assert jobs == 1

    def test_candidate_pairs_are_capped(self):
        np.random.seed(42)
        X = np.random.randn(200, 12)
        y = np.random.randint(0, 2, 200)
        top_k = list(range(10))
        _, result = discover_interactions(
            X,
            y,
            top_k_indices=top_k,
            task="classification",
            max_candidate_pairs=5,
        )
        assert result.n_candidates_tested <= 5

    def test_effective_parallel_jobs_caps_for_large_matrices(self):
        jobs = _effective_parallel_jobs(
            requested_jobs=8,
            n_tasks=50,
            n_samples=2_000,
            n_features=1_000,
        )
        assert jobs == 2


class TestSafeRatio:
    def test_normal_division(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = _safe_ratio(a, b)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])

    def test_zero_denominator_returns_zero(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 1e-12])
        result = _safe_ratio(a, b)
        assert result[0] == 0.0

    def test_overflow_input_returns_finite_array(self):
        a = np.array([8.98846567431158e307])
        b = np.array([0.5])
        result = _safe_ratio(a, b)
        assert result is not None
        assert np.isfinite(result).all()


class TestPruneFeatures:
    def test_prunes_zero_importance_features(self):
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        X = np.random.randn(200, 5)
        X[:, 2:] = np.random.randn(200, 3) * 0.001
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        X_pruned, result = prune_features(
            model,
            X,
            y,
            feature_names=["f0", "f1", "noise0", "noise1", "noise2"],
            min_importance=0.005,
            task="classification",
        )
        assert result.n_original == 5
        assert result.n_kept < 5
        assert result.n_dropped > 0
        assert X_pruned.shape[1] == result.n_kept

    def test_no_pruning_when_all_important(self):
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        X_pruned, result = prune_features(
            model,
            X,
            y,
            min_importance=0.0,
            task="classification",
        )
        _ = X_pruned
        assert result.n_kept == 3
        assert result.n_dropped == 0

    def test_prune_with_no_predict_model(self):
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        X_pruned, result = prune_features(
            "not_a_model",
            X,
            y,
            feature_names=["a", "b", "c"],
            min_importance=0.001,
        )
        assert result.n_kept == 3
        assert result.n_dropped == 0
        assert X_pruned.shape == X.shape


def test_run_embedding_uses_real_dataframe_and_target_col(monkeypatch, tmp_path):
    captured = {}

    def fake_fit_transform(self, *, df, X, y, feature_names, target_col, run_id, data_hash=""):
        captured["df"] = df
        captured["target_col"] = target_col
        captured["run_id"] = run_id
        captured["feature_names"] = feature_names
        return X, feature_names

    monkeypatch.setattr(
        "iter8ml.data.embedding.EmbeddingEngine.fit_transform",
        fake_fit_transform,
    )

    data_prep_result = SimpleNamespace(
        dataframe=pl.DataFrame({"cat": ["a", "b"], "target": [0, 1]}),
        X=np.array([[1.0], [2.0]]),
        y=np.array([0, 1]),
        feature_names=["cat"],
    )

    X_out, names_out = _run_embedding(
        data_prep_result=data_prep_result,
        target_col="target",
        task="classification",
        random_seed=42,
        run_id="run_1",
        workspace=Workspace(root=tmp_path),
    )

    assert captured["df"] is data_prep_result.dataframe
    assert captured["target_col"] == "target"
    assert captured["run_id"] == "run_1"
    np.testing.assert_array_equal(X_out, data_prep_result.X)
    assert names_out == data_prep_result.feature_names
