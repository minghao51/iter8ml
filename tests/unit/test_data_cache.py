"""Tests for PreprocessingCache."""

import numpy as np
import pytest

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.data.cache import PreprocessingCache, _cache_key


@pytest.fixture
def config():
    return ExperimentConfig(
        name="test",
        task="classification",
        target_col="target",
        data_path="data.csv",
    )


@pytest.fixture
def cache(tmp_path):
    return PreprocessingCache(workspace_dir=tmp_path)


class TestCacheKey:
    def test_deterministic(self, config):
        key1 = _cache_key("abc123", config)
        key2 = _cache_key("abc123", config)
        assert key1 == key2

    def test_different_data_hash(self, config):
        key1 = _cache_key("abc123", config)
        key2 = _cache_key("def456", config)
        assert key1 != key2

    def test_different_config(self, config):
        config2 = config.model_copy(update={"cv_folds": 10})
        key1 = _cache_key("abc123", config)
        key2 = _cache_key("abc123", config2)
        assert key1 != key2


class TestPreprocessingCache:
    def test_miss_when_no_files(self, cache, config):
        result = cache.load("abc123", config)
        assert result is None

    def test_miss_when_partial_files(self, cache, config, tmp_path):
        key = _cache_key("abc123", config)
        (tmp_path / ".tabblueprint/cache" / f"{key}_X.npy").write_text("junk")
        result = cache.load("abc123", config)
        assert result is None

    def test_miss_when_corrupted_meta(self, cache, config):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        cache.save("abc123", config, X, y, feature_names=["a", "b"])
        key = _cache_key("abc123", config)
        meta_path = cache.cache_dir / f"{key}_meta.json"
        meta_path.write_text("not valid json")
        result = cache.load("abc123", config)
        assert result is None

    def test_save_and_load_roundtrip(self, cache, config):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        feature_names = ["a", "b"]

        cache.save("abc123", config, X, y, feature_names)
        loaded_X, loaded_y, loaded_names = cache.load("abc123", config)

        assert loaded_X is not None
        assert loaded_y is not None
        assert loaded_names is not None
        np.testing.assert_array_equal(loaded_X, X)
        np.testing.assert_array_equal(loaded_y, y)
        assert loaded_names == feature_names

    def test_save_and_load_different_hash(self, cache, config):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        cache.save("abc123", config, X, y, feature_names=["a", "b"])

        result = cache.load("different_hash", config)
        assert result is None

    def test_clear_removes_all_entries(self, cache, config):
        cache.save("abc1", config, np.array([[1.0]]), np.array([0]), feature_names=["a"])
        cache.save("abc2", config, np.array([[2.0]]), np.array([1]), feature_names=["b"])

        count = cache.clear()
        assert count >= 4

        assert cache.load("abc1", config) is None
        assert cache.load("abc2", config) is None

    def test_clear_returns_zero_on_empty(self, cache):
        count = cache.clear()
        assert count == 0

    def test_works_with_full_config(self, cache):
        full_config = ExperimentConfig(
            name="full_test",
            task="regression",
            target_col="price",
            data_path="data.parquet",
            cv_folds=10,
            afe_enabled=True,
            embedding_enabled=True,
        )
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        cache.save("full_hash", full_config, X, y, feature_names=["a", "b", "c", "d", "e"])

        loaded_X, loaded_y, _loaded_names = cache.load("full_hash", full_config)
        assert loaded_X is not None
        np.testing.assert_array_equal(loaded_X, X)
        np.testing.assert_array_equal(loaded_y, y)
