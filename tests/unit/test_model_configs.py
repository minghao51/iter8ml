"""Tests for per-model configuration and HPO search spaces."""

from iter8ml.engine.models.model_configs import (
    CatBoostConfig,
    FTTransformerConfig,
    LightGBMConfig,
    ModelConfigs,
    TabNetConfig,
    TabPFNConfig,
    XGBoostConfig,
)


class TestCatBoostConfig:
    def test_defaults(self):
        config = CatBoostConfig()
        assert config.random_seed == 42

    def test_hpo_search_space_structure(self):
        config = CatBoostConfig()
        space = config.hpo_search_space()
        expected_keys = {"depth", "learning_rate", "l2_leaf_reg", "iterations"}
        assert set(space.keys()) == expected_keys
        for _key, val in space.items():
            assert isinstance(val, tuple)
            assert len(val) >= 2


class TestLightGBMConfig:
    def test_defaults(self):
        config = LightGBMConfig()
        assert config.random_seed == 42

    def test_hpo_search_space_keys(self):
        config = LightGBMConfig()
        space = config.hpo_search_space()
        expected_keys = {
            "max_depth",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
        }
        assert set(space.keys()) == expected_keys


class TestXGBoostConfig:
    def test_defaults(self):
        config = XGBoostConfig()
        assert config.random_seed == 42

    def test_hpo_search_space_keys(self):
        config = XGBoostConfig()
        space = config.hpo_search_space()
        expected_keys = {
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "gamma",
        }
        assert set(space.keys()) == expected_keys


class TestTabPFNConfig:
    def test_defaults(self):
        config = TabPFNConfig()
        assert config.n_estimators == 4
        assert config.device == "cpu"
        assert config.random_seed == 42
        assert config.max_rows == 50_000

    def test_hpo_search_space_empty(self):
        config = TabPFNConfig()
        assert config.hpo_search_space() == {}


class TestTabNetConfig:
    def test_defaults(self):
        config = TabNetConfig()
        assert config.n_epochs == 50
        assert config.batch_size == 256
        assert config.learning_rate == 1e-3
        assert config.random_seed == 42

    def test_hpo_search_space_keys(self):
        config = TabNetConfig()
        space = config.hpo_search_space()
        expected_keys = {"learning_rate", "batch_size", "n_epochs"}
        assert set(space.keys()) == expected_keys


class TestFTTransformerConfig:
    def test_defaults(self):
        config = FTTransformerConfig()
        assert config.n_epochs == 100
        assert config.batch_size == 128
        assert config.learning_rate == 1e-4
        assert config.n_heads == 4
        assert config.d_hidden == 128
        assert config.n_layers == 3
        assert config.dropout == 0.1
        assert config.random_seed == 42

    def test_hpo_search_space_keys(self):
        config = FTTransformerConfig()
        space = config.hpo_search_space()
        expected_keys = {"learning_rate", "d_hidden", "n_heads", "n_layers", "dropout"}
        assert set(space.keys()) == expected_keys


class TestModelConfigs:
    def test_default_construction(self):
        configs = ModelConfigs()
        assert isinstance(configs.catboost, CatBoostConfig)
        assert isinstance(configs.lightgbm, LightGBMConfig)
        assert isinstance(configs.xgboost, XGBoostConfig)
        assert isinstance(configs.tabpfn, TabPFNConfig)
        assert isinstance(configs.ft_transformer, FTTransformerConfig)
        assert isinstance(configs.tabnet, TabNetConfig)

    def test_override_single_config(self):
        configs = ModelConfigs(catboost=CatBoostConfig(random_seed=7))
        assert configs.catboost.random_seed == 7
        assert configs.lightgbm.random_seed == 42

    def test_all_hpo_spaces_return_dicts(self):
        configs = ModelConfigs()
        for name in ("catboost", "lightgbm", "xgboost", "tabpfn", "ft_transformer", "tabnet"):
            space = getattr(configs, name).hpo_search_space()
            assert isinstance(space, dict)
