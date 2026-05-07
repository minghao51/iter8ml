"""Tests for per-model configuration and HPO search spaces."""

from tabular_blueprint.models.model_configs import (
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
        assert config.iterations == 1000
        assert config.depth == 6
        assert config.learning_rate == 0.05
        assert config.l2_leaf_reg == 3.0
        assert config.early_stopping_rounds == 50
        assert config.task_type in ("CPU", "GPU")
        assert config.random_seed == 42

    def test_resolve_task_type_auto_falls_back_to_cpu(self):
        """On systems without catboost get_gpu_count, "auto" resolves to CPU."""
        config = CatBoostConfig(task_type="auto")
        assert config.task_type == "CPU"

    def test_explicit_task_type_unchanged(self):
        config = CatBoostConfig(task_type="GPU")
        config.resolve_task_type()
        assert config.task_type == "GPU"

    def test_hpo_search_space_structure(self):
        config = CatBoostConfig()
        space = config.hpo_search_space()
        expected_keys = {"depth", "learning_rate", "l2_leaf_reg", "iterations"}
        assert set(space.keys()) == expected_keys
        for _key, val in space.items():
            assert isinstance(val, tuple)
            assert len(val) >= 2

    def test_explicit_values(self):
        config = CatBoostConfig(iterations=500, depth=8, learning_rate=0.1)
        assert config.iterations == 500
        assert config.depth == 8
        assert config.learning_rate == 0.1


class TestLightGBMConfig:
    def test_defaults(self):
        config = LightGBMConfig()
        assert config.n_estimators == 1000
        assert config.max_depth == -1
        assert config.learning_rate == 0.05
        assert config.num_leaves == 31
        assert config.min_child_samples == 20
        assert config.subsample == 0.8
        assert config.colsample_bytree == 0.8
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

    def test_explicit_values(self):
        config = LightGBMConfig(num_leaves=63, subsample=1.0)
        assert config.num_leaves == 63
        assert config.subsample == 1.0


class TestXGBoostConfig:
    def test_defaults(self):
        config = XGBoostConfig()
        assert config.n_estimators == 1000
        assert config.max_depth == 6
        assert config.learning_rate == 0.05
        assert config.subsample == 0.8
        assert config.colsample_bytree == 0.8
        assert config.gamma == 0.0
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
        configs = ModelConfigs(catboost=CatBoostConfig(iterations=500))
        assert configs.catboost.iterations == 500
        assert configs.lightgbm.n_estimators == 1000

    def test_all_hpo_spaces_return_dicts(self):
        configs = ModelConfigs()
        for name in ("catboost", "lightgbm", "xgboost", "tabpfn", "ft_transformer", "tabnet"):
            space = getattr(configs, name).hpo_search_space()
            assert isinstance(space, dict)
