"""Unit tests for EmbeddingEngine orchestrator."""

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from tabular_blueprint.config import ExperimentConfig  # noqa: E402
from tabular_blueprint.constants import EmbeddingMethod, TaskType  # noqa: E402
from tabular_blueprint.engine.embedding_trainer import EmbeddingEngine  # noqa: E402
from tabular_blueprint.engine.tracker import JSONLTracker  # noqa: E402


@pytest.fixture
def make_config(tmp_path):
    def _make(**overrides):
        defaults = dict(  # noqa: C408
            name="test",
            task=TaskType.CLASSIFICATION,
            target_col="target",
            data_path="",
            workspace_dir=tmp_path / "workspace",
            embedding_enabled=True,
            embedding_dim=4,
            embedding_epochs=2,
            embedding_max_categories=10,
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    return _make


@pytest.fixture
def tracker(tmp_path):
    return JSONLTracker(log_path=str(tmp_path / "test.jsonl"))


@pytest.fixture
def high_card_df():
    n = 100
    return pl.DataFrame(
        {
            "user_id": [f"u_{i % 80}" for i in range(n)],
            "product_id": [f"p_{i % 120}" for i in range(n)],
            "numeric_feat": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        }
    )


class TestEmbeddingEngineEntity:
    def test_augments_features(self, make_config, tracker, high_card_df):
        config = make_config(embedding_method=EmbeddingMethod.ENTITY, embedding_dim=4)
        engine = EmbeddingEngine(config, tracker)

        X = high_card_df.drop("target").to_numpy()
        y = high_card_df["target"].to_numpy()
        feature_names = [c for c in high_card_df.columns if c != "target"]

        X_aug, aug_names = engine.fit_transform(
            df=high_card_df,
            X=X,
            y=y,
            feature_names=feature_names,
            target_col="target",
            run_id="test_run",
            data_hash="abc",
        )

        n_emb_cols = 2
        expected_new_cols = X.shape[1] - n_emb_cols + n_emb_cols * config.embedding_dim
        assert X_aug.shape[0] == 100
        assert X_aug.shape[1] == expected_new_cols
        assert any("_emb_" in n for n in aug_names)

    def test_saves_artifacts(self, make_config, tracker, high_card_df, tmp_path):
        config = make_config(embedding_method=EmbeddingMethod.ENTITY, embedding_dim=4)
        engine = EmbeddingEngine(config, tracker)

        X = high_card_df.drop("target").to_numpy()
        y = high_card_df["target"].to_numpy()
        feature_names = [c for c in high_card_df.columns if c != "target"]

        engine.fit_transform(
            df=high_card_df,
            X=X,
            y=y,
            feature_names=feature_names,
            target_col="target",
            run_id="save_test",
            data_hash="abc",
        )

        emb_dir = config.workspace_dir / "embeddings"
        assert (emb_dir / "save_test.pt").exists()
        assert (emb_dir / "save_test_mappings.json").exists()

    def test_noop_when_no_high_card(self, make_config, tracker):
        config = make_config(embedding_method=EmbeddingMethod.ENTITY, embedding_max_categories=500)
        engine = EmbeddingEngine(config, tracker)

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
        X = df.drop("target").to_numpy()
        y = df["target"].to_numpy()
        feature_names = ["a", "b"]

        X_aug, aug_names = engine.fit_transform(
            df=df,
            X=X,
            y=y,
            feature_names=feature_names,
            target_col="target",
            run_id="noop_test",
            data_hash="abc",
        )
        assert X_aug.shape == X.shape
        assert aug_names == feature_names


class TestEmbeddingEngineAutoencoder:
    def test_augments_features(self, make_config, tracker, high_card_df):
        config = make_config(
            embedding_method=EmbeddingMethod.AUTOENCODER,
            embedding_dim=4,
            embedding_ae_latent_dim=8,
        )
        engine = EmbeddingEngine(config, tracker)

        X = high_card_df.drop("target").to_numpy()
        y = high_card_df["target"].to_numpy()
        feature_names = [c for c in high_card_df.columns if c != "target"]

        X_aug, _aug_names = engine.fit_transform(
            df=high_card_df,
            X=X,
            y=y,
            feature_names=feature_names,
            target_col="target",
            run_id="ae_test",
            data_hash="abc",
        )

        n_cat = 2
        expected_dim = (X.shape[1] - n_cat) + config.embedding_ae_latent_dim
        assert X_aug.shape == (100, expected_dim)


class TestEmbeddingEngineRegression:
    def test_entity_regression(self, make_config, tracker, high_card_df):
        config = make_config(
            task=TaskType.REGRESSION,
            embedding_method=EmbeddingMethod.ENTITY,
            embedding_dim=4,
        )
        engine = EmbeddingEngine(config, tracker)

        X = high_card_df.drop("target").to_numpy()
        y = np.random.randn(100).astype(np.float64)
        feature_names = [c for c in high_card_df.columns if c != "target"]

        X_aug, _aug_names = engine.fit_transform(
            df=high_card_df,
            X=X,
            y=y,
            feature_names=feature_names,
            target_col="target",
            run_id="reg_test",
            data_hash="abc",
        )
        assert X_aug.shape[0] == 100
        assert X_aug.shape[1] > 0
