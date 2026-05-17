"""Unit tests for EmbeddingEngine orchestrator (moved from engine/embedding_trainer)."""

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from iter8ml.config import EmbeddingConfig  # noqa: E402
from iter8ml.constants import EmbeddingMethod  # noqa: E402
from iter8ml.data.embedding import EmbeddingEngine  # noqa: E402
from iter8ml.workspace import Workspace  # noqa: E402


@pytest.fixture
def high_card_df():
    n = 100
    rng = np.random.RandomState(42)
    return pl.DataFrame(
        {
            "user_id": [f"u_{i % 80}" for i in range(n)],
            "product_id": [f"p_{i % 120}" for i in range(n)],
            "numeric_feat": rng.randn(n),
            "target": rng.randint(0, 2, n),
        }
    )


def _make_engine(tmp_path, **config_overrides):
    config = EmbeddingConfig(
        method=EmbeddingMethod.ENTITY,
        dim=4,
        epochs=2,
        max_categories=10,
    )
    for key, value in config_overrides.items():
        if key == "task":
            continue
        if key == "embedding_method":
            config = config.model_copy(update={"method": EmbeddingMethod(value)})
        elif key == "embedding_dim":
            config = config.model_copy(update={"dim": value})
        elif key == "embedding_epochs":
            config = config.model_copy(update={"epochs": value})
        elif key == "embedding_max_categories":
            config = config.model_copy(update={"max_categories": value})
        elif key == "embedding_ae_latent_dim":
            config = config.model_copy(update={"ae_latent_dim": value})
    return EmbeddingEngine(
        task=config_overrides.get("task", "classification"),
        workspace=Workspace(root=tmp_path / "workspace"),
        config=config,
    )


class TestEmbeddingEngineEntity:
    def test_augments_features(self, high_card_df, tmp_path):
        engine = _make_engine(tmp_path, embedding_method="entity", embedding_dim=4)
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
        )

        n_emb_cols = 2
        expected_new_cols = X.shape[1] - n_emb_cols + n_emb_cols * 4
        assert X_aug.shape[0] == 100
        assert X_aug.shape[1] == expected_new_cols
        assert any("_emb_" in n for n in aug_names)

    def test_saves_artifacts(self, high_card_df, tmp_path):
        engine = _make_engine(tmp_path, embedding_method="entity", embedding_dim=4)
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
        )

        emb_dir = tmp_path / "workspace" / "embeddings"
        assert (emb_dir / "save_test.pt").exists()
        assert (emb_dir / "save_test_mappings.json").exists()

    def test_noop_when_no_high_card(self, tmp_path):
        engine = _make_engine(tmp_path, embedding_max_categories=500)
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
        )
        assert X_aug.shape == X.shape
        assert aug_names == feature_names


class TestEmbeddingEngineAutoencoder:
    def test_augments_features(self, high_card_df, tmp_path):
        engine = _make_engine(
            tmp_path,
            embedding_method="autoencoder",
            embedding_dim=4,
            embedding_ae_latent_dim=8,
        )
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
        )

        n_cat = 2
        expected_dim = (X.shape[1] - n_cat) + 8
        assert X_aug.shape == (100, expected_dim)


class TestEmbeddingEngineRegression:
    def test_entity_regression(self, high_card_df, tmp_path):
        engine = _make_engine(
            tmp_path, task="regression", embedding_method="entity", embedding_dim=4
        )
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
        )
        assert X_aug.shape[0] == 100
        assert X_aug.shape[1] > 0
