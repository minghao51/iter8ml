"""Unit tests for embedding_engine helper functions."""

import numpy as np
import polars as pl
import pytest

from tabular_blueprint.data.embedding_engine import (
    augment_with_embeddings,
    detect_high_cardinality_columns,
    extract_cat_codes,
)


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "user_id": [f"user_{i % 200}" for i in range(100)],
            "product_id": [f"prod_{i % 300}" for i in range(100)],
            "region": [f"region_{i % 5}" for i in range(100)],
            "age": np.random.randint(18, 80, 100),
            "target": np.random.randint(0, 2, 100),
        }
    )


class TestDetectHighCardinalityColumns:
    def test_detects_high_cardinality(self, sample_df):
        result = detect_high_cardinality_columns(sample_df, max_categories=50)
        assert "user_id" in result
        assert "product_id" in result

    def test_excludes_low_cardinality(self, sample_df):
        result = detect_high_cardinality_columns(sample_df, max_categories=50)
        assert "region" not in result

    def test_excludes_target(self, sample_df):
        result = detect_high_cardinality_columns(sample_df, max_categories=50, target_col="target")
        assert "target" not in result

    def test_empty_when_none_high(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = detect_high_cardinality_columns(df, max_categories=50)
        assert result == []

    def test_detects_integer_high_cardinality(self):
        ids = list(range(200))
        df = pl.DataFrame({"id_col": ids, "val": [0] * 200})
        result = detect_high_cardinality_columns(df, max_categories=50)
        assert "id_col" in result


class TestExtractCatCodes:
    def test_returns_contiguous_codes(self):
        df = pl.DataFrame({"cat": ["b", "a", "c", "a", "b"]})
        codes, vocab_sizes, mappings = extract_cat_codes(df, ["cat"])
        assert set(codes["cat"]) == {0, 1, 2}
        assert vocab_sizes["cat"] == 3
        assert len(mappings["cat"]) == 3

    def test_multiple_columns(self, sample_df):
        codes, _vocab_sizes, _mappings = extract_cat_codes(sample_df, ["user_id", "product_id"])
        assert "user_id" in codes
        assert "product_id" in codes
        assert codes["user_id"].shape == (100,)
        assert codes["product_id"].shape == (100,)

    def test_empty_columns(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        codes, _vocab_sizes, _mappings = extract_cat_codes(df, [])
        assert codes == {}
        assert _vocab_sizes == {}


class TestAugmentWithEmbeddings:
    def test_removes_cat_columns_and_adds_embeddings(self):
        X = np.random.randn(10, 4)
        embeddings = np.random.randn(10, 8)
        feature_names = ["num_1", "cat_a", "num_2", "cat_b"]
        cat_columns = ["cat_a", "cat_b"]

        X_aug, names = augment_with_embeddings(
            X,
            embeddings,
            feature_names,
            cat_columns,
            per_col_dim=4,
        )

        assert X_aug.shape == (10, 10)
        assert "num_1" in names
        assert "num_2" in names
        assert "cat_a" not in names
        assert "cat_b" not in names
        assert any("_emb_" in n for n in names)

    def test_no_remaining_features(self):
        X = np.random.randn(5, 2)
        embeddings = np.random.randn(5, 6)
        feature_names = ["cat_a", "cat_b"]
        cat_columns = ["cat_a", "cat_b"]

        X_aug, names = augment_with_embeddings(
            X,
            embeddings,
            feature_names,
            cat_columns,
            per_col_dim=3,
        )
        assert X_aug.shape == (5, 6)
        assert len(names) == 6

    def test_no_cat_columns_to_remove(self):
        X = np.random.randn(5, 3)
        embeddings = np.random.randn(5, 4)
        feature_names = ["a", "b", "c"]
        cat_columns = ["cat_x"]

        X_aug, names = augment_with_embeddings(X, embeddings, feature_names, cat_columns)
        assert X_aug.shape == (5, 7)
        assert "a" in names
        assert "b" in names
        assert "c" in names
