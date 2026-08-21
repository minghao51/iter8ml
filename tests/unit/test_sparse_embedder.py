"""Unit tests for sparse_embedder PyTorch modules."""

import pytest

torch = pytest.importorskip("torch")

from iter8ml.engine.models.sparse_embedder import EntityEmbedding, TabularDAE  # noqa: E402


@pytest.fixture
def vocab_sizes():
    return {"cat_a": 100, "cat_b": 200}


@pytest.fixture
def sample_codes(vocab_sizes):
    n = 32
    return {col: torch.randint(0, vocab, (n,)) for col, vocab in vocab_sizes.items()}


class TestEntityEmbedding:
    def test_forward_shapes(self, vocab_sizes, sample_codes):
        model = EntityEmbedding(
            vocab_sizes,
            embedding_dim=8,
            mlp_width=32,
            mlp_depth=2,
            task="classification",
            n_classes=2,
        )
        logits, embeddings = model(sample_codes)
        assert logits.shape == (32, 2)
        assert embeddings.shape == (32, 2 * 8)

    def test_get_embeddings_shape(self, vocab_sizes, sample_codes):
        model = EntityEmbedding(
            vocab_sizes, embedding_dim=8, mlp_width=32, mlp_depth=1, task="regression"
        )
        emb = model.get_embeddings(sample_codes)
        assert emb.shape == (32, 2 * 8)

    def test_regression_output_dim(self, vocab_sizes, sample_codes):
        model = EntityEmbedding(
            vocab_sizes, embedding_dim=4, mlp_width=16, mlp_depth=1, task="regression"
        )
        logits, _ = model(sample_codes)
        assert logits.shape == (32, 1)

    def test_single_column(self):
        model = EntityEmbedding(
            {"x": 50},
            embedding_dim=16,
            mlp_width=32,
            mlp_depth=1,
            task="classification",
            n_classes=3,
        )
        codes = {"x": torch.randint(0, 50, (16,))}
        logits, emb = model(codes)
        assert logits.shape == (16, 3)
        assert emb.shape == (16, 16)


class TestTabularDAE:
    def test_forward_shapes(self, vocab_sizes, sample_codes):
        model = TabularDAE(vocab_sizes, embedding_dim=8, latent_dim=16, dropout=0.1)
        recon, clean = model(sample_codes, swap_prob=0.15)
        total_dim = 2 * 8
        assert recon.shape == (32, total_dim)
        assert clean.shape == (32, total_dim)

    def test_encode_shape(self, vocab_sizes, sample_codes):
        model = TabularDAE(vocab_sizes, embedding_dim=8, latent_dim=16, dropout=0.1)
        latent = model.encode(sample_codes)
        assert latent.shape == (32, 16)

    def test_no_swap_noise(self):
        x = torch.randn(10, 8)
        out = TabularDAE._apply_swap_noise(x, swap_prob=0.0)
        assert torch.equal(x, out)

    def test_swap_noise_changes_values(self):
        torch.manual_seed(42)
        x = torch.randn(100, 8)
        out = TabularDAE._apply_swap_noise(x, swap_prob=0.5)
        assert not torch.equal(x, out)

    def test_single_column(self):
        model = TabularDAE({"x": 50}, embedding_dim=8, latent_dim=4, dropout=0.1)
        codes = {"x": torch.randint(0, 50, (16,))}
        recon, _clean = model(codes)
        assert recon.shape == (16, 8)
        latent = model.encode(codes)
        assert latent.shape == (16, 4)
