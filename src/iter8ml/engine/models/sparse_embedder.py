"""PyTorch modules for learning dense embeddings from sparse high-cardinality features."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _check_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for embedding models. Install with: uv sync --extra deep"
        )


class EntityEmbedding(nn.Module):
    """Per-column embedding tables with a configurable MLP training head.

    Each high-cardinality column gets its own ``nn.Embedding`` lookup table.
    The concatenated embeddings are fed through an MLP head that is trained
    on the supervised target.  After training, call :meth:`get_embeddings`
    to extract the dense vectors for downstream use.
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        embedding_dim: int = 16,
        mlp_width: int = 128,
        mlp_depth: int = 2,
        task: str = "classification",
        n_classes: int = 1,
    ) -> None:
        _check_torch()
        super().__init__()
        self._column_order = sorted(vocab_sizes.keys())
        self.embedding_dim = embedding_dim
        self.task = task
        self.n_classes = n_classes

        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(int(vocab_sizes[col]), embedding_dim) for col in self._column_order}
        )

        total_emb_dim = len(self._column_order) * embedding_dim
        output_dim = n_classes if task == "classification" else 1

        layers: list[nn.Module] = []
        in_dim = total_emb_dim
        for _ in range(mlp_depth):
            layers.append(nn.Linear(in_dim, mlp_width))
            layers.append(nn.ReLU())
            in_dim = mlp_width
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, cat_codes: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        emb_list = [self.embeddings[col](cat_codes[col]) for col in self._column_order]
        concatenated = torch.cat(emb_list, dim=1)
        logits = self.mlp_head(concatenated)
        return logits, concatenated

    def get_embeddings(self, cat_codes: dict[str, torch.Tensor]) -> torch.Tensor:
        emb_list = [self.embeddings[col](cat_codes[col]) for col in self._column_order]
        return torch.cat(emb_list, dim=1)


class TabularDAE(nn.Module):
    """Denoising autoencoder for sparse high-cardinality categorical features.

    Architecture:
      1. Entity-embed each categorical column (same lookup tables as EntityEmbedding).
      2. Concatenate embeddings and apply swap noise.
      3. Encode through a bottleneck to a low-dimensional latent space.
      4. Decode back to the concatenated embedding dimension.

    Trained unsupervised (no ``y`` required).  After training, call
    :meth:`encode` to extract the latent vectors.
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        embedding_dim: int = 16,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        _check_torch()
        super().__init__()
        self._column_order = sorted(vocab_sizes.keys())
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(int(vocab_sizes[col]), embedding_dim) for col in self._column_order}
        )

        total_emb_dim = len(self._column_order) * embedding_dim
        hidden_dim = max(total_emb_dim // 2, 64)

        self.encoder = nn.Sequential(
            nn.Linear(total_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_emb_dim),
        )

    def _embed_and_concat(self, cat_codes: dict[str, torch.Tensor]) -> torch.Tensor:
        emb_list = [self.embeddings[col](cat_codes[col]) for col in self._column_order]
        return torch.cat(emb_list, dim=1)

    @staticmethod
    def _apply_swap_noise(x: torch.Tensor, swap_prob: float = 0.15) -> torch.Tensor:
        if swap_prob <= 0.0:
            return x
        mask = torch.rand_like(x) < swap_prob
        perm = torch.randperm(x.size(0), device=x.device)
        return torch.where(mask, x[perm], x)

    def forward(
        self,
        cat_codes: dict[str, torch.Tensor],
        swap_prob: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self._embed_and_concat(cat_codes)
        noisy = self._apply_swap_noise(clean, swap_prob)
        latent = self.encoder(noisy)
        reconstruction = self.decoder(latent)
        return reconstruction, clean

    def encode(self, cat_codes: dict[str, torch.Tensor]) -> torch.Tensor:
        clean = self._embed_and_concat(cat_codes)
        return self.encoder(clean)
