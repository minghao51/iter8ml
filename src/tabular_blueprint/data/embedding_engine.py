"""Embedding engine: detect, train, transform, and persist high-cardinality feature embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def detect_high_cardinality_columns(
    df: pl.DataFrame,
    max_categories: int = 50,
    target_col: str = "",
) -> list[str]:
    """Return column names with unique count > *max_categories*.

    Checks string, categorical, and integer columns.  Skips the target
    column if provided.
    """
    cols: list[str] = []
    for c in df.columns:
        if c == target_col:
            continue
        dtype = df[c].dtype
        is_cat = dtype in (
            pl.Categorical,
            pl.String,
            pl.Utf8,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )
        if not is_cat:
            continue
        if df[c].n_unique() > max_categories:
            cols.append(c)
    return cols


def extract_cat_codes(
    df: pl.DataFrame,
    cat_columns: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, dict]]:
    """Extract contiguous integer codes for each categorical column.

    Returns:
        codes:       ``{col_name: np.ndarray of int64, shape (n_rows,)}``
        vocab_sizes: ``{col_name: n_unique_values}``
        mappings:    ``{col_name: {original_value: contiguous_code}}``
    """
    codes: dict[str, np.ndarray] = {}
    vocab_sizes: dict[str, int] = {}
    mappings: dict[str, dict] = {}

    for col in cat_columns:
        series = df[col]
        unique_vals = series.unique().sort()
        val_to_code: dict = {v: i for i, v in enumerate(unique_vals)}
        mapping = val_to_code

        def _map_val(v: Any, _m: dict = mapping) -> int:
            return _m.get(v, 0)

        code_series = series.map_elements(_map_val, return_dtype=pl.Int64)
        codes[col] = code_series.to_numpy().astype(np.int64)
        vocab_sizes[col] = len(unique_vals)
        mappings[col] = val_to_code

    return codes, vocab_sizes, mappings


def augment_with_embeddings(
    X: np.ndarray,
    embeddings: np.ndarray,
    feature_names: list[str],
    cat_columns: list[str],
    per_col_dim: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Remove categorical columns from *X* and concatenate embedding vectors.

    Embedding feature names are formatted as
    ``"{col}_emb_{i}"`` for each dimension *i* per column.
    If *per_col_dim* does not evenly divide the embedding width, generic
    names ``"emb_{i}"`` are used instead.
    """
    cat_set = set(cat_columns)
    keep_indices = [i for i, name in enumerate(feature_names) if name not in cat_set]
    kept_names = [feature_names[i] for i in keep_indices]
    X_kept = X[:, keep_indices] if keep_indices else np.empty((X.shape[0], 0), dtype=X.dtype)

    n_emb_cols = embeddings.shape[1]
    emb_names: list[str] = []

    if per_col_dim and per_col_dim > 0 and len(cat_columns) * per_col_dim == n_emb_cols:
        for col in sorted(cat_columns):
            emb_names.extend(f"{col}_emb_{d}" for d in range(per_col_dim))
    else:
        emb_names = [f"emb_{i}" for i in range(n_emb_cols)]

    if X_kept.shape[1] > 0:
        X_aug = np.hstack([X_kept, embeddings.astype(X.dtype)])
    else:
        X_aug = embeddings.astype(X.dtype)

    return X_aug, kept_names + emb_names


class EmbeddingEngine:
    def __init__(
        self,
        task: str,
        workspace_dir: str | Path,
        embedding_method: str = "entity",
        embedding_dim: int = 16,
        embedding_max_categories: int = 50,
        embedding_epochs: int = 10,
        embedding_lr: float = 1e-3,
        embedding_mlp_width: int = 128,
        embedding_mlp_depth: int = 2,
        embedding_ae_latent_dim: int = 32,
        embedding_ae_dropout: float = 0.2,
        random_seed: int = 42,
    ):
        self._task = task
        self._workspace_dir = Path(workspace_dir)
        self._embedding_method = embedding_method
        self._embedding_dim = embedding_dim
        self._embedding_max_categories = embedding_max_categories
        self._embedding_epochs = embedding_epochs
        self._embedding_lr = embedding_lr
        self._embedding_mlp_width = embedding_mlp_width
        self._embedding_mlp_depth = embedding_mlp_depth
        self._embedding_ae_latent_dim = embedding_ae_latent_dim
        self._embedding_ae_dropout = embedding_ae_dropout
        self._random_seed = random_seed
        self._model: Any = None
        self._cat_columns: list[str] = []
        self._vocab_sizes: dict[str, int] = {}
        self._mappings: dict[str, dict] = {}
        self._embed_dim: int = 0

    def fit_transform(
        self,
        df: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        target_col: str,
        run_id: str = "",
        data_hash: str = "",
    ) -> tuple[np.ndarray, list[str]]:
        if not isinstance(df, pl.DataFrame):
            return X, feature_names

        cat_columns = detect_high_cardinality_columns(
            df,
            max_categories=self._embedding_max_categories,
            target_col=target_col,
        )
        if not cat_columns:
            return X, feature_names

        codes, vocab_sizes, mappings = extract_cat_codes(df, cat_columns)
        self._cat_columns = cat_columns
        self._vocab_sizes = vocab_sizes
        self._mappings = mappings

        method = self._embedding_method
        if method == "entity":
            model, embed_dim = self._train_entity(codes, y, vocab_sizes)
        elif method == "autoencoder":
            model, embed_dim = self._train_autoencoder(codes, vocab_sizes)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        self._model = model
        self._embed_dim = embed_dim

        embeddings_np = self._generate_embeddings(model, codes, method, embed_dim)
        per_col_dim = self._embedding_dim if method == "entity" else None
        X_aug, aug_names = augment_with_embeddings(
            X,
            embeddings_np,
            feature_names,
            cat_columns,
            per_col_dim=per_col_dim,
        )

        if run_id:
            self._save(run_id)

        return X_aug, aug_names

    def _train_entity(
        self,
        codes: dict[str, np.ndarray],
        y: np.ndarray,
        vocab_sizes: dict[str, int],
    ) -> tuple[Any, int]:
        import torch
        import torch.nn as nn
        import torch.utils.data as torch_data

        from tabular_blueprint.models.deep.sparse_embedder import EntityEmbedding

        task = self._task
        n_classes = len(np.unique(y)) if task == "classification" else 1
        if task == "classification" and n_classes == 2:
            n_classes = 1

        model = EntityEmbedding(
            vocab_sizes=vocab_sizes,
            embedding_dim=self._embedding_dim,
            mlp_width=self._embedding_mlp_width,
            mlp_depth=self._embedding_mlp_depth,
            task=task,
            n_classes=n_classes,
        )

        device = torch.device("cpu")
        model.to(device)

        sorted_cols = sorted(codes.keys())
        cat_tensors = [torch.from_numpy(codes[c]).long().to(device) for c in sorted_cols]
        y_tensor = torch.from_numpy(y).float().to(device)
        if task == "classification":
            y_tensor = y_tensor.long()

        dataset = torch_data.TensorDataset(y_tensor, *cat_tensors)
        loader = torch_data.DataLoader(dataset, batch_size=256, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self._embedding_lr)

        for _epoch in range(self._embedding_epochs):
            model.train()
            for batch in loader:
                batch_y = batch[0]
                batch_cats = {c: batch[i + 1] for i, c in enumerate(sorted_cols)}
                optimizer.zero_grad()
                logits, _ = model(batch_cats)
                if task == "classification":
                    if n_classes == 1:
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits.squeeze(-1), batch_y.float()
                        )
                    else:
                        loss = nn.functional.cross_entropy(logits, batch_y.long())
                else:
                    loss = nn.functional.mse_loss(logits.squeeze(-1), batch_y)
                loss.backward()
                optimizer.step()

        model._update_oov_means()
        model.eval()
        return model, len(sorted_cols) * self._embedding_dim

    def _train_autoencoder(
        self,
        codes: dict[str, np.ndarray],
        vocab_sizes: dict[str, int],
    ) -> tuple[Any, int]:
        import torch
        import torch.utils.data as torch_data

        from tabular_blueprint.models.deep.sparse_embedder import TabularDAE

        model = TabularDAE(
            vocab_sizes=vocab_sizes,
            embedding_dim=self._embedding_dim,
            latent_dim=self._embedding_ae_latent_dim,
            dropout=self._embedding_ae_dropout,
        )

        device = torch.device("cpu")
        model.to(device)

        sorted_cols = sorted(codes.keys())
        cat_tensors = [torch.from_numpy(codes[c]).long().to(device) for c in sorted_cols]

        dataset = torch_data.TensorDataset(*cat_tensors)
        loader = torch_data.DataLoader(dataset, batch_size=256, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self._embedding_lr)

        for _epoch in range(self._embedding_epochs):
            model.train()
            for batch in loader:
                batch_cats = {c: batch[i] for i, c in enumerate(sorted_cols)}
                optimizer.zero_grad()
                reconstruction, clean = model(batch_cats)
                loss = torch.nn.functional.mse_loss(reconstruction, clean)
                loss.backward()
                optimizer.step()

        model._update_oov_means()
        model.eval()
        return model, self._embedding_ae_latent_dim

    def _generate_embeddings(
        self,
        model: Any,
        codes: dict[str, np.ndarray],
        method: str,
        embed_dim: int,
    ) -> np.ndarray:
        import torch

        device = torch.device("cpu")
        sorted_cols = sorted(codes.keys())

        batch_size = 4096
        n_rows = codes[sorted_cols[0]].shape[0]
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                cat_dict = {
                    c: torch.from_numpy(codes[c][start:end]).long().to(device) for c in sorted_cols
                }
                if method == "entity":
                    emb = model.get_embeddings(cat_dict)
                else:
                    emb = model.encode(cat_dict)
                all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def _save(self, run_id: str) -> None:
        import torch

        save_dir = self._workspace_dir / "embeddings"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{run_id}.pt"
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "model_class": type(self._model).__name__,
                "vocab_sizes": self._vocab_sizes,
                "cat_columns": self._cat_columns,
                "embed_dim": self._embed_dim,
            },
            str(model_path),
        )

        mappings_path = save_dir / f"{run_id}_mappings.json"
        serializable: dict[str, dict] = {}
        for col, mapping in self._mappings.items():
            serializable[col] = {str(k): int(v) for k, v in mapping.items()}
        mappings_path.write_text(json.dumps(serializable, indent=2))
