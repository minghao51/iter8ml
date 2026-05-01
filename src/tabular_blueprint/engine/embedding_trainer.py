"""Embedding orchestrator: detect, train, transform, persist."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.constants import EmbeddingMethod
from tabular_blueprint.data.embedding_engine import (
    augment_with_embeddings,
    detect_high_cardinality_columns,
    extract_cat_codes,
)
from tabular_blueprint.engine.tracker import Tracker


class EmbeddingEngine:
    """Fit / transform orchestrator for high-cardinality feature embeddings."""

    def __init__(self, config: ExperimentConfig, tracker: Tracker) -> None:
        self.config = config
        self.tracker = tracker
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
        run_id: str,
        data_hash: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Detect high-cardinality columns, train embeddings, augment features."""
        import polars as pl

        if not isinstance(df, pl.DataFrame):
            return X, feature_names

        cat_columns = detect_high_cardinality_columns(
            df,
            max_categories=self.config.embedding_max_categories,
            target_col=target_col,
        )
        if not cat_columns:
            self.tracker.log_event(
                {
                    "event": "embedding_skipped",
                    "run_id": run_id,
                    "reason": "no_high_cardinality_columns",
                }
            )
            return X, feature_names

        codes, vocab_sizes, mappings = extract_cat_codes(df, cat_columns)
        self._cat_columns = cat_columns
        self._vocab_sizes = vocab_sizes
        self._mappings = mappings

        method = self.config.embedding_method
        if method == EmbeddingMethod.ENTITY:
            model, embed_dim = self._train_entity(codes, y, vocab_sizes)
        elif method == EmbeddingMethod.AUTOENCODER:
            model, embed_dim = self._train_autoencoder(codes, vocab_sizes)
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        self._model = model
        self._embed_dim = embed_dim

        embeddings_np = self._generate_embeddings(model, codes, method, embed_dim)
        per_col_dim = self.config.embedding_dim if method == EmbeddingMethod.ENTITY else None
        X_aug, aug_names = augment_with_embeddings(
            X,
            embeddings_np,
            feature_names,
            cat_columns,
            per_col_dim=per_col_dim,
        )

        self._save(run_id)

        self.tracker.log_event(
            {
                "event": "embedding_completed",
                "run_id": run_id,
                "data_hash": data_hash,
                "method": method.value,
                "n_cat_columns": len(cat_columns),
                "cat_columns": cat_columns,
                "vocab_sizes": {k: int(v) for k, v in vocab_sizes.items()},
                "embed_dim": embed_dim,
                "output_features": len(aug_names),
            }
        )

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

        cfg = self.config
        task = cfg.task.value
        n_classes = len(np.unique(y)) if task == "classification" else 1
        if task == "classification" and n_classes == 2:
            n_classes = 1

        model = EntityEmbedding(
            vocab_sizes=vocab_sizes,
            embedding_dim=cfg.embedding_dim,
            mlp_width=cfg.embedding_mlp_width,
            mlp_depth=cfg.embedding_mlp_depth,
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

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.embedding_lr)

        for _epoch in range(cfg.embedding_epochs):
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
        return model, len(sorted_cols) * cfg.embedding_dim

    def _train_autoencoder(
        self,
        codes: dict[str, np.ndarray],
        vocab_sizes: dict[str, int],
    ) -> tuple[Any, int]:
        import torch
        import torch.utils.data as torch_data

        from tabular_blueprint.models.deep.sparse_embedder import TabularDAE

        cfg = self.config

        model = TabularDAE(
            vocab_sizes=vocab_sizes,
            embedding_dim=cfg.embedding_dim,
            latent_dim=cfg.embedding_ae_latent_dim,
            dropout=cfg.embedding_ae_dropout,
        )

        device = torch.device("cpu")
        model.to(device)

        sorted_cols = sorted(codes.keys())
        cat_tensors = [torch.from_numpy(codes[c]).long().to(device) for c in sorted_cols]

        dataset = torch_data.TensorDataset(*cat_tensors)
        loader = torch_data.DataLoader(dataset, batch_size=256, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.embedding_lr)

        for _epoch in range(cfg.embedding_epochs):
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
        return model, cfg.embedding_ae_latent_dim

    def _generate_embeddings(
        self,
        model: Any,
        codes: dict[str, np.ndarray],
        method: EmbeddingMethod,
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
                if method == EmbeddingMethod.ENTITY:
                    emb = model.get_embeddings(cat_dict)
                else:
                    emb = model.encode(cat_dict)
                all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def _save(self, run_id: str) -> None:
        import torch

        save_dir = self.config.workspace_dir / "embeddings"
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
