"""DeBERTa-v3 text encoder for text-as-feature extraction."""

from typing import Any

import numpy as np
import polars as pl


class TextEncoder:
    """
    DeBERTa-v3 embedding extractor.
    Converts text columns into dense embeddings as Polars columns.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        device: str | None = None,
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if self._has_gpu() else "cpu")
        self.tokenizer: Any = None
        self.model: Any = None

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, df: pl.DataFrame, text_cols: list[str]) -> pl.DataFrame:
        """Encode text columns into embeddings and append as new columns."""
        if self.model is None:
            self._load_model()

        for col in text_cols:
            texts = df[col].to_list()
            embeddings = self._batch_encode(texts)
            emb_cols = {f"{col}_emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
            df = df.with_columns([pl.Series(name, vals) for name, vals in emb_cols.items()])

        return df

    def _batch_encode(self, texts: list[str]) -> np.ndarray:
        import torch

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model and tokenizer must be loaded before encoding")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    @property
    def encoder_name(self) -> str:
        return "DeBERTa-TextEncoder"
