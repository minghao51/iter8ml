"""DataAdapter: converts Polars DataFrames to model-specific formats."""

from typing import Literal

import numpy as np
import polars as pl


class DataAdapter:
    """
    Single point of truth for format conversion.
    Detects target format from the model type and converts accordingly.

    Supported outputs:
      - "numpy"   -> (np.ndarray, np.ndarray) for GBDTs
      - "tensor"  -> (torch.Tensor, torch.Tensor) for PyTorch models
      - "dataset" -> HuggingFace Dataset for Transformers
    """

    def __init__(self, target_format: Literal["numpy", "tensor", "dataset"] = "numpy"):
        self.target_format = target_format

    def transform(self, df: pl.DataFrame, target_col: str) -> tuple:
        """Convert DataFrame to target format and split features/target."""
        X = df.drop(target_col)
        y = df[target_col]

        if self.target_format == "numpy":
            return self._to_numpy(X, y)
        elif self.target_format == "tensor":
            return self._to_tensor(X, y)
        elif self.target_format == "dataset":
            return self._to_dataset(X, y)
        else:
            raise ValueError(f"Unsupported target format: {self.target_format}")

    def _to_numpy(self, X: pl.DataFrame, y: pl.Series) -> tuple[np.ndarray, np.ndarray]:
        """Convert to NumPy arrays."""
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        return X_np, y_np

    def _to_tensor(self, X: pl.DataFrame, y: pl.Series) -> tuple:
        """Convert to PyTorch tensors using native Polars to_torch for zero-copy."""
        import torch

        # Native Polars → PyTorch conversion (zero-copy when conditions met)
        df = X.with_columns(y.alias("_label_"))
        tensors: dict[str, torch.Tensor] = df.to_torch(
            return_type="dict",
            dtype=None,  # Preserve original dtypes, cast in torch
        )

        # Stack feature columns
        X_tensor = torch.stack([tensors[col] for col in X.columns]).T
        y_tensor = tensors["_label_"]

        # Cast to appropriate dtypes
        X_tensor = X_tensor.to(torch.float32)
        y_tensor = y_tensor.to(torch.float32 if y.dtype.is_float() else torch.long)

        return X_tensor, y_tensor

    def _to_dataset(self, X: pl.DataFrame, y: pl.Series):
        """Convert to HuggingFace Dataset using Arrow format for zero-copy."""
        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(
                "The 'datasets' package is required for target_format='dataset'. "
                "Install it with: uv sync --extra transformers"
            ) from e

        # Use PyArrow as interchange format (zero-copy when possible)
        table = X.with_columns(y.alias("label")).to_arrow()
        return Dataset.from_arrow(table)
