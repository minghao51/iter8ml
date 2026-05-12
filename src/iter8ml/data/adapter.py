"""DataAdapter: converts Polars DataFrames to NumPy arrays for model consumption."""

import numpy as np
import polars as pl


class DataAdapter:
    """Single point of truth for Polars -> NumPy conversion at the model boundary."""

    def transform(self, df: pl.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to NumPy arrays and split features/target."""
        X = df.drop(target_col)
        y = df[target_col]
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        return X_np, y_np
