"""TabNet model wrapper using pytorch-tabular.

Note: uses pandas internally because pytorch-tabular requires DataFrames.
This is a deep-model-only dependency, gated behind the `[deep]` extras group.
"""

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


class TabNetModel:
    def __init__(self, task: str = "classification", **kwargs: Any):
        self.task = task
        self.params = kwargs
        self.model: Any = None

    def _build_model(self, n_features: int, n_classes: int | None = None) -> Any:
        try:
            from pytorch_tabular import TabularModel
            from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        except ImportError as e:
            raise ImportError(
                "pytorch-tabular is required for TabNet. Install it with: uv add pytorch-tabular"
            ) from e

        data_config = DataConfig(
            target=[],
            continuous_cols=[f"f_{i}" for i in range(n_features)],
            categorical_cols=[],
        )

        trainer_config = TrainerConfig(
            max_epochs=self.params.get("n_epochs", 50),
            batch_size=self.params.get("batch_size", 256),
            early_stopping="valid_loss",
            early_stopping_patience=10,
        )

        optimizer_config = OptimizerConfig(
            optimizer_params={"lr": self.params.get("learning_rate", 1e-3)},
        )

        model_config = self._get_model_config()

        return TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

    def _get_model_config(self) -> Any:
        from pytorch_tabular.models import TabNetModelConfig

        return TabNetModelConfig(
            task=self.task,
            learning_rate=self.params.get("learning_rate", 1e-3),
        )

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge per-model hyperparameter overrides into self.params."""
        self.params.update(overrides)

    def _to_dataframe(
        self, X: np.ndarray, include_target: np.ndarray | None = None
    ) -> "pd.DataFrame":
        if pd is None:
            raise ImportError("pandas is required by pytorch-tabular. Install with: uv add pandas")
        col_names = [f"f_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=col_names)
        if include_target is not None:
            df["target"] = include_target
        return df

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        n_features = X.shape[1]
        n_classes = len(np.unique(y)) if self.task == "classification" else None
        self.model = self._build_model(n_features, n_classes)

        df = self._to_dataframe(X, include_target=y)
        self.model.fit(train=df, target_col=["target"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        df = self._to_dataframe(X)
        result = self.model.predict(df)
        return result.iloc[:, 0].to_numpy()  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task != "classification":
            return None
        if self.model is None:
            raise ValueError("Model not fitted")
        df = self._to_dataframe(X)
        result = self.model.predict(df)
        proba_cols = [c for c in result.columns if c.startswith("probability")]
        if proba_cols:
            return result[proba_cols].to_numpy()  # type: ignore[no-any-return]
        return None

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        try:
            from pytorch_tabular import TabularModel
        except ImportError as e:
            raise ImportError("pytorch-tabular is required.") from e

        self.model = TabularModel.load_from_checkpoint(path)

    @property
    def model_name(self) -> str:
        return "TabNet"
