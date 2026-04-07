"""FT-Transformer wrapper for PyTorch-based tabular learning."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configs.model_configs import FTTransformerConfig


class FTTransformerModel:
    """
    FT-Transformer implementation with HuggingFace accelerate support.
    VRAM-gated entry via ModelSelector.
    """

    def __init__(
        self,
        task: str = "classification",
        n_features: int = 10,
        n_classes: int = 2,
        config: FTTransformerConfig | None = None,
    ):
        self.task = task
        self.n_features = n_features
        self.n_classes = n_classes
        self.config = config or FTTransformerConfig()
        # Extract values from config
        self.n_heads = self.config.n_heads
        self.d_hidden = self.config.d_hidden
        self.n_layers = self.config.n_layers
        self.dropout = self.config.dropout
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.n_epochs = self.config.n_epochs
        self.random_seed = self.config.random_seed
        self.model: nn.Module | None = None
        self.accelerator = None

    def _build_model(self) -> nn.Module:
        return _FTTransformer(
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_heads=self.n_heads,
            d_hidden=self.d_hidden,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        from accelerate import Accelerator

        torch.manual_seed(self.random_seed)
        self.accelerator = Accelerator()

        self.model = self._build_model()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(
            y,
            dtype=torch.float32 if self.task == "regression" else torch.long,
        )

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()

        self.model, optimizer, loader = self.accelerator.prepare(self.model, optimizer, loader)

        self.model.train()
        for _ in range(self.n_epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                if self.task == "regression":
                    batch_y = batch_y.unsqueeze(1)
                loss = criterion(output, batch_y)
                self.accelerator.backward(loss)
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=next(self.model.parameters()).device)
        with torch.no_grad():
            output = self.model(X_tensor)
        if self.task == "classification":
            preds = output.argmax(dim=1)
            return preds.cpu().numpy()
        return output.squeeze().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task != "classification":
            return None
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=next(self.model.parameters()).device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return torch.softmax(output, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.accelerator is not None:
            self.accelerator.save(self.model.state_dict(), path)
        else:
            torch.save(
                {
                    "model_state": (self.model.state_dict() if self.model is not None else None),
                    "n_features": self.n_features,
                    "n_classes": self.n_classes,
                },
                path,
            )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.n_features = checkpoint["n_features"]
        self.n_classes = checkpoint["n_classes"]
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.accelerator = None

    @property
    def model_name(self) -> str:
        return "FT-Transformer"


class _FTTransformer(nn.Module):
    """Simplified FT-Transformer architecture."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_heads: int = 4,
        d_hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_embed = nn.Linear(n_features, d_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_embed(x).unsqueeze(1)
        x = self.transformer(x)
        return self.head(x.squeeze(1))
