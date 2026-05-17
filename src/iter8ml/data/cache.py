"""Preprocessing cache: avoid recomputing data transformations across runs."""

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from iter8ml.config import ExperimentConfig
    from iter8ml.workspace import Workspace

_CACHE_DIR = ".iter8ml/cache"


def _cache_key(data_hash: str, config: "ExperimentConfig") -> str:
    """Deterministic cache key from data hash + config."""
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
    return f"{data_hash}_{config_hash}"


def _array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


class PreprocessingCache:
    def __init__(self, workspace: "Workspace") -> None:
        self.cache_dir = workspace.root / _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self, data_hash: str, config: "ExperimentConfig"
    ) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
        """Load cached preprocessing result. Returns (X, y, feature_names) or None."""
        key = _cache_key(data_hash, config)
        meta_path = self.cache_dir / f"{key}_meta.json"
        x_path = self.cache_dir / f"{key}_X.npy"
        y_path = self.cache_dir / f"{key}_y.npy"

        if not meta_path.exists() or not x_path.exists() or not y_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            X = np.load(x_path)
            y = np.load(y_path)
            if _array_hash(X) != meta.get("x_hash", "") or _array_hash(y) != meta.get("y_hash", ""):
                return None
            feature_names: list[str] = meta["feature_names"]
            return X, y, feature_names
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def save(
        self,
        data_hash: str,
        config: "ExperimentConfig",
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Save preprocessed data to cache."""
        key = _cache_key(data_hash, config)
        meta_path = self.cache_dir / f"{key}_meta.json"
        x_path = self.cache_dir / f"{key}_X.npy"
        y_path = self.cache_dir / f"{key}_y.npy"

        np.save(x_path, X)
        np.save(y_path, y)
        meta_path.write_text(
            json.dumps(
                {
                    "feature_names": feature_names,
                    "x_hash": _array_hash(X),
                    "y_hash": _array_hash(y),
                }
            )
        )

    def clear(self) -> int:
        """Remove all cached files. Returns count of deleted entries."""
        count = 0
        for f in self.cache_dir.glob("*"):
            if f.is_file():
                f.unlink()
                count += 1
        return count
