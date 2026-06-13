"""Constants and enums for type-safe configuration."""

from enum import Enum


class FeatureStrategy(Enum):
    """Feature engineering strategy."""

    NONE = "none"
    AFE = "afe"
    EMBEDDING = "embedding"


class TaskType(Enum):
    """Supported machine learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class CVStrategy(Enum):
    """Cross-validation strategies."""

    KFOLD = "kfold"
    STRATIFIED = "stratified"
    TIMESERIES = "timeseries"


class EmbeddingMethod(Enum):
    """Available embedding methods for high-cardinality features."""

    ENTITY = "entity"
    AUTOENCODER = "autoencoder"


class TrackerType(Enum):
    """Supported tracking backends."""

    JSONL = "jsonl"
    WANDB = "wandb"
    MLFLOW = "mlflow"
