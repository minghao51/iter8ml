"""Constants and enums for type-safe configuration."""

from enum import Enum


class TaskType(Enum):
    """Supported machine learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class CVStrategy(Enum):
    """Cross-validation strategies."""

    KFOLD = "kfold"
    STRATIFIED = "stratified"
    TIMESERIES = "timeseries"


class ModelName(Enum):
    """Available model names."""

    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    TABPFN = "tabpfn"
    FT_TRANSFORMER = "ft_transformer"
    TABNET = "tabnet"
    NAIVE_BASELINE = "naive_baseline"
    LINEAR_BASELINE = "linear_baseline"


class TrackerType(Enum):
    """Supported tracking backends."""

    JSONL = "jsonl"
    WANDB = "wandb"
    MLFLOW = "mlflow"


# Conversion utilities for backward compatibility
def from_task_type(value: str | TaskType) -> TaskType:
    """Convert string or TaskType to TaskType enum."""
    if isinstance(value, TaskType):
        return value
    return TaskType(value)


def from_cv_strategy(value: str | CVStrategy) -> CVStrategy:
    """Convert string or CVStrategy to CVStrategy enum."""
    if isinstance(value, CVStrategy):
        return value
    return CVStrategy(value)


def from_model_name(value: str | ModelName) -> ModelName:
    """Convert string or ModelName to ModelName enum."""
    if isinstance(value, ModelName):
        return value
    return ModelName(value)


def from_tracker_type(value: str | TrackerType) -> TrackerType:
    """Convert string or TrackerType to TrackerType enum."""
    if isinstance(value, TrackerType):
        return value
    return TrackerType(value)
