"""Custom exception hierarchy for iter8ml."""

from collections.abc import Callable
from functools import wraps
from typing import Any


class TabularBlueprintError(Exception):
    """Base exception for all iter8ml errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class DataLoadError(TabularBlueprintError):
    """Raised when data loading or validation fails."""


class ModelFitError(TabularBlueprintError):
    """Raised when model training fails."""


class ModelNotFittedError(ValueError):
    """Raised when predict/save is called on an unfitted model."""


class RegistryError(TabularBlueprintError):
    """Raised when registry operations fail."""


class TrainerStatePublishError(TabularBlueprintError):
    """Raised when the required trainer state publication seam fails."""


class HamiltonUnavailableError(TabularBlueprintError):
    """Raised when a Hamilton-backed operation is requested without Hamilton."""


class ArtifactError(TabularBlueprintError):
    """Raised when a durable artifact cannot be created, verified, or read."""


class CancellationRequested(TabularBlueprintError):
    """Raised at a stage boundary after a local cancellation request."""


Iter8MLError = TabularBlueprintError


def track_errors(error_cls: type[TabularBlueprintError]) -> Callable[..., Any]:
    """Decorator that catches exceptions and re-raises as a typed error.

    Works on both standalone functions and instance methods.  Existing
    ``TabularBlueprintError`` subclasses pass through unchanged.

    Usage::

        @track_errors(DataLoadError)
        def load_data(path): ...

        @track_errors(ModelFitError)
        def fit(self, X, y): ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except TabularBlueprintError:
                raise
            except Exception as e:
                raise error_cls(
                    str(e),
                    context={"original_type": type(e).__name__},
                ) from e

        return wrapper

    return decorator
