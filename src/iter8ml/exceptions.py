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


class RegistryError(TabularBlueprintError):
    """Raised when registry operations fail."""


class TrainerStatePublishError(TabularBlueprintError):
    """Raised when the required trainer state publication seam fails."""


_DATA_KEYWORDS = frozenset(
    {
        "target_col",
        "file not found",
        "unsupported file format",
        "invalid json",
        "query cannot be empty",
        "only select queries",
        "destructive keywords",
        "database error",
    }
)


def track_errors(tracker_attr: str = "tracker") -> Callable[..., Any]:
    """Decorator that catches exceptions, logs them as events, and re-raises typed errors.

    Usage::

        class Trainer:
            @track_errors()
            def _train_single_model(self, ...):
                ...
    """

    def _classify(exc: Exception) -> type[TabularBlueprintError]:
        msg = str(exc).lower()
        if isinstance(exc, ValueError) and any(kw in msg for kw in _DATA_KEYWORDS):
            return DataLoadError
        return ModelFitError

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except TabularBlueprintError:
                raise
            except Exception as e:
                exc_type = _classify(e)
                tracker = getattr(self, tracker_attr, None)
                if tracker is not None:
                    tracker.log_event(
                        {
                            "event": "error",
                            "error_type": exc_type.__name__,
                            "message": str(e),
                        }
                    )
                raise exc_type(str(e), context={"original_type": type(e).__name__}) from e

        return wrapper

    return decorator
