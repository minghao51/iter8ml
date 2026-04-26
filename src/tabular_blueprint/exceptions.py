"""Custom exception hierarchy for Tabular Blueprint."""

from collections.abc import Callable
from functools import wraps
from typing import Any


class TabularBlueprintError(Exception):
    """Base exception for all Tabular Blueprint errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class DataLoadError(TabularBlueprintError):
    """Raised when data loading or validation fails."""


class ModelFitError(TabularBlueprintError):
    """Raised when model training fails."""


class RegistryError(TabularBlueprintError):
    """Raised when registry operations fail."""


def track_errors(tracker_attr: str = "tracker") -> Callable:
    """Decorator that catches exceptions, logs them as events, and re-raises typed errors.

    Usage::

        class Trainer:
            @track_errors()
            def _train_single_model(self, ...):
                ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except TabularBlueprintError:
                raise
            except ValueError as e:
                tracker = getattr(self, tracker_attr, None)
                if tracker is not None:
                    tracker.log_event(
                        {"event": "error", "error_type": "DataLoadError", "message": str(e)}
                    )
                raise DataLoadError(str(e)) from e
            except RuntimeError as e:
                tracker = getattr(self, tracker_attr, None)
                if tracker is not None:
                    tracker.log_event(
                        {"event": "error", "error_type": "ModelFitError", "message": str(e)}
                    )
                raise ModelFitError(str(e)) from e
            except Exception as e:
                tracker = getattr(self, tracker_attr, None)
                if tracker is not None:
                    tracker.log_event(
                        {
                            "event": "error",
                            "error_type": type(e).__name__,
                            "message": str(e),
                        }
                    )
                raise ModelFitError(str(e), context={"original_type": type(e).__name__}) from e

        return wrapper

    return decorator
