from __future__ import annotations

from collections.abc import Callable
from typing import Any


def hamilton_config() -> Any:
    try:
        from hamilton.function_modifiers import config
    except ImportError:
        return None
    return config


def hamilton_stub(feature: str) -> Callable[..., None]:
    def _stub(**_kwargs: Any) -> None:
        raise ImportError(
            f"Hamilton is required for {feature}. Install with: pip install sf-hamilton"
        )

    _stub.__name__ = f"<hamilton_stub:{feature}>"
    return _stub
