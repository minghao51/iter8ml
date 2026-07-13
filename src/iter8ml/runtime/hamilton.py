"""Hamilton driver factory with an explicit capability error."""

from __future__ import annotations

from typing import Any

from iter8ml.domain.hashing import digest
from iter8ml.exceptions import HamiltonUnavailableError


def build_driver(*modules: Any, config: dict[str, Any] | None = None) -> Any:
    try:
        from hamilton import driver
    except ImportError as exc:
        raise HamiltonUnavailableError(
            "Hamilton is required for DAG compilation. Install with `uv sync --extra train`."
        ) from exc
    builder = driver.Builder().with_modules(*modules)
    if config:
        builder = builder.with_config(config)
    return builder.build()


def graph_version(*modules: Any) -> str:
    return digest([getattr(module, "__name__", str(module)) for module in modules])
