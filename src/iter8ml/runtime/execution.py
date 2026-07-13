"""Stable result contract for a compiled Hamilton stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageExecutionResult:
    stage: str
    status: str
    values: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    error: str | None = None
