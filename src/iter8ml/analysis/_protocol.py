"""Shared drift-detection contract: one report base + one detector protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class DriftReportBase(BaseModel):
    """Common base for all drift detector reports.

    ``method`` tags which detector produced the report, letting the engine
    and callers treat every report uniformly behind a single type.
    """

    drift_detected: bool
    method: str = ""


@runtime_checkable
class DriftDetectorProtocol(Protocol):
    """Uniform contract implemented by every drift detector.

    A detector is constructed with a reference frame and produces a
    :class:`DriftReportBase` (or subclass) from a live/new frame via
    ``detect``.
    """

    def detect(self, live_df: Any) -> DriftReportBase: ...
