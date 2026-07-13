"""Scheduler-neutral protocol for future remote orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from iter8ml.domain.manifests import RunPlan


@dataclass(frozen=True)
class RunHandle:
    run_id: str


class Orchestrator(Protocol):
    def submit(self, plan: RunPlan) -> RunHandle: ...
    def status(self, run_id: str) -> dict[str, Any]: ...
    def cancel(self, run_id: str) -> None: ...
