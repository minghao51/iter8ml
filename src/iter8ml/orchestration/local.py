"""Synchronous local implementation of the scheduler-neutral seam."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from iter8ml.domain.manifests import RunPlan
from iter8ml.orchestration.protocol import RunHandle
from iter8ml.workspace import Workspace


class LocalOrchestrator:
    def __init__(self, workspace: Workspace, runner: Callable[[RunPlan], Any] | None = None):
        self.workspace = workspace.init()
        self.runner = runner

    def submit(self, plan: RunPlan) -> RunHandle:
        handle = RunHandle(run_id=f"planned_{plan.run_key[7:19]}")
        if self.runner is not None:
            self.runner(plan)
        return handle

    def status(self, run_id: str) -> dict[str, Any]:
        path = self.workspace.runs_dir / run_id / "run.json"
        if not path.exists():
            return {"run_id": run_id, "status": "unknown"}
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def cancel(self, run_id: str) -> None:
        path = self.workspace.runs_dir / run_id / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") in {"succeeded", "failed", "cancelled"}:
            return
        data["status"] = "cancelled"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
