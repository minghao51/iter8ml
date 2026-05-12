from __future__ import annotations

from typing import Any

from iter8ml.engine.tracker import Tracker


class TrackingHook:
    def __init__(self, tracker: Tracker, run_id: str | None = None) -> None:
        self._tracker = tracker
        self._run_id = run_id

    def run_before_node_execution(
        self,
        node: Any,
        context: Any,
        execution_context: Any,
    ) -> None:
        pass

    def run_after_node_execution(
        self,
        node: Any,
        result: Any,
        context: Any,
        execution_context: Any,
    ) -> None:
        pass

    def run_on_node_error(
        self,
        node: Any,
        exception: Exception,
        context: Any,
        execution_context: Any,
    ) -> None:
        node_name = getattr(node, "name", str(node))
        self._tracker.log_event(
            {
                "event": "node_error",
                "node": node_name,
                "error": str(exception),
            }
        )

    def run_on_node_success(
        self,
        node: Any,
        result: Any,
        context: Any,
        execution_context: Any,
    ) -> None:
        node_name = getattr(node, "name", str(node))
        duration = getattr(execution_context, "duration", None)
        event: dict[str, Any] = {
            "event": "node_completed",
            "node": node_name,
        }
        if duration is not None:
            event["duration_seconds"] = round(duration, 4)
        self._tracker.log_event(event)
