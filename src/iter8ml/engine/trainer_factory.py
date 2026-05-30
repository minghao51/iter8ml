"""Default adapter factory for Trainer seams."""

from __future__ import annotations

from typing import Any, Protocol

from iter8ml.engine.state_observer import StateObserver


class TrainerEventAdapter(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...


class TrainerStateAdapter(Protocol):
    def publish(self) -> str: ...


class TrackerEventAdapter:
    """Adapter that publishes trainer events through a Tracker."""

    def __init__(self, tracker: Any):
        self._tracker = tracker

    def publish(self, event: dict[str, Any]) -> None:
        self._tracker.log_event(event)


class ObserverStateAdapter:
    """Adapter that publishes workspace state via StateObserver."""

    def __init__(self, observer: StateObserver):
        self._observer = observer

    def publish(self) -> str:
        return self._observer.generate()


def build_trainer_event_adapter(tracker: Any) -> TrainerEventAdapter:
    return TrackerEventAdapter(tracker)


def build_trainer_state_adapter(
    *,
    workspace: Any,
    llm_enabled: bool,
    llm_model: str | None,
) -> TrainerStateAdapter:
    observer = StateObserver(
        workspace=workspace,
        llm_enabled=llm_enabled,
        llm_model=llm_model,
    )
    return ObserverStateAdapter(observer)
