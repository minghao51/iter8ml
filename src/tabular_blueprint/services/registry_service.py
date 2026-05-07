"""Unified model registry service with cross-platform file locking."""

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel

from tabular_blueprint.services.report_service import metric_value_is_better, resolve_primary_score
from tabular_blueprint.utils.jsonl import iter_events


class PromotionResult(BaseModel):
    """Result of attempting to promote a run to champion status."""

    status: str
    message: str
    entry: dict[str, Any] | None = None
    selected_model: str | None = None
    selected_metric: str | None = None
    selected_score: float | None = None


class RegistryService:
    """Thread-safe and process-safe model registry with file locking.

    Locking contract:
    - Public methods (update_if_better, get, get_all, promote_run) acquire
      the file lock internally via ``_acquire_lock()``.
    - ``_update_if_better_locked`` MUST only be called from within a lock
      context (i.e. by a method that has already called ``_acquire_lock``).
      It does NOT acquire the lock itself — the caller is responsible.
    - ``load()`` and ``_save()`` are NOT thread-safe on their own; they
      must be called from within a lock context.
    """

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.lock_path = str(self.registry_path.with_suffix(".lock"))

    def _acquire_lock(self) -> FileLock:
        return FileLock(self.lock_path)

    def load(self) -> dict[str, Any]:
        """Load registry from disk, returns empty dict if not exists."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return {}

    def update_if_better(
        self,
        key: str,
        model_name: str,
        run_id: str,
        score: float,
        artifact_path: str,
        metric_name: str | None = None,
    ) -> bool:
        """Update registry only if new score beats existing champion."""
        with self._acquire_lock():
            registry = self.load()
            existing_score = registry.get(key, {}).get("score")

            if (
                key not in registry
                or existing_score is None
                or metric_value_is_better(metric_name, score, existing_score)
            ):
                registry[key] = {
                    "model": model_name,
                    "run_id": run_id,
                    "score": score,
                    "metric_name": metric_name,
                    "artifact_path": artifact_path,
                    "registered_at": datetime.now(UTC).isoformat(),
                }
                self._save(registry)
                return True
            return False

    def get(self, key: str) -> dict[str, Any] | None:
        """Get entry by key, returns None if not found."""
        with self._acquire_lock():
            registry = self.load()
            return registry.get(key)

    def get_all(self) -> dict[str, Any]:
        """Get all registry entries."""
        with self._acquire_lock():
            return self.load()

    def promote_run(self, run_id: str, key: str, log_path: str | Path) -> PromotionResult:
        """Promote a completed run into the registry."""
        run_events = [
            event
            for event in iter_events(log_path)
            if event.get("run_id") == run_id and event.get("event") == "model_completed"
        ]
        run_event = self._select_best_run_event(run_events)
        if run_event is None:
            return PromotionResult(
                status="not_found",
                message=f"Run {run_id} not found.",
            )

        metric_name, score = resolve_primary_score(run_event.get("cv_scores", {}))

        with self._acquire_lock():
            updated = self._update_if_better_locked(
                key=key,
                model_name=run_event.get("model", ""),
                run_id=run_id,
                score=score,
                artifact_path=run_event.get("artifact_path", ""),
                metric_name=metric_name,
            )
            entry = self.load().get(key)

        if updated:
            return PromotionResult(
                status="promoted",
                message=f"Promoted {run_id} to champion for {key} using {metric_name}.",
                entry=entry,
                selected_model=run_event.get("model"),
                selected_metric=metric_name,
                selected_score=score,
            )

        return PromotionResult(
            status="rejected",
            message=f"Existing champion for {key} has better score.",
            entry=entry,
            selected_model=run_event.get("model"),
            selected_metric=metric_name,
            selected_score=score,
        )

    def _select_best_run_event(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not events:
            return None
        best_event: dict[str, Any] | None = None
        best_metric: str | None = None
        best_score: float | None = None
        best_timestamp: datetime | None = None
        for event in events:
            metric_name, score = resolve_primary_score(event.get("cv_scores", {}))
            timestamp = self._parse_timestamp(event.get("timestamp"))
            if best_event is None:
                best_event = event
                best_metric = metric_name
                best_score = score
                best_timestamp = timestamp
                continue

            assert best_score is not None
            assert best_metric is not None
            assert best_timestamp is not None
            if metric_value_is_better(metric_name, score, best_score):
                best_event = event
                best_metric = metric_name
                best_score = score
                best_timestamp = timestamp
                continue
            if score == best_score and timestamp > best_timestamp:
                best_event = event
                best_metric = metric_name
                best_score = score
                best_timestamp = timestamp
        return best_event

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                return datetime.fromisoformat(value)
        return datetime.min.replace(tzinfo=UTC)

    def _update_if_better_locked(
        self,
        key: str,
        model_name: str,
        run_id: str,
        score: float,
        artifact_path: str,
        metric_name: str | None = None,
    ) -> bool:
        """Update registry if new score beats existing champion.

        NOTE: Caller MUST already hold the file lock (acquired via
        ``_acquire_lock()``). This method does NOT acquire the lock itself.
        """
        registry = self.load()
        existing_score = registry.get(key, {}).get("score")

        if (
            key not in registry
            or existing_score is None
            or metric_value_is_better(metric_name, score, existing_score)
        ):
            registry[key] = {
                "model": model_name,
                "run_id": run_id,
                "score": score,
                "metric_name": metric_name,
                "artifact_path": artifact_path,
                "registered_at": datetime.now(UTC).isoformat(),
            }
            self._save(registry)
            return True
        return False

    def _save(self, registry: dict[str, Any]) -> None:
        """Save registry to disk atomically using temp file + rename."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.registry_path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            os.replace(tmp_path, str(self.registry_path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
