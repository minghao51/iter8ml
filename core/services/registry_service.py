"""Unified model registry service with file locking."""

import fcntl
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.services.report_service import resolve_primary_score
from core.utils.jsonl import load_events


@dataclass(frozen=True)
class PromotionResult:
    """Structured outcome for registry promotion attempts."""

    status: str
    message: str
    entry: dict[str, Any] | None = None


class RegistryService:
    """Thread-safe model registry with file locking."""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.lock_path = str(self.registry_path.with_suffix(".lock"))

    def load(self) -> dict[str, Any]:
        """Load registry from disk, returns empty dict if not exists."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}

    def update_if_better(
        self,
        key: str,
        model_name: str,
        run_id: str,
        score: float,
        artifact_path: str,
    ) -> bool:
        """Update registry only if new score beats existing champion."""
        with open(self.lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                registry = self.load()

                if key not in registry or score > registry[key].get("score", -float("inf")):
                    registry[key] = {
                        "model": model_name,
                        "run_id": run_id,
                        "score": score,
                        "artifact_path": artifact_path,
                        "registered_at": datetime.now(UTC).isoformat(),
                    }
                    self._save(registry)
                    return True
                return False
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get(self, key: str) -> dict[str, Any] | None:
        """Get entry by key, returns None if not found."""
        registry = self.load()
        return registry.get(key)

    def get_all(self) -> dict[str, Any]:
        """Get all registry entries."""
        return self.load()

    def promote_run(self, run_id: str, key: str, log_path: str | Path) -> PromotionResult:
        """Promote a completed run into the registry."""
        events = load_events(log_path)
        run_event = next(
            (
                event
                for event in events
                if event.get("run_id") == run_id and event.get("event") == "model_completed"
            ),
            None,
        )
        if not run_event:
            return PromotionResult(
                status="not_found",
                message=f"Run {run_id} not found.",
            )

        metric_name, score = resolve_primary_score(run_event.get("cv_scores", {}))
        updated = self.update_if_better(
            key=key,
            model_name=run_event.get("model", ""),
            run_id=run_id,
            score=score,
            artifact_path=run_event.get("artifact_path", ""),
        )
        entry = self.get(key)

        if updated:
            return PromotionResult(
                status="promoted",
                message=f"Promoted {run_id} to champion for {key} using {metric_name}.",
                entry=entry,
            )

        return PromotionResult(
            status="rejected",
            message=f"Existing champion for {key} has better score.",
            entry=entry,
        )

    def _save(self, registry: dict[str, Any]) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
