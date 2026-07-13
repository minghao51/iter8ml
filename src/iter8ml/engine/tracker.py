"""Pluggable Tracker protocol and JSONL implementation with log rotation."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from iter8ml.workspace import Workspace


class Tracker(Protocol):
    current_run_id: str | None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def log_event(self, event: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...


class JSONLTracker:
    """Default tracker with log rotation.

    Writes structured events to workspace/experiments.jsonl.
    """

    def __init__(
        self,
        log_path: str = "workspace/experiments.jsonl",
        max_file_size_mb: float = 100.0,
        backup_count: int = 5,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.backup_count = backup_count
        self.current_run_id: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_workspace(cls, workspace: "Workspace", **kwargs: Any) -> "JSONLTracker":
        return cls(log_path=str(workspace.experiments_path), **kwargs)

    def _should_rotate(self) -> bool:
        """Check if the log file exceeds the size limit."""
        if not self.log_path.exists():
            return False
        return self.log_path.stat().st_size >= self.max_file_size_bytes

    def _rotate_log(self) -> None:
        """Rotate log files, keeping backup_count historical files."""
        if not self.log_path.exists():
            return

        # Remove oldest backup if we have too many
        oldest_backup = self.log_path.with_suffix(f".jsonl.{self.backup_count}")
        if oldest_backup.exists():
            oldest_backup.unlink()

        # Rotate existing backups: .N -> .N+1
        for i in range(self.backup_count - 1, 0, -1):
            old_backup = self.log_path.with_suffix(f".jsonl.{i}")
            new_backup = self.log_path.with_suffix(f".jsonl.{i + 1}")
            if old_backup.exists():
                old_backup.rename(new_backup)

        # Move current log to .1
        self.log_path.rename(self.log_path.with_suffix(".jsonl.1"))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.log_event({"event": "metrics", "metrics": metrics, "step": step})

    def log_params(self, params: dict[str, Any]) -> None:
        self.log_event({"event": "params", "params": params})

    def log_artifact(self, path: str) -> None:
        self.log_event({"event": "artifact", "path": path})

    def log_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            # Check if we need to rotate before writing
            if self._should_rotate():
                self._rotate_log()

            enriched_event = dict(event)
            enriched_event["run_id"] = self.current_run_id or "unknown"
            enriched_event["timestamp"] = datetime.now(UTC).isoformat()
            with open(self.log_path, "a") as f:
                f.write(json.dumps(enriched_event) + "\n")

    def finish(self) -> None:
        self.log_event({"event": "run_completed"})
        with self._lock:
            self.current_run_id = None


class WandbTracker:
    """Optional [wandb] extra. Mirrors all events to W&B run."""

    def __init__(self, project: str = "iter8ml", **kwargs: Any):
        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(project=project, **kwargs)
            self.current_run_id: str | None = self.run.id
        except ImportError as e:
            raise ImportError("wandb is required. Install with: pip install wandb") from e

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.wandb.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        self.run.config.update(params)

    def log_artifact(self, path: str) -> None:
        artifact = self.wandb.Artifact(name=Path(path).stem, type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_event(self, event: dict[str, Any]) -> None:
        self.wandb.log(event)

    def finish(self) -> None:
        self.run.finish()
        self.current_run_id = None


class MLflowTracker:
    """Optional [mlflow] extra. Logs to a local or remote MLflow server."""

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "iter8ml"):
        try:
            import mlflow

            self.mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run()
            self.current_run_id: str | None = self.run.info.run_id
        except ImportError as e:
            raise ImportError("mlflow is required. Install with: pip install mlflow") from e

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        self.mlflow.log_params(params)

    def log_artifact(self, path: str) -> None:
        self.mlflow.log_artifact(path)

    def log_event(self, event: dict[str, Any]) -> None:
        for key, value in event.items():
            if isinstance(value, (int, float, str)):
                self.mlflow.log_param(key, str(value))
            elif isinstance(value, dict):
                self.mlflow.log_dict(value, key)

    def finish(self) -> None:
        self.mlflow.end_run()
        self.current_run_id = None
