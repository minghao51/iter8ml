"""Pluggable Tracker protocol and JSONL implementation."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def log_event(self, event: dict) -> None: ...
    def finish(self) -> None: ...


class JSONLTracker:
    """Default tracker. Writes structured events to workspace/experiments.jsonl."""

    def __init__(self, log_path: str = "workspace/experiments.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_run_id = None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.log_event({"event": "metrics", "metrics": metrics, "step": step})

    def log_params(self, params: dict) -> None:
        self.log_event({"event": "params", "params": params})

    def log_artifact(self, path: str) -> None:
        self.log_event({"event": "artifact", "path": path})

    def log_event(self, event: dict) -> None:
        event["run_id"] = self.current_run_id or "unknown"
        event["timestamp"] = datetime.now(UTC).isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def finish(self) -> None:
        self.log_event({"event": "run_completed"})
        self.current_run_id = None


class WandbTracker:
    """Optional [wandb] extra. Mirrors all events to W&B run."""

    def __init__(self, project: str = "tabular-blueprint", **kwargs):
        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(project=project, **kwargs)
        except ImportError as e:
            raise ImportError("wandb is required. Install with: pip install wandb") from e

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.wandb.log(metrics, step=step)

    def log_params(self, params: dict) -> None:
        self.run.config.update(params)

    def log_artifact(self, path: str) -> None:
        artifact = self.wandb.Artifact(name=Path(path).stem, type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_event(self, event: dict) -> None:
        self.wandb.log(event)

    def finish(self) -> None:
        self.run.finish()


class MLflowTracker:
    """Optional [mlflow] extra. Logs to a local or remote MLflow server."""

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "tabular-blueprint"):
        try:
            import mlflow

            self.mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run()
        except ImportError as e:
            raise ImportError("mlflow is required. Install with: pip install mlflow") from e

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict) -> None:
        self.mlflow.log_params(params)

    def log_artifact(self, path: str) -> None:
        self.mlflow.log_artifact(path)

    def log_event(self, event: dict) -> None:
        for key, value in event.items():
            if isinstance(value, (int, float, str)):
                self.mlflow.log_param(key, str(value))
            elif isinstance(value, dict):
                self.mlflow.log_dict(value, key)

    def finish(self) -> None:
        self.mlflow.end_run()
