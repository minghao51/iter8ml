"""Trainer: slim orchestrator that ties config + data + model into a run."""

from __future__ import annotations

import logging
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from iter8ml.config import ExperimentConfig, HardwareProfile, TrackerType
from iter8ml.domain.hashing import dataframe_digest
from iter8ml.engine import trainer_factory
from iter8ml.engine.pipelines.executor import PipelineExecutor, PipelineMode
from iter8ml.engine.tracker import JSONLTracker, Tracker
from iter8ml.exceptions import TrainerStatePublishError

if TYPE_CHECKING:
    from iter8ml.workspace import Workspace

# Library versions recorded once per run in the experiment_started event so the
# flat run path also satisfies the "every run reproducible" contract.
_VERSION_PINNED_PACKAGES: tuple[str, ...] = (
    "polars",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
)


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _VERSION_PINNED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


_TRACKER_SETTINGS_KEYS: dict[str, frozenset[str]] = {
    "jsonl": frozenset({"max_file_size_mb", "backup_count"}),
    # wandb.init accepts many more kwargs; this allowlist covers the common
    # run-shaping ones. Extend deliberately, not speculatively.
    "wandb": frozenset(
        {"project", "entity", "mode", "tags", "name", "notes", "group", "job_type", "dir"}
    ),
    "mlflow": frozenset({"tracking_uri", "experiment_name"}),
}


def _validate_tracker_settings(settings: dict[str, Any], backend: str) -> None:
    """Fail loudly on tracker_settings keys the backend cannot accept."""
    unknown = sorted(set(settings) - _TRACKER_SETTINGS_KEYS[backend])
    if unknown:
        raise ValueError(
            f"Unknown tracker_settings keys for backend '{backend}': {unknown}. "
            f"Valid keys: {sorted(_TRACKER_SETTINGS_KEYS[backend])}"
        )


def _build_tracker(config: ExperimentConfig, workspace: Workspace) -> Tracker:
    """Build the tracker named by ``config.tracker`` (jsonl/wandb/mlflow).

    ``config.tracker_settings`` flows through as constructor kwargs after
    per-backend validation (unknown keys raise; nothing is silently dropped).
    """
    settings = dict(config.tracker_settings or {})
    if config.tracker == TrackerType.WANDB:
        _validate_tracker_settings(settings, "wandb")
        from iter8ml.engine.tracker import WandbTracker

        return WandbTracker(**settings)  # type: ignore[no-any-return]
    if config.tracker == TrackerType.MLFLOW:
        _validate_tracker_settings(settings, "mlflow")
        from iter8ml.engine.tracker import MLflowTracker

        return MLflowTracker(**settings)  # type: ignore[no-any-return]
    _validate_tracker_settings(settings, "jsonl")
    return JSONLTracker(log_path=str(workspace.experiments_path), **settings)


class Trainer:
    """Thin orchestrator that ties config + data + model into an experiment run."""

    def __init__(
        self,
        config: ExperimentConfig,
        workspace: Workspace,
        tracker: Tracker | None = None,
        resume_run_id: str | None = None,
    ):
        HardwareProfile.configure_omp_threads()
        self.config = config
        self.workspace = workspace
        self.resume_run_id = resume_run_id
        self._completed_models: set[str] = set()
        if resume_run_id:
            self._completed_models = _load_completed_models(
                workspace.experiments_path, resume_run_id
            )
            if self._completed_models:
                logging.getLogger(__name__).info(
                    "[resume] Skipping completed models: %s",
                    ", ".join(sorted(self._completed_models)),
                )
        _tracker: Tracker
        _tracker = tracker if tracker is not None else _build_tracker(config, workspace)
        self.tracker = _tracker
        self._event_adapter = trainer_factory.build_trainer_event_adapter(self.tracker)
        self._state_adapter = trainer_factory.build_trainer_state_adapter(
            workspace=self.workspace,
            llm_enabled=self.config.llm_enabled,
            llm_model=self.config.llm_model,
        )
        self.hardware = HardwareProfile.detect()

    def run(self, df: pl.DataFrame, split_frame: pl.DataFrame | None = None) -> dict:
        """Run full experiment on a Polars DataFrame via Hamilton DAG."""
        run_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"
        self.tracker.current_run_id = run_id

        # Honor config.data_sample (e.g. --quick's 20%) so quick runs train on
        # what they claim. Skipped when a medallion split_frame is supplied:
        # sampling there would break row_id alignment with the assigned folds.
        if self.config.data_sample < 1.0 and split_frame is None:
            n_before = len(df)
            df = df.sample(fraction=self.config.data_sample, seed=self.config.random_seed)
            logging.getLogger(__name__).info(
                "[data_sample] Using %.0f%% of data: %d -> %d rows",
                self.config.data_sample * 100,
                n_before,
                len(df),
            )

        self._publish_event(
            {
                "event": "experiment_started",
                "config": self.config.model_dump(mode="json"),
                "run_id": run_id,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "data_digest": dataframe_digest(df),
                "library_versions": _library_versions(),
            }
        )

        try:
            training_executor = PipelineExecutor(mode=PipelineMode.TRAINING, tracker=self.tracker)
            state = training_executor.run_training(
                config=self.config,
                df=df,
                run_id=run_id,
                vram_gb=self.hardware.vram_gb,
                completed_models=self._completed_models,
                workspace=self.workspace,
                split_frame=split_frame,
            )

            if state is not None:
                self._log_state_events(state, run_id)

            self._update_state()
            return state.results if state is not None else {}
        except KeyboardInterrupt:
            self._publish_event({"event": "experiment_cancelled", "run_id": run_id})
            raise
        except Exception as exc:
            self._publish_event(
                {
                    "event": "experiment_failed",
                    "run_id": run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        finally:
            self.tracker.finish()

    def _log_state_events(self, state: Any, run_id: str) -> None:
        for model_name, entry in state.results.items():
            if entry.get("is_baseline"):
                continue
            if "error" in entry:
                self._publish_event(
                    {
                        "event": "model_failed",
                        "run_id": run_id,
                        "model": model_name,
                        "error": entry["error"],
                    }
                )
                continue
            self._publish_event(
                {
                    "event": "model_completed",
                    "run_id": run_id,
                    "model": entry.get("model_name", model_name),
                    "task": self.config.task.value,
                    "params": entry.get("params", {}),
                    "cv_scores": entry.get("cv_scores", {}),
                    "cv_std": entry.get("cv_std", {}),
                    "calibration": entry.get("calibration"),
                    "duration_seconds": entry.get("duration_seconds", 0),
                    "artifact_path": entry.get("artifact_path", ""),
                }
            )

    def _update_state(self) -> None:
        try:
            self._state_adapter.publish()
        except Exception as e:
            run_id = self.tracker.current_run_id or "unknown"
            adapter_name = type(self._state_adapter).__name__
            raise TrainerStatePublishError(
                "Trainer state publication failed",
                context={
                    "run_id": run_id,
                    "adapter": adapter_name,
                    "original_type": type(e).__name__,
                    "original_message": str(e),
                },
            ) from e

    def _publish_event(self, event: dict[str, Any]) -> None:
        try:
            self._event_adapter.publish(event)
        except Exception:
            run_id = self.tracker.current_run_id or "unknown"
            adapter_name = type(self._event_adapter).__name__
            logging.getLogger(__name__).warning(
                "Trainer event publication failed (run_id=%s adapter=%s event=%s)",
                run_id,
                adapter_name,
                event.get("event", "unknown"),
            )


def _load_completed_models(log_path: Path, run_id: str) -> set[str]:
    from iter8ml.utils.io import iter_events

    completed: set[str] = set()
    # Torn trailing writes from a crashed run must not brick resume.
    for e in iter_events(log_path, on_error="skip_trailing"):
        if e.get("event") == "model_completed" and e.get("run_id") == run_id and "model" in e:
            completed.add(e["model"])
    return completed
