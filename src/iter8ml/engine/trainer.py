"""Trainer: slim orchestrator that ties config + data + model into a run."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from iter8ml.config import ExperimentConfig, HardwareProfile
from iter8ml.engine.pipelines.executor import PipelineExecutor, PipelineMode
from iter8ml.engine.tracker import JSONLTracker, Tracker

if TYPE_CHECKING:
    from iter8ml.workspace import Workspace


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
        if tracker is not None:
            _tracker = tracker
        else:
            _tracker = JSONLTracker(log_path=str(workspace.experiments_path))
        self.tracker = _tracker
        self.hardware = HardwareProfile.detect()

    def run(self, df: pl.DataFrame) -> dict:
        """Run full experiment on a Polars DataFrame via Hamilton DAG."""
        run_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"
        self.tracker.current_run_id = run_id

        self.tracker.log_event(
            {
                "event": "experiment_started",
                "config": self.config.model_dump(mode="json"),
                "run_id": run_id,
            }
        )

        training_executor = PipelineExecutor(mode=PipelineMode.TRAINING, tracker=self.tracker)
        state = training_executor.run_training(
            config=self.config,
            df=df,
            run_id=run_id,
            vram_gb=self.hardware.vram_gb,
            completed_models=self._completed_models,
            workspace=self.workspace,
        )

        if state is not None:
            self._log_state_events(state, run_id)

        self._update_state()
        self.tracker.finish()
        return state.results if state is not None else {}

    def _log_state_events(self, state: Any, run_id: str) -> None:
        for model_name, entry in state.results.items():
            if entry.get("is_baseline"):
                continue
            if "error" in entry:
                self.tracker.log_event(
                    {
                        "event": "model_failed",
                        "run_id": run_id,
                        "model": model_name,
                        "error": entry["error"],
                    }
                )
                continue
            self.tracker.log_event(
                {
                    "event": "model_completed",
                    "run_id": run_id,
                    "model": entry.get("model_name", model_name),
                    "task": self.config.task.value,
                    "params": entry.get("params", {}),
                    "cv_scores": entry.get("cv_scores", {}),
                    "duration_seconds": entry.get("duration_seconds", 0),
                    "artifact_path": entry.get("artifact_path", ""),
                }
            )

    def _update_state(self) -> None:
        from iter8ml.engine.state_observer import StateObserver

        observer = StateObserver(
            workspace=self.workspace,
            llm_enabled=self.config.llm_enabled,
            llm_model=self.config.llm_model,
        )
        observer.generate()


def _load_completed_models(log_path: Path, run_id: str) -> set[str]:
    from iter8ml.utils.io import iter_events

    completed: set[str] = set()
    for e in iter_events(log_path):
        if e.get("event") == "model_completed" and e.get("run_id") == run_id and "model" in e:
            completed.add(e["model"])
    return completed
