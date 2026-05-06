"""Trainer: slim orchestrator that ties config + data + model into a run."""

import time
import uuid
from pathlib import Path

import polars as pl

from tabular_blueprint.config import ExperimentConfig, HardwareProfile
from tabular_blueprint.engine.tracker import JSONLTracker, Tracker
from tabular_blueprint.pipelines.executor import PipelineExecutor, PipelineMode


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        tracker: Tracker | None = None,
        run_leakage_audit: bool = True,
        resume_run_id: str | None = None,
    ):
        HardwareProfile.configure_omp_threads()
        self.config = config
        self.resume_run_id = resume_run_id
        self._completed_models: set[str] = set()
        if resume_run_id:
            self._completed_models = _load_completed_models(
                config.workspace_dir / "experiments.jsonl", resume_run_id
            )
            if self._completed_models:
                import typer

                typer.echo(
                    f"[resume] Skipping completed models: "
                    f"{', '.join(sorted(self._completed_models))}"
                )
        _tracker: Tracker
        if tracker is not None:
            _tracker = tracker
        else:
            _tracker = JSONLTracker(log_path=str(self.config.workspace_dir / "experiments.jsonl"))
        self.tracker = _tracker
        self.hardware = HardwareProfile.detect()
        self.run_leakage_audit = run_leakage_audit

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
            run_leakage_audit=self.run_leakage_audit,
            completed_models=self._completed_models,
        )

        if state is not None:
            self._log_state_events(state, run_id)

        self._update_state()
        self.tracker.finish()
        return state.results if state is not None else {}

    def _log_state_events(self, state: object, run_id: str) -> None:
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
        from tabular_blueprint.engine.state_observer import StateObserver

        observer = StateObserver(
            log_path=str(self.config.workspace_dir / "experiments.jsonl"),
            registry_path=str(self.config.workspace_dir / "registry.json"),
            output_path=str(self.config.workspace_dir / "current_state.md"),
            leaderboard_path=str(self.config.workspace_dir / "leaderboard.md"),
            llm_enabled=self.config.llm_enabled,
            llm_model=self.config.llm_model,
        )
        observer.generate()


def _load_completed_models(log_path: Path, run_id: str) -> set[str]:
    from tabular_blueprint.utils.jsonl import load_events

    events = load_events(log_path)
    return {
        e["model"]
        for e in events
        if e.get("event") == "model_completed" and e.get("run_id") == run_id and "model" in e
    }
