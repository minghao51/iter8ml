"""Primary programmatic API: ExperimentSession."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from iter8ml.config import ExperimentConfig
from iter8ml.domain.manifests import RunPlan
from iter8ml.engine.state_observer import StateObserver
from iter8ml.engine.tracker import Tracker
from iter8ml.engine.trainer import Trainer
from iter8ml.orchestration.service import ExecutionResult, MedallionExecutionService
from iter8ml.runtime.plan import compile_run_plan
from iter8ml.services.export import ExportService
from iter8ml.services.registry import RegistryService
from iter8ml.services.reporting import ReportService
from iter8ml.workspace import Workspace

__all__ = ["ExperimentSession"]


class ExperimentSession:
    """Primary high-level interface for iter8ml experiments."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        tracker: Tracker | None = None,
    ):
        self.workspace = workspace or Workspace()
        self.workspace.init()
        self.tracker = tracker

    def run(
        self,
        config: ExperimentConfig,
        df: pl.DataFrame,
        *,
        resume_run_id: str | None = None,
    ) -> dict:
        trainer = Trainer(
            config=config,
            workspace=self.workspace,
            tracker=self.tracker,
            resume_run_id=resume_run_id,
        )
        return trainer.run(df)

    def plan(self, config: ExperimentConfig, *, materialization: str = "reproducible") -> RunPlan:
        """Compile a deterministic medallion plan without executing it."""
        return compile_run_plan(config, materialization=materialization)

    def medallion_run(
        self,
        config: ExperimentConfig,
        df: pl.DataFrame,
        *,
        execute_training: bool = True,
    ) -> ExecutionResult:
        """Run the explicit local Bronze-to-Platinum lifecycle."""
        return MedallionExecutionService(self.workspace).run(
            config, df, execute_training=execute_training
        )

    def drift_check(
        self,
        reference_df: pl.DataFrame,
        live_df: pl.DataFrame,
        method: str = "psi",
    ) -> dict:
        from iter8ml.engine.pipelines.executor import PipelineExecutor, PipelineMode

        executor = PipelineExecutor(mode=PipelineMode.DRIFT)
        return executor.run_drift(reference_df, live_df, drift_method=method)

    def leaderboard(self, metric: str | None = None, limit: int | None = None) -> pl.DataFrame:
        report = ReportService(workspace=self.workspace).build_report(metric=metric, limit=limit)
        rows: list[dict[str, Any]] = []
        for entry in report.leaderboard:
            rows.append(
                {
                    "model": entry.model,
                    "run_id": entry.run_id,
                    "primary_metric": entry.primary_metric,
                    "primary_score": entry.primary_score,
                    "duration_seconds": entry.duration_seconds,
                    "timestamp": entry.timestamp,
                    "task": entry.task,
                }
            )
        return pl.DataFrame(rows)

    def export(self, key: str, output_dir: str | Path | None = None) -> Path:
        service = ExportService(workspace=self.workspace)
        return service.export(key, output_dir=output_dir)

    def promote(self, run_id: str, key: str) -> Any:
        service = RegistryService(workspace=self.workspace)
        return service.promote_run(run_id, key, self.workspace.experiments_path)

    def state(self, llm_enabled: bool = False) -> str:
        observer = StateObserver(workspace=self.workspace, llm_enabled=llm_enabled)
        return observer.generate()
