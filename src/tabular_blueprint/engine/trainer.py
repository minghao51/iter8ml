"""Trainer: slim orchestrator that ties config + data + model into a run."""

import time
import uuid
from typing import Any

import polars as pl

from tabular_blueprint.config import ExperimentConfig, HardwareProfile
from tabular_blueprint.data.loaders import get_data_hash
from tabular_blueprint.engine.data_preparation import DataPreparationService
from tabular_blueprint.engine.drift_checker import DriftChecker
from tabular_blueprint.engine.explainability_service import ExplainabilityService
from tabular_blueprint.engine.feature_engineer import FeatureEngineer
from tabular_blueprint.engine.model_trainer import ModelTrainer
from tabular_blueprint.engine.tracker import JSONLTracker, Tracker
from tabular_blueprint.models.baselines import LinearBaseline, NaiveBaseline
from tabular_blueprint.models.selector import ModelSelector
from tabular_blueprint.pipelines.hamilton_executor import HamiltonExecutor

BASELINE_MODELS = {
    "naive_baseline": NaiveBaseline,
    "linear_baseline": LinearBaseline,
}


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        tracker: Tracker | None = None,
        run_baselines: bool = True,
        run_leakage_audit: bool = True,
    ):
        HardwareProfile.configure_omp_threads()
        self.config = config
        _tracker: Tracker
        if tracker is not None:
            _tracker = tracker
        else:
            _tracker = JSONLTracker(
                log_path=str(self.config.workspace_dir / "experiments.jsonl")
            )
        self.tracker = _tracker
        self.hardware = HardwareProfile.detect()
        self.selector = ModelSelector()
        self.run_baselines = run_baselines
        self.run_leakage_audit = run_leakage_audit
        self.executor = HamiltonExecutor()

        self._data_prep = DataPreparationService(config, _tracker)
        self._feature_eng = FeatureEngineer(config, _tracker)
        self._model_trainer = ModelTrainer(config, _tracker, self.hardware)
        self._drift = DriftChecker(config, _tracker)
        self._explainer = ExplainabilityService(config, _tracker)

    def run(self, df: pl.DataFrame) -> dict:
        """Run full experiment on a Polars DataFrame."""
        run_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"
        self.tracker.current_run_id = run_id

        df = self.executor.run(df)

        data_hash = get_data_hash(df)
        n_rows = len(df)
        n_features = len(df.columns) - 1

        models_to_run = (
            self.selector.select(
                n_rows=n_rows,
                task=self.config.task.value,
                vram_gb=self.hardware.vram_gb,
            )
            if self.config.models == "auto"
            else self.config.models
        )

        if "tabpfn" in models_to_run and n_rows > self.selector.TABPFN_ROW_LIMIT:
            from rich.console import Console

            Console().print(
                f"[bold yellow]Warning:[/bold yellow] TabPFN is recommended for datasets < "
                f"{self.selector.TABPFN_ROW_LIMIT} rows. Current dataset has {n_rows} rows. "
                "Performance may degrade or run out of memory."
            )

        self.tracker.log_event(
            {
                "event": "experiment_started",
                "config": self.config.model_dump(mode="json"),
                "data_hash": data_hash,
                "n_rows": n_rows,
                "n_features": n_features,
                "pipeline_lineage": self.executor.get_mermaid_graph(),
            }
        )

        prep_result = self._data_prep.prepare(df, run_id, self.run_leakage_audit)
        X, y = prep_result.X, prep_result.y
        feature_names = prep_result.feature_names

        evaluator = self._build_evaluator()

        baseline_scores: dict[str, dict[str, float]] = {}
        if self.run_baselines:
            baseline_scores = self._model_trainer.run_baselines(
                X, y, evaluator, run_id, data_hash, prep_result.n_rows, prep_result.n_features,
                BASELINE_MODELS,
            )

        if self.config.afe_enabled:
            X, feature_names = self._feature_eng.run_afe(
                X, y, run_id, data_hash, models_to_run, feature_names,
            )

        max_workers = self.config.max_workers
        if self.hardware.has_gpu and self.hardware.vram_gb < 16:
            max_workers = 1

        results = self._model_trainer.train_all(
            models_to_run, X, y, evaluator, run_id, data_hash,
            prep_result.n_rows, prep_result.n_features, max_workers,
            baseline_scores=baseline_scores,
            feature_names=feature_names,
        )

        if self.config.drift_detection != "none":
            self._drift.check(df, run_id)

        self._update_state()
        self.tracker.finish()

        return results

    def _build_evaluator(self) -> Any:
        from tabular_blueprint.engine.evaluator import Evaluator as Ev

        return Ev(self.config)

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
