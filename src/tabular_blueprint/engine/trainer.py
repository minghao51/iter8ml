"""Trainer: slim orchestrator that ties config + data + model into a run."""

import time
import uuid
from pathlib import Path
from typing import Any

import polars as pl

from tabular_blueprint.config import ExperimentConfig, HardwareProfile
from tabular_blueprint.data.loaders import get_data_hash
from tabular_blueprint.engine.data_preparation import DataPreparationService
from tabular_blueprint.engine.drift_checker import DriftChecker
from tabular_blueprint.engine.embedding_trainer import EmbeddingEngine
from tabular_blueprint.engine.explainability_service import ExplainabilityService
from tabular_blueprint.engine.feature_engineer import FeatureEngineer
from tabular_blueprint.engine.model_trainer import ModelTrainer
from tabular_blueprint.engine.tracker import JSONLTracker, Tracker
from tabular_blueprint.models.baselines import LinearBaseline, NaiveBaseline
from tabular_blueprint.models.selector import ModelSelector
from tabular_blueprint.pipelines.executor import PipelineExecutor, PipelineMode

BASELINE_MODELS = {
    "naive_baseline": NaiveBaseline,
    "linear_baseline": LinearBaseline,
}


def _load_completed_models(log_path: Path, run_id: str) -> set[str]:
    """Load model names that already completed in a given run."""
    from tabular_blueprint.utils.jsonl import load_events

    events = load_events(log_path)
    return {
        e["model"]
        for e in events
        if e.get("event") == "model_completed" and e.get("run_id") == run_id and "model" in e
    }


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        tracker: Tracker | None = None,
        run_baselines: bool = True,
        run_leakage_audit: bool = True,
        use_cache: bool = True,
        resume_run_id: str | None = None,
    ):
        HardwareProfile.configure_omp_threads()
        self.config = config
        self.use_cache = use_cache
        self.resume_run_id = resume_run_id
        self._completed_models: set[str] = set()
        self._hamilton_warning_emitted = False
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
        self.selector = ModelSelector()
        self.run_baselines = run_baselines
        self.run_leakage_audit = run_leakage_audit
        self.executor = PipelineExecutor()

        self._data_prep = DataPreparationService(config, _tracker)
        self._feature_eng = FeatureEngineer(config, _tracker)
        self._embedding_eng = EmbeddingEngine(config, _tracker)
        self._model_trainer = ModelTrainer(config, _tracker, self.hardware)
        self._drift = DriftChecker(config, _tracker)
        self._explainer = ExplainabilityService(config, _tracker)

    def run(self, df: pl.DataFrame) -> dict:
        """Run full experiment on a Polars DataFrame."""
        run_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"
        self.tracker.current_run_id = run_id

        hamilton_result = self._try_hamilton_training(df, run_id)
        if hamilton_result is not None:
            self._update_state()
            self.tracker.finish()
            return hamilton_result

        return self._run_imperative(df, run_id)

    def _try_hamilton_training(self, df: pl.DataFrame, run_id: str) -> dict | None:
        training_executor = PipelineExecutor(mode=PipelineMode.TRAINING)
        if not training_executor.available:
            return None

        try:
            state = training_executor.run_training(
                df=df,
                target_col=self.config.target_col,
                task=self.config.task.value,
                config_models=self.config.models,
                experiment_name=self.config.name,
                run_id=run_id,
                workspace_dir=str(self.config.workspace_dir),
                vram_gb=self.hardware.vram_gb,
                cv_folds=self.config.cv_folds,
                cv_strategy=self.config.cv_strategy.value,
                metrics=self.config.metrics,
                calibration=self.config.calibration,
                afe_enabled=self.config.afe_enabled,
                afe_top_k=self.config.afe_top_k,
                afe_lift_threshold=self.config.afe_lift_threshold,
                afe_pruning=self.config.afe_pruning,
                afe_prune_min_importance=self.config.afe_prune_min_importance,
                random_seed=self.config.random_seed,
                run_quality_audit=self.config.run_quality_audit,
                auto_clean_noise=self.config.auto_clean_noise,
                noise_quality_threshold=self.config.noise_quality_threshold,
                run_leakage_audit=self.run_leakage_audit,
                target_transform=self.config.target_transform,
                target_skewness_threshold=self.config.target_skewness_threshold,
                embedding_enabled=self.config.embedding_enabled,
                embedding_method=self.config.embedding_method.value,
                embedding_dim=self.config.embedding_dim,
                embedding_max_categories=self.config.embedding_max_categories,
                embedding_epochs=self.config.embedding_epochs,
                embedding_lr=self.config.embedding_lr,
                embedding_mlp_width=self.config.embedding_mlp_width,
                embedding_mlp_depth=self.config.embedding_mlp_depth,
                embedding_ae_latent_dim=self.config.embedding_ae_latent_dim,
                embedding_ae_dropout=self.config.embedding_ae_dropout,
            )
            if state is not None:
                self._log_hamilton_state_events(state, run_id)
                return state.results
        except Exception as e:
            self.tracker.log_event(
                {
                    "event": "hamilton_fallback",
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            if not self._hamilton_warning_emitted:
                import typer

                typer.echo(
                    "[warning] Hamilton training path failed; falling back to imperative execution."
                )
                self._hamilton_warning_emitted = True
        return None

    def _run_imperative(self, df: pl.DataFrame, run_id: str) -> dict:
        if self.executor.available:
            df = self.executor.run_preprocessing(df)

        if self.config.data_sample < 1.0:
            fraction = max(0.01, min(1.0, self.config.data_sample))
            df = df.sample(fraction=fraction, seed=self.config.random_seed, shuffle=True)

        data_hash = get_data_hash(df)
        n_rows = len(df)
        n_features = len(df.columns) - 1

        cached = None
        if self.use_cache:
            from tabular_blueprint.data.cache import PreprocessingCache

            cache = PreprocessingCache(self.config.workspace_dir)
            cached = cache.load(data_hash, self.config)
            if cached is not None:
                import typer

                typer.echo(f"[cache] Loaded preprocessed data (hit for {data_hash})")
                X, y, feature_names = cached

        if cached is None:
            prep_result = self._data_prep.prepare(df, run_id, self.run_leakage_audit)
            X, y = prep_result.X, prep_result.y
            feature_names = prep_result.feature_names

            if self.use_cache:
                from tabular_blueprint.data.cache import PreprocessingCache

                cache = PreprocessingCache(self.config.workspace_dir)
                cache.save(data_hash, self.config, X, y, feature_names)

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

        if self.config.embedding_enabled:
            X, feature_names = self._embedding_eng.fit_transform(
                df=df,
                X=X,
                y=y,
                feature_names=feature_names,
                target_col=self.config.target_col,
                run_id=run_id,
                data_hash=data_hash,
            )

        evaluator = self._build_evaluator()

        baseline_scores: dict[str, dict[str, float]] = {}
        if self.run_baselines:
            baseline_scores = self._model_trainer.run_baselines(
                X,
                y,
                evaluator,
                run_id,
                data_hash,
                n_rows,
                n_features,
                BASELINE_MODELS,
            )

        if self.config.afe_enabled:
            X, feature_names = self._feature_eng.run_afe(
                X,
                y,
                run_id,
                data_hash,
                models_to_run,
                feature_names,
            )

        max_workers = self.config.max_workers
        if self.hardware.has_gpu and self.hardware.vram_gb < 16:
            max_workers = 1

        models_to_run = [m for m in models_to_run if m not in self._completed_models]

        results = self._model_trainer.train_all(
            models_to_run,
            X,
            y,
            evaluator,
            run_id,
            data_hash,
            n_rows,
            n_features,
            max_workers,
            baseline_scores=baseline_scores,
            feature_names=feature_names,
        )

        self._update_state()
        self.tracker.finish()

        return results

    def _build_evaluator(self) -> Any:
        from tabular_blueprint.engine.evaluator import Evaluator as Ev

        return Ev(self.config)

    def _log_hamilton_state_events(self, state: object, run_id: str) -> None:
        self.tracker.log_event(
            {
                "event": "experiment_started",
                "config": self.config.model_dump(mode="json"),
                "run_id": run_id,
            }
        )
        for model_name, entry in state.results.items():
            if entry.get("is_baseline"):
                continue
            if "error" in entry:
                self.tracker.log_event(
                    {"event": "model_failed", "model": model_name, "error": entry["error"]}
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
