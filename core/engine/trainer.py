"""Trainer: ties config + data + model into a run."""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from configs.experiment import ExperimentConfig
from configs.hardware import HardwareProfile

# Configure OpenMP threads based on hardware profile
HardwareProfile.configure_omp_threads()

from core.data.adapter import DataAdapter  # noqa: E402
from core.data.loaders import get_data_hash  # noqa: E402
from core.engine.evaluator import Evaluator  # noqa: E402
from core.engine.tracker import JSONLTracker, Tracker  # noqa: E402
from core.models.factory import get_model_class  # noqa: E402
from core.models.selector import ModelSelector  # noqa: E402
from core.services.registry_service import RegistryService  # noqa: E402

WORKSPACE_DIR = Path("workspace")


class Trainer:
    """Orchestrates experiment runs."""

    def __init__(
        self,
        config: ExperimentConfig,
        tracker: Tracker | None = None,
    ):
        self.config = config
        self.tracker = tracker or JSONLTracker(
            log_path=str(self.config.workspace_dir / "experiments.jsonl")
        )
        self.hardware = HardwareProfile.detect()
        self.selector = ModelSelector()

    def run(self, df: pl.DataFrame) -> dict:
        """Run full experiment on a Polars DataFrame."""
        run_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:6]}"
        self.tracker.current_run_id = run_id

        data_hash = get_data_hash(df)
        n_rows = len(df)
        n_features = len(df.columns) - 1

        self.tracker.log_event(
            {
                "event": "experiment_started",
                "config": self.config.model_dump(mode="json"),
                "data_hash": data_hash,
                "n_rows": n_rows,
                "n_features": n_features,
            }
        )

        models_to_run = (
            self.selector.select(
                n_rows=n_rows,
                task=self.config.task.value,
                vram_gb=self.hardware.vram_gb,
            )
            if self.config.models == "auto"
            else self.config.models
        )

        if self.config.target_col not in df.columns:
            raise ValueError(
                f"target_col '{self.config.target_col}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        adapter = DataAdapter(target_format="numpy")
        X, y = adapter.transform(df, self.config.target_col)

        evaluator = Evaluator(self.config)

        # Determine max workers based on GPU availability
        max_workers = self.config.max_workers
        if self.hardware.has_gpu and self.hardware.vram_gb < 16:
            # Single GPU with limited VRAM - run models sequentially
            max_workers = 1

        # Use sequential training if max_workers is 1
        if max_workers == 1:
            results = self._train_sequential(
                models_to_run, X, y, evaluator, run_id, data_hash, n_rows, n_features
            )
        else:
            results = self._train_concurrent(
                models_to_run, X, y, evaluator, run_id, data_hash, n_rows, n_features, max_workers
            )

        self._update_state()
        self.tracker.finish()

        return results

    def _update_champion_if_better(
        self, key: str, model_name: str, run_id: str, score: float, artifact_path: str
    ) -> bool:
        """Update registry if new model beats champion."""
        registry = RegistryService(str(self.config.workspace_dir / "registry.json"))
        return registry.update_if_better(key, model_name, run_id, score, artifact_path)

    def _train_sequential(
        self, models_to_run, X, y, evaluator, run_id, data_hash, n_rows, n_features
    ):
        """Train models sequentially."""
        results = {}
        best_score = -float("inf")
        primary_metric = self.config.metrics[0]

        for model_name in models_to_run:
            result = self._train_single_model(
                model_name, X, y, evaluator, run_id, data_hash, n_rows, n_features
            )
            results[model_name] = result

            if "error" not in result:
                score = result.get("cv_scores", {}).get(primary_metric, 0)
                if score > best_score:
                    best_score = score
                    self._update_champion_if_better(
                        f"{self.config.name}:{self.config.task.value}",
                        result["model_name"],
                        run_id,
                        score,
                        result["artifact_path"],
                    )

        return results

    def _train_concurrent(
        self, models_to_run, X, y, evaluator, run_id, data_hash, n_rows, n_features, max_workers
    ):
        """Train models concurrently using ThreadPoolExecutor."""
        results = {}
        best_score = -float("inf")
        primary_metric = self.config.metrics[0]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training jobs
            futures = {
                executor.submit(
                    self._train_single_model,
                    model_name,
                    X,
                    y,
                    evaluator,
                    run_id,
                    data_hash,
                    n_rows,
                    n_features,
                ): model_name
                for model_name in models_to_run
            }

            # Collect results as they complete
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result

                    if "error" not in result:
                        score = result.get("cv_scores", {}).get(primary_metric, 0)
                        if score > best_score:
                            best_score = score
                            self._update_champion_if_better(
                                f"{self.config.name}:{self.config.task.value}",
                                result["model_name"],
                                run_id,
                                score,
                                result["artifact_path"],
                            )
                except Exception as e:
                    results[model_name] = {"error": str(e)}

        return results

    def _train_single_model(
        self, model_name, X, y, evaluator, run_id, data_hash, n_rows, n_features
    ):
        """Train a single model and return results."""
        start = time.time()
        try:
            model_cls = get_model_class(model_name)

            cv_scores = evaluator.evaluate(model_cls, X, y, task=self.config.task.value)

            # Train on full dataset only after CV succeeds
            # FTTransformer requires n_features and n_classes
            if model_name == "ft_transformer":
                n_classes = len(np.unique(y)) if self.config.task.value == "classification" else 1
                model = model_cls(
                    task=self.config.task.value,
                    n_features=n_features,
                    n_classes=n_classes,
                )
            else:
                model = model_cls(task=self.config.task.value)
            model.fit(X, y)
            artifact_path = str(self.config.workspace_dir / "artifacts" / f"{model_name}_{run_id}")
            model.save(artifact_path)

            duration = time.time() - start

            event = {
                "event": "model_completed",
                "run_id": run_id,
                "model": model.model_name,
                "task": self.config.task.value,
                "data_hash": data_hash,
                "n_rows": n_rows,
                "n_features": n_features,
                "cv_scores": cv_scores,
                "duration_seconds": round(duration, 2),
                "artifact_path": artifact_path,
                "hardware": {
                    "device": "cuda" if self.hardware.has_gpu else "cpu",
                    "vram_used_gb": 0.0,
                },
            }

            self.tracker.log_event(event)
            self.tracker.log_metrics(cv_scores)

            return {
                "model_name": model.model_name,
                "cv_scores": cv_scores,
                "artifact_path": artifact_path,
                "duration_seconds": round(duration, 2),
            }

        except Exception as e:
            self.tracker.log_event(
                {
                    "event": "model_failed",
                    "model": model_name,
                    "error": str(e),
                }
            )
            return {"error": str(e)}

    def _update_state(self):
        """Generate LLM-readable state summary."""
        from core.engine.state_observer import StateObserver

        observer = StateObserver(
            log_path=str(self.config.workspace_dir / "experiments.jsonl"),
            registry_path=str(self.config.workspace_dir / "registry.json"),
            output_path=str(self.config.workspace_dir / "current_state.md"),
            leaderboard_path=str(self.config.workspace_dir / "leaderboard.md"),
        )
        observer.generate()
