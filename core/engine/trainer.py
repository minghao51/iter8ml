"""Trainer: ties config + data + model into a run."""

import fcntl
import json
import os
import platform
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from configs.experiment import ExperimentConfig
from configs.hardware import HardwareProfile

if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
from core.data.adapter import DataAdapter
from core.data.loaders import get_data_hash
from core.engine.evaluator import Evaluator
from core.engine.tracker import JSONLTracker, Tracker
from core.models.selector import ModelSelector
from core.utils.jsonl import load_events

_MODEL_REGISTRY = {
    "catboost": ("core.models.conventional.catboost_model", "CatBoostModel"),
    "lightgbm": ("core.models.conventional.lightgbm_model", "LightGBMModel"),
    "xgboost": ("core.models.conventional.xgboost_model", "XGBoostModel"),
    "tabpfn": ("core.models.tabular_foundation.tabpfn_model", "TabPFNModel"),
    "ft_transformer": ("core.models.deep.ft_transformer", "FTTransformerModel"),
}
_MODEL_CLASS_CACHE: dict[str, type] = {}


def _get_model_class(model_name: str):
    """Factory to get model class by name."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in _MODEL_CLASS_CACHE:
        return _MODEL_CLASS_CACHE[model_name]

    import importlib

    module_path, class_name = _MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _MODEL_CLASS_CACHE[model_name] = cls
    return cls


def _update_registry(
    registry_path: str,
    key: str,
    model_name: str,
    run_id: str,
    score: float,
    artifact_path: str,
):
    """Update model registry if new model beats champion."""
    registry = {}
    registry_file = Path(registry_path)
    lock_path = str(registry_file.with_suffix(".lock"))

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            try:
                with open(registry_path) as f:
                    registry = json.load(f)
            except FileNotFoundError:
                registry = {}

            if key not in registry or score > registry[key].get("score", -float("inf")):
                registry[key] = {
                    "model": model_name,
                    "run_id": run_id,
                    "score": score,
                    "artifact_path": artifact_path,
                    "registered_at": datetime.now(UTC).isoformat(),
                }
                with open(registry_path, "w") as f:
                    json.dump(registry, f, indent=2)
                return True
            return False
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _generate_leaderboard(log_path: str, output_path: str):
    """Generate leaderboard.md from JSONL events."""
    events = load_events(log_path)

    completed = [e for e in events if e.get("event") == "model_completed"]
    completed.sort(
        key=lambda x: x.get("cv_scores", {}).get("roc_auc", x.get("cv_scores", {}).get("r2", 0)),
        reverse=True,
    )

    lines = [
        "# Experiment Leaderboard\n",
        "| Model | Run ID | CV Scores | Duration | Timestamp |",
        "|---|---|---|---|---|",
    ]
    for e in completed:
        scores = ", ".join(f"{k}={v:.4f}" for k, v in e.get("cv_scores", {}).items())
        lines.append(
            f"| {e.get('model', '?')} | {e.get('run_id', '?')} | {scores} "
            f"| {e.get('duration_seconds', '?')}s | {e.get('timestamp', '?')} |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


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

        _generate_leaderboard(
            str(self.config.workspace_dir / "experiments.jsonl"),
            str(self.config.workspace_dir / "leaderboard.md"),
        )
        self._update_state()
        self.tracker.finish()

        return results

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
                    _update_registry(
                        str(self.config.workspace_dir / "registry.json"),
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
                            _update_registry(
                                str(self.config.workspace_dir / "registry.json"),
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
            model_cls = _get_model_class(model_name)

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
        )
        observer.generate()
