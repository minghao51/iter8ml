"""Cross-validation, evaluation, and calibration benchmarks."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.engine.calibration import CalibratedModel
from tabular_blueprint.engine.evaluator import Evaluator
from tabular_blueprint.models.factory import get_model_class

from benchmarks.benchmark_utils import (
    BenchResult,
    bench_fn,
    make_numpy,
)


def bench_cv_evaluate(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv_folds: int = 5,
    *,
    warmup: int = 1,
    runs: int = 3,
) -> BenchResult:
    with tempfile.TemporaryDirectory():
        config = ExperimentConfig(
            name="bench",
            task=task,
            target_col="target",
            data_path="",
            cv_folds=cv_folds,
        )
        evaluator = Evaluator(config)
        model_cls = get_model_class(model_name)

        def evaluate() -> None:
            evaluator.evaluate(model_cls, X, y, task=task)

        return bench_fn(
            evaluate,
            name=f"cv_evaluate/{model_name}",
            category="cross_validation",
            params={
                "task": task,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "cv_folds": cv_folds,
            },
            warmup=warmup,
            runs=runs,
        )


def bench_cv_strategies(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    *,
    warmup: int = 1,
    runs: int = 3,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    model_cls = get_model_class("catboost")

    for strategy in ["kfold", "stratified"]:
        for folds in [3, 5]:
            with tempfile.TemporaryDirectory():
                config = ExperimentConfig(
                    name="bench",
                    task=task,
                    target_col="target",
                    data_path="",
                    cv_folds=folds,
                    cv_strategy=strategy,
                )
                evaluator = Evaluator(config)

                def ev(
                    e: Evaluator = evaluator,
                    mc: type = model_cls,
                    xx: np.ndarray = X,
                    yy: np.ndarray = y,
                    t: str = task,
                ) -> None:
                    e.evaluate(mc, xx, yy, task=t)

                results.append(
                    bench_fn(
                        ev,
                        name=f"cv_strategy/{strategy}/folds={folds}",
                        category="cross_validation",
                        params={"strategy": strategy, "cv_folds": folds, "n_samples": X.shape[0]},
                        warmup=warmup,
                        runs=runs,
                    )
                )
    return results


def bench_calibration(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    warmup: int = 1,
    runs: int = 3,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    if len(np.unique(y)) < 2:
        return results

    model_cls = get_model_class(model_name)

    for method in ["platt", "isotonic"]:
        model = model_cls(task="classification")

        def calibrate(
            m: object = model,
            xx: np.ndarray = X,
            yy: np.ndarray = y,
            mt: str = method,
        ) -> None:
            nonlocal model_cls
            base = model_cls(task="classification")
            calibrated = CalibratedModel(base, method=mt)
            calibrated.fit(xx, yy)

        results.append(
            bench_fn(
                calibrate,
                name=f"calibration/{model_name}/{method}",
                category="calibration",
                params={"model": model_name, "method": method, "n_samples": X.shape[0]},
                warmup=warmup,
                runs=runs,
            )
        )
    return results


def run_all_cv_benchmarks(
    sizes: list[tuple[int, int]] | None = None,
    models: list[str] | None = None,
    warmup: int = 1,
    runs: int = 3,
) -> list[BenchResult]:
    if sizes is None:
        sizes = [(1_000, 10)]
    if models is None:
        models = ["catboost", "lightgbm", "xgboost", "naive_baseline", "linear_baseline"]

    results: list[BenchResult] = []

    for n_samples, n_features in sizes:
        X, y = make_numpy(n_samples, n_features, task="classification")

        for model_name in models:
            try:
                results.append(
                    bench_cv_evaluate(
                        model_name,
                        X,
                        y,
                        "classification",
                        warmup=warmup,
                        runs=runs,
                    )
                )
            except Exception as e:
                print(f"  SKIP cv/{model_name} @ ({n_samples}, {n_features}): {e}")

        results.extend(bench_cv_strategies(X, y, "classification", warmup=warmup, runs=runs))

        for model_name in models:
            if model_name in ("naive_baseline",):
                continue
            try:
                results.extend(bench_calibration(model_name, X, y, warmup=warmup, runs=runs))
            except Exception as e:
                print(f"  SKIP cal/{model_name}: {e}")

    return results
