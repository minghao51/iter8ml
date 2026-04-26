"""Model training & prediction benchmarks for all registered models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_utils import (
    BENCH_RUNS,
    WARMUP_RUNS,
    BenchResult,
    bench_fn,
    make_numpy,
)
from tabular_blueprint.models.factory import get_model_class

CONVENTIONAL_MODELS = ["catboost", "lightgbm", "xgboost"]
BASELINE_MODELS = ["naive_baseline", "linear_baseline"]
DEEP_MODELS = ["ft_transformer", "tabnet", "tabpfn"]

PREDICT_SIZES = [1_000, 10_000]

DEEP_WARMUP = 1
DEEP_RUNS = 2


def _is_available(model_name: str) -> bool:
    try:
        get_model_class(model_name)
        return True
    except Exception:
        return False


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _make_model(model_name: str, task: str, n_features: int, y: np.ndarray) -> Any:
    from tabular_blueprint.models.factory import get_model_class as _gmc

    cls = _gmc(model_name)
    if model_name == "ft_transformer":
        n_classes = len(np.unique(y)) if task == "classification" else 1
        return cls(task=task, n_features=n_features, n_classes=n_classes)
    if model_name == "tabpfn":
        device = "cuda" if _has_cuda() else "cpu"
        return cls(task=task, device=device)
    return cls(task=task)


def _effective_runs(model_name: str, warmup: int, runs: int) -> tuple[int, int]:
    if model_name in DEEP_MODELS:
        return min(warmup, DEEP_WARMUP), min(runs, DEEP_RUNS)
    return warmup, runs


def bench_model_fit(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    *,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCH_RUNS,
) -> BenchResult:
    w, r = _effective_runs(model_name, warmup, runs)

    def fit_once() -> None:
        model = _make_model(model_name, task, X.shape[1], y)
        model.fit(X, y)

    return bench_fn(
        fit_once,
        name=f"fit/{model_name}",
        category="model_fit",
        params={"task": task, "n_samples": X.shape[0], "n_features": X.shape[1]},
        warmup=w,
        runs=r,
    )


def bench_model_predict(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task: str,
    *,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCH_RUNS,
) -> BenchResult:
    model = _make_model(model_name, task, X_train.shape[1], y_train)
    model.fit(X_train, y_train)
    w, r = _effective_runs(model_name, warmup, runs)

    def predict_once() -> None:
        model.predict(X_test)

    return bench_fn(
        predict_once,
        name=f"predict/{model_name}",
        category="model_predict",
        params={
            "task": task,
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "n_features": X_train.shape[1],
        },
        warmup=w,
        runs=r,
    )


def bench_model_predict_proba(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task: str,
    *,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCH_RUNS,
) -> BenchResult | None:
    if task != "classification":
        return None

    model = _make_model(model_name, task, X_train.shape[1], y_train)
    model.fit(X_train, y_train)

    if model.predict_proba(X_test) is None:
        return None

    w, r = _effective_runs(model_name, warmup, runs)

    def proba_once() -> None:
        model.predict_proba(X_test)

    return bench_fn(
        proba_once,
        name=f"predict_proba/{model_name}",
        category="model_predict_proba",
        params={
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "n_features": X_train.shape[1],
        },
        warmup=w,
        runs=r,
    )


def run_all_model_benchmarks(
    sizes: list[tuple[int, int]] | None = None,
    models: list[str] | None = None,
    task: str = "classification",
    warmup: int = WARMUP_RUNS,
    runs: int = BENCH_RUNS,
) -> list[BenchResult]:
    if sizes is None:
        sizes = [(5_000, 20)]
    if models is None:
        models = [m for m in CONVENTIONAL_MODELS + BASELINE_MODELS if _is_available(m)]

    results: list[BenchResult] = []

    for n_samples, n_features in sizes:
        X, y = make_numpy(n_samples, n_features, task=task)
        X_test, _ = make_numpy(1_000, n_features, task=task)

        for model_name in models:
            try:
                results.append(
                    bench_model_fit(
                        model_name, X, y, task, warmup=warmup, runs=runs,
                    )
                )
                results.append(
                    bench_model_predict(
                        model_name, X, y, X_test, task, warmup=warmup, runs=runs,
                    )
                )
                proba_result = bench_model_predict_proba(
                    model_name, X, y, X_test, task, warmup=warmup, runs=runs,
                )
                if proba_result is not None:
                    results.append(proba_result)
            except Exception as e:
                print(f"  SKIP {model_name} @ ({n_samples}, {n_features}): {e}")

    return results
