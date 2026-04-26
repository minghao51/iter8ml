"""Pipeline benchmarks: data preparation, feature engineering, adapters, preprocessing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_utils import (
    BenchResult,
    bench_fn,
    make_class_df,
    make_numpy,
)
from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.data.adapter import DataAdapter
from tabular_blueprint.data.feature_engine import (
    discover_interactions,
    extract_top_k_features,
    prune_features,
    transform_target,
)
from tabular_blueprint.engine.data_preparation import DataPreparationService
from tabular_blueprint.engine.tracker import JSONLTracker


def bench_data_adapter_numpy(sizes: list[tuple[int, int]] | None = None) -> list[BenchResult]:
    if sizes is None:
        sizes = [(5_000, 20), (20_000, 50)]
    results: list[BenchResult] = []
    adapter = DataAdapter(target_format="numpy")

    for n_samples, n_features in sizes:
        df = make_class_df(n_samples, n_features)

        def convert(d: pl.DataFrame = df, a: DataAdapter = adapter) -> None:
            a.transform(d, "target")

        results.append(bench_fn(
            convert,
            name="adapter/numpy",
            category="data_adapter",
            params={"n_samples": n_samples, "n_features": n_features},
        ))
    return results


def bench_data_adapter_tensor(sizes: list[tuple[int, int]] | None = None) -> list[BenchResult]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return []

    if sizes is None:
        sizes = [(5_000, 20)]
    results: list[BenchResult] = []
    adapter = DataAdapter(target_format="tensor")

    for n_samples, n_features in sizes:
        df = make_class_df(n_samples, n_features)

        def convert(d: pl.DataFrame = df, a: DataAdapter = adapter) -> None:
            a.transform(d, "target")

        results.append(bench_fn(
            convert,
            name="adapter/tensor",
            category="data_adapter",
            params={"n_samples": n_samples, "n_features": n_features},
        ))
    return results


def bench_data_hash(sizes: list[tuple[int, int]] | None = None) -> list[BenchResult]:
    from tabular_blueprint.data.loaders import get_data_hash

    if sizes is None:
        sizes = [(5_000, 20), (20_000, 50)]
    results: list[BenchResult] = []

    for n_samples, n_features in sizes:
        df = make_class_df(n_samples, n_features)

        def hash_df(d: pl.DataFrame = df) -> None:
            get_data_hash(d)

        results.append(bench_fn(
            hash_df,
            name="data_hash",
            category="data_io",
            params={"n_samples": n_samples, "n_features": n_features},
        ))
    return results


def bench_target_transform() -> list[BenchResult]:
    results: list[BenchResult] = []
    _X, y_reg = make_numpy(5_000, 20, task="regression")

    rng = np.random.RandomState(42)
    skewed_y = np.exp(rng.randn(5_000) * 2) + 1.0

    cases = [
        ("log1p", skewed_y.copy()),
        ("yeo-johnson", skewed_y.copy()),
        ("box-cox", skewed_y.copy()),
        ("auto", skewed_y.copy()),
        ("none", y_reg.copy()),
    ]

    for method, y_input in cases:
        def transform(m: str = method, y: np.ndarray = y_input) -> None:
            transform_target(y, method=m)

        results.append(bench_fn(
            transform,
            name=f"target_transform/{method}",
            category="feature_engineering",
            params={"method": method, "n_samples": len(y_input)},
        ))
    return results


def bench_extract_top_k() -> list[BenchResult]:
    from sklearn.linear_model import LogisticRegression

    results: list[BenchResult] = []
    X, y = make_numpy(5_000, 20, task="classification")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    for k in [5, 10, 20]:
        def extract(m: object = model, xx: np.ndarray = X, yy: np.ndarray = y, kk: int = k) -> None:
            extract_top_k_features(m, xx, yy, k=kk)

        results.append(bench_fn(
            extract,
            name=f"extract_top_k/k={k}",
            category="feature_engineering",
            params={"n_samples": X.shape[0], "n_features": X.shape[1], "top_k": k},
        ))
    return results


def bench_discover_interactions() -> list[BenchResult]:
    results: list[BenchResult] = []
    X, y = make_numpy(1_000, 10, task="classification")
    top_k = list(range(10))

    def discover(xx: np.ndarray = X, yy: np.ndarray = y, tk: list[int] = top_k) -> None:
        discover_interactions(xx, yy, top_k_indices=tk, lift_threshold=0.01)

    results.append(bench_fn(
        discover,
        name="discover_interactions",
        category="feature_engineering",
        params={"n_samples": X.shape[0], "n_features": X.shape[1], "top_k": len(top_k)},
        warmup=1,
        runs=3,
    ))
    return results


def bench_prune_features() -> list[BenchResult]:
    from sklearn.linear_model import LogisticRegression

    results: list[BenchResult] = []
    X, y = make_numpy(5_000, 30, task="classification")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    def prune(m: object = model, xx: np.ndarray = X, yy: np.ndarray = y) -> None:
        prune_features(m, xx, yy, min_importance=0.001, task="classification")

    results.append(bench_fn(
        prune,
        name="prune_features",
        category="feature_engineering",
        params={"n_samples": X.shape[0], "n_features": X.shape[1]},
    ))
    return results


def bench_data_preparation(sizes: list[tuple[int, int]] | None = None) -> list[BenchResult]:
    if sizes is None:
        sizes = [(1_000, 10)]
    results: list[BenchResult] = []

    for n_samples, n_features in sizes:
        df = make_class_df(n_samples, n_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            config = ExperimentConfig(
                name="bench",
                task="classification",
                target_col="target",
                data_path="",
                workspace_dir=ws,
                cv_folds=3,
            )
            tracker = JSONLTracker(str(ws / "experiments.jsonl"))
            service = DataPreparationService(config, tracker)

            def prepare(d: pl.DataFrame = df, s: DataPreparationService = service) -> None:
                s.prepare(d, "bench_run", run_leakage_audit=False)

            results.append(bench_fn(
                prepare,
                name="data_preparation/basic",
                category="pipeline",
                params={"n_samples": n_samples, "n_features": n_features},
                warmup=1,
                runs=3,
            ))

    return results


def run_all_pipeline_benchmarks(
    sizes: list[tuple[int, int]] | None = None,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    results.extend(bench_data_adapter_numpy(sizes))
    results.extend(bench_data_adapter_tensor(sizes))
    results.extend(bench_data_hash(sizes))
    results.extend(bench_target_transform())
    results.extend(bench_extract_top_k())
    results.extend(bench_discover_interactions())
    results.extend(bench_prune_features())
    results.extend(bench_data_preparation(sizes))
    return results
