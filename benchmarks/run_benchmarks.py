"""Main benchmark runner: orchestrates all benchmark suites and produces reports.

Usage:
    uv run benchmarks/run_benchmarks.py [--quick] [--models ...] [--json PATH] [--csv PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_cv import run_all_cv_benchmarks
from benchmarks.benchmark_models import run_all_model_benchmarks
from benchmarks.benchmark_pipeline import run_all_pipeline_benchmarks
from benchmarks.benchmark_utils import (
    BenchResult,
    print_results,
    save_csv,
    save_json,
)

QUICK_SIZES = [(1_000, 10)]
DEFAULT_SIZES = [(1_000, 10), (5_000, 20)]
LARGE_SIZES = [(1_000, 10), (5_000, 20), (20_000, 50)]

DEFAULT_MODELS = [
    "catboost",
    "lightgbm",
    "xgboost",
    "naive_baseline",
    "linear_baseline",
    "ft_transformer",
    "tabnet",
    "tabpfn",
]


def run_all(
    sizes: list[tuple[int, int]] | None = None,
    models: list[str] | None = None,
    quick: bool = False,
) -> list[BenchResult]:
    if sizes is None:
        sizes = QUICK_SIZES if quick else DEFAULT_SIZES
    if models is None:
        models = DEFAULT_MODELS

    all_results: list[BenchResult] = []
    t_total = time.perf_counter()

    print("=" * 60)
    print("  tabular-blueprint Benchmark Suite")
    print("=" * 60)
    print(f"  Data sizes : {sizes}")
    print(f"  Models     : {models}")
    print(f"  Mode       : {'quick' if quick else 'standard'}")
    print()

    print("[1/3] Model training & prediction benchmarks...")
    t0 = time.perf_counter()
    model_results = run_all_model_benchmarks(sizes=sizes, models=models, task="classification")
    model_results += run_all_model_benchmarks(sizes=sizes, models=models, task="regression")
    all_results.extend(model_results)
    print(f"      Done: {len(model_results)} scenarios in {time.perf_counter() - t0:.1f}s")

    print("[2/3] Pipeline & data benchmarks...")
    t0 = time.perf_counter()
    pipeline_results = run_all_pipeline_benchmarks(sizes=sizes)
    all_results.extend(pipeline_results)
    print(f"      Done: {len(pipeline_results)} scenarios in {time.perf_counter() - t0:.1f}s")

    print("[3/3] Cross-validation & calibration benchmarks...")
    t0 = time.perf_counter()
    cv_results = run_all_cv_benchmarks(sizes=sizes, models=models)
    all_results.extend(cv_results)
    print(f"      Done: {len(cv_results)} scenarios in {time.perf_counter() - t0:.1f}s")

    total = time.perf_counter() - t_total
    print(f"\nTotal: {len(all_results)} benchmark scenarios in {total:.1f}s")
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tabular-blueprint benchmark suite")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with minimal sizes for fast feedback",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to benchmark",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Save results as CSV to this path",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=None,
        help="Custom sizes as n_samples n_features pairs",
    )
    args = parser.parse_args()

    sizes: list[tuple[int, int]] | None = None
    if args.sizes:
        if len(args.sizes) % 2 != 0:
            print("Error: --sizes must be pairs of n_samples n_features")
            return
        sizes = [(args.sizes[i], args.sizes[i + 1]) for i in range(0, len(args.sizes), 2)]

    results = run_all(sizes=sizes, models=args.models, quick=args.quick)

    print()
    print_results(results)

    if args.json:
        save_json(results, args.json)
    else:
        save_json(results, "benchmark_results.json")

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
