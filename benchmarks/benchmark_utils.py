"""Shared utilities for benchmarking: timing, statistics, data generation, reporting."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.datasets import make_classification, make_regression

WARMUP_RUNS = 2
BENCH_RUNS = 5
DEFAULT_SEED = 42

DATA_SIZES: list[tuple[int, int]] = [
    (500, 10),
    (5_000, 20),
    (20_000, 50),
]


@dataclass
class BenchResult:
    name: str
    category: str
    params: dict[str, Any] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)
    memory_peak_mb: float | None = None

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) >= 2 else 0.0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "params": self.params,
            "n_runs": len(self.times),
            "mean_s": round(self.mean, 6),
            "median_s": round(self.median, 6),
            "stdev_s": round(self.stdev, 6),
            "min_s": round(self.min, 6),
            "max_s": round(self.max, 6),
        }
        if self.memory_peak_mb is not None:
            d["memory_peak_mb"] = round(self.memory_peak_mb, 2)
        return d


def _get_process_memory_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def bench_fn(
    fn: Callable[..., Any],
    *,
    name: str,
    category: str,
    params: dict[str, Any] | None = None,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCH_RUNS,
    track_memory: bool = False,
) -> BenchResult:
    for _ in range(warmup):
        fn()

    times: list[float] = []
    mem_before = _get_process_memory_mb() if track_memory else 0.0
    mem_peak = mem_before

    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if track_memory:
            mem_after = _get_process_memory_mb()
            mem_peak = max(mem_peak, mem_after)
            del mem_after

    result = BenchResult(
        name=name,
        category=category,
        params=params or {},
        times=times,
    )
    if track_memory:
        result.memory_peak_mb = max(0.0, mem_peak - mem_before)

    return result


def make_class_df(n_samples: int, n_features: int, seed: int = DEFAULT_SEED) -> pl.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(5, n_features // 2),
        random_state=seed,
    )
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(n_features)})
    return df.with_columns(target=pl.Series(y))


def make_reg_df(n_samples: int, n_features: int, seed: int = DEFAULT_SEED) -> pl.DataFrame:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(5, n_features // 2),
        random_state=seed,
    )
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(n_features)})
    return df.with_columns(target=pl.Series(y))


def make_numpy(
    n_samples: int, n_features: int, task: str = "classification", seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if task == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(5, n_features // 2),
            random_state=seed,
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(5, n_features // 2),
            random_state=seed,
        )
    return X, y


def print_results(results: list[BenchResult]) -> None:
    rich_available = False
    try:
        import importlib

        importlib.util.find_spec("rich")
        rich_available = True
    except (ImportError, ModuleNotFoundError):
        pass

    if rich_available:
        _print_rich(results)
    else:
        _print_plain(results)


def _print_rich(results: list[BenchResult]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    categories: dict[str, list[BenchResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat, cat_results in categories.items():
        table = Table(title=f"Benchmark: {cat}", show_lines=True)
        table.add_column("Scenario", style="cyan", max_width=50)
        table.add_column("Params", style="dim", max_width=40)
        table.add_column("Runs", justify="right")
        table.add_column("Mean (s)", justify="right", style="green")
        table.add_column("Median (s)", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("Min (s)", justify="right")
        table.add_column("Max (s)", justify="right")
        table.add_column("Peak Mem (MB)", justify="right", style="yellow")

        for r in cat_results:
            param_str = ", ".join(f"{k}={v}" for k, v in r.params.items()) if r.params else "-"
            mem_str = f"{r.memory_peak_mb:.1f}" if r.memory_peak_mb is not None else "-"
            table.add_row(
                r.name,
                param_str,
                str(len(r.times)),
                f"{r.mean:.4f}",
                f"{r.median:.4f}",
                f"{r.stdev:.4f}",
                f"{r.min:.4f}",
                f"{r.max:.4f}",
                mem_str,
            )

        console.print(table)
        console.print()


def _print_plain(results: list[BenchResult]) -> None:
    categories: dict[str, list[BenchResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat, cat_results in categories.items():
        print(f"\n{'=' * 80}")
        print(f"  Benchmark: {cat}")
        print(f"{'=' * 80}")
        fmt = "{:<45} {:<25} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}"
        print(fmt.format(
            "Scenario", "Params", "Runs", "Mean(s)", "Median(s)", "StdDev",
            "Min(s)", "Max(s)", "Mem(MB)",
        ))
        print("-" * 136)
        for r in cat_results:
            param_str = ", ".join(f"{k}={v}" for k, v in r.params.items()) if r.params else "-"
            mem_str = f"{r.memory_peak_mb:.1f}" if r.memory_peak_mb is not None else "-"
            print(fmt.format(
                r.name[:45],
                param_str[:25],
                str(len(r.times)),
                f"{r.mean:.4f}",
                f"{r.median:.4f}",
                f"{r.stdev:.4f}",
                f"{r.min:.4f}",
                f"{r.max:.4f}",
                mem_str,
            ))
        print()


def save_json(results: list[BenchResult], path: str | Path) -> None:
    path = Path(path)
    data = [r.to_dict() for r in results]
    path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {path}")


def save_csv(results: list[BenchResult], path: str | Path) -> None:
    path = Path(path)
    lines: list[str] = [
        "name,category,params,n_runs,mean_s,median_s,stdev_s,min_s,max_s,"
        "memory_peak_mb"
    ]
    for r in results:
        param_str = "; ".join(f"{k}={v}" for k, v in r.params.items())
        mem = f"{r.memory_peak_mb:.2f}" if r.memory_peak_mb is not None else ""
        lines.append(
            f'"{r.name}","{r.category}","{param_str}",{len(r.times)},'
            f"{r.mean:.6f},{r.median:.6f},{r.stdev:.6f},{r.min:.6f},{r.max:.6f},{mem}"
        )
    path.write_text("\n".join(lines))
    print(f"Results saved to {path}")
