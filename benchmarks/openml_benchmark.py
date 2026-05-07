"""OpenML/sklearn benchmark suite for validating default hyperparameters.

Each dataset in the config is loaded, converted to Polars, and run through
each default model via the Evaluator.  Metrics + timing are recorded as JSON
to ``benchmarks/results/``.

Supports parameter sweeps and regression checking against a baseline.

Usage::

    uv run python -m benchmarks.run_openml_benchmark [--quick]
    uv run python -m benchmarks.run_openml_benchmark \
        --sweep-config sweeps/default_vs_gpu.yaml
    uv run python -m benchmarks.run_openml_benchmark \
        --check-regression benchmarks/results/baseline_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

BENCHMARKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
CONFIG_PATH = BENCHMARKS_DIR / "configs" / "default_benchmark.yaml"
BASELINE_PATH = RESULTS_DIR / "baseline_summary.json"

REGRESSION_THRESHOLD = 0.02  # 2% default allowed regression


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_openml_dataset(dataset_id: int) -> tuple[pl.DataFrame, str]:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=dataset_id, as_frame=False, parser="auto")
    X = data.data
    if not hasattr(X, "shape"):
        X = np.asarray(X)

    # Build columns, handling object arrays with mixed types (e.g. NaN + strings)
    columns: dict[str, pl.Series] = {}
    for i in range(X.shape[1]):
        col = X[:, i]
        if hasattr(col, "dtype") and col.dtype == object:
            # Mixed types: try numeric, fallback to string
            str_vals = [str(v) if v is not None else None for v in col]
            try:
                float_vals = [float(v) for v in str_vals]
                columns[f"feat_{i}"] = pl.Series(float_vals)
            except ValueError:
                columns[f"feat_{i}"] = pl.Series(str_vals)
        else:
            columns[f"feat_{i}"] = pl.Series(col)

    df = pl.DataFrame(columns)
    target_col = "target"
    target_vals = data.target
    if hasattr(target_vals, "dtype") and target_vals.dtype.kind in ("O", "U"):
        target_series = pl.Series([str(v) if v is not None else None for v in target_vals])
    elif not hasattr(target_vals, "dtype"):
        target_series = pl.Series(np.asarray(target_vals))
    else:
        target_series = pl.Series(target_vals)
    df = df.with_columns(target_series.alias(target_col))
    return df, target_col


def load_sklearn_dataset(name: str) -> tuple[pl.DataFrame, str]:
    import sklearn.datasets as skd

    loader = getattr(skd, f"load_{name}", None)
    if loader is None:
        raise ValueError(f"No sklearn loader for: {name}")
    data = loader()
    X = data.data
    y = data.target
    n_features = X.shape[1]
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(n_features)})
    target_col = "target"
    df = df.with_columns(pl.Series(target_col, y.astype(np.float64)))
    return df, target_col


def load_dataset(source: str, dataset_id: int | None, name: str) -> tuple[pl.DataFrame, str]:
    if source == "openml":
        assert dataset_id is not None
        return load_openml_dataset(dataset_id)
    if source == "sklearn":
        return load_sklearn_dataset(name)
    raise ValueError(f"Unknown source: {source}")


def _model_kwargs(model_name: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    if overrides and model_name in overrides:
        return overrides[model_name]
    return {}


def _preprocess_for_benchmark(
    df: pl.DataFrame, target_col: str, task: str
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    from tabular_blueprint.data.adapter import DataAdapter

    adapter = DataAdapter()
    X, y = adapter.transform(df, target_col)

    if y.dtype == object or (hasattr(y.dtype, "kind") and y.dtype.kind in ("U", "S", "O")):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    if X.dtype == object or (hasattr(X.dtype, "kind") and X.dtype.kind in ("U", "S", "O")):
        str_cols = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if hasattr(col, "dtype") and col.dtype == object:
                try:
                    X[:, i] = col.astype(float)
                except (ValueError, TypeError):
                    str_cols.append(i)
        if str_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[:, str_cols] = enc.fit_transform(X[:, str_cols])
        X = X.astype(np.float64)

    return X, y


def run_benchmark_for_dataset(
    df: pl.DataFrame,
    target_col: str,
    task: str,
    models: list[str],
    metrics: list[str],
    cv_folds: int,
    random_seed: int,
    dataset_name: str,
    variant_name: str,
    model_overrides: dict[str, Any] | None,
    quick: bool = False,
) -> list[dict[str, Any]]:
    from tabular_blueprint.config import CVStrategy, ExperimentConfig
    from tabular_blueprint.constants import TaskType
    from tabular_blueprint.engine.evaluator import Evaluator
    from tabular_blueprint.models.factory import get_model_class

    X, y = _preprocess_for_benchmark(df, target_col, task)
    n_rows, n_features = X.shape

    config = ExperimentConfig(
        name=f"benchmark_{dataset_name}_{variant_name}",
        task=TaskType(task),
        target_col=target_col,
        data_path="",
        cv_folds=cv_folds,
        cv_strategy=CVStrategy.STRATIFIED if task == "classification" else CVStrategy.KFOLD,
        metrics=metrics,
        random_seed=random_seed,
    )
    evaluator = Evaluator(config)

    results: list[dict[str, Any]] = []
    for model_name in models:
        try:
            model_cls = get_model_class(model_name)
        except Exception:
            continue

        start = time.perf_counter()
        try:
            kwargs = _model_kwargs(model_name, model_overrides)
            cv_scores = evaluator.evaluate(model_cls, X, y, task=task, **kwargs)
            duration = round(time.perf_counter() - start, 3)
        except Exception as e:
            results.append(
                {
                    "dataset": dataset_name,
                    "variant": variant_name,
                    "model": model_name,
                    "task": task,
                    "n_rows": n_rows,
                    "n_features": n_features,
                    "error": str(e),
                }
            )
            continue

        results.append(
            {
                "dataset": dataset_name,
                "variant": variant_name,
                "model": model_name,
                "task": task,
                "n_rows": n_rows,
                "n_features": n_features,
                "cv_scores": {k: round(float(v), 6) for k, v in cv_scores.items()},
                "duration_seconds": duration,
            }
        )

    return results


def load_sweep_config(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [{"name": "default", "config": {}}]
    data = yaml.safe_load(Path(path).read_text())
    return data.get("variants", [{"name": "default", "config": {}}])


def run_all(
    quick: bool = False,
    sweep_config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    config = load_config()
    all_results: list[dict[str, Any]] = []
    variants = load_sweep_config(sweep_config_path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        variant_name = variant["name"]
        model_overrides = variant.get("config", {})
        print(f"\n[variant: {variant_name}] overrides={model_overrides}")

        for ds in config["datasets"]:
            name = ds["name"]
            task = ds["task"]
            models = config["models"][task]
            metrics = config["metrics"][task]
            source = ds["source"]
            dataset_id = ds.get("id")

            print(f"  [{name}] loading...")
            try:
                df, target_col = load_dataset(source, dataset_id, name)
            except Exception as e:
                print(f"  [{name}] SKIPPED (load failed: {e})")
                continue

            if quick and df.shape[0] > 10_000:
                df = df.sample(fraction=0.2, seed=config["random_seed"])
                print(f"  [{name}] sampled to {df.shape[0]} rows (quick mode)")

            print(
                f"  [{name}] {df.shape[0]} rows, {df.shape[1] - 1} features, "
                f"running {len(models)} models..."
            )
            ds_results = run_benchmark_for_dataset(
                df=df,
                target_col=target_col,
                task=task,
                models=models,
                metrics=metrics,
                cv_folds=config["cv_folds"],
                random_seed=config["random_seed"],
                dataset_name=name,
                variant_name=variant_name,
                model_overrides=model_overrides,
                quick=quick,
            )

            result_path = RESULTS_DIR / f"{name}_{variant_name}.json"
            result_path.write_text(json.dumps(ds_results, indent=2))
            print(f"  [{name}] saved {len(ds_results)} results to {result_path}")
            all_results.extend(ds_results)

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary: {len(all_results)} total results saved to {summary_path}")
    return all_results


def check_regression(
    current_results: list[dict[str, Any]],
    baseline_path: str | Path,
    threshold: float = REGRESSION_THRESHOLD,
) -> bool:
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found at {baseline_path}")
        return False

    baseline_data = json.loads(baseline_path.read_text())
    baseline_scores: dict[tuple[str, str, str], float] = {}
    for r in baseline_data:
        if "error" in r:
            continue
        key = (r["dataset"], r["variant"], r["model"])
        # Use first metric as primary score
        first_metric = next(iter(r["cv_scores"]))
        baseline_scores[key] = r["cv_scores"][first_metric]

    regressions: list[str] = []
    for r in current_results:
        if "error" in r:
            continue
        key = (r["dataset"], r["variant"], r["model"])
        if key not in baseline_scores:
            continue
        first_metric = next(iter(r["cv_scores"]))
        baseline = baseline_scores[key]
        current = r["cv_scores"][first_metric]
        # For metrics where higher is better
        # (assume classification=auc, regression=rmse needs flip)
        # Simple heuristic: if metric name contains 'loss' or 'error'
        # or 'mae' or 'rmse', lower is better
        metric_name = first_metric.lower()
        lower_is_better = any(
            word in metric_name for word in ("loss", "error", "mae", "rmse", "mse")
        )
        if lower_is_better:
            change = (baseline - current) / abs(baseline) if baseline != 0 else 0
        else:
            change = (current - baseline) / abs(baseline) if baseline != 0 else 0

        if change < -threshold:
            regressions.append(
                f"  {key[0]}/{key[2]}: {first_metric} "
                f"baseline={baseline:.4f} current={current:.4f} ({change:.1%})"
            )

    if regressions:
        print(f"\nREGRESSION DETECTED (>{threshold:.0%}):")
        for line in regressions:
            print(line)
        return False
    print(f"\nNo regressions detected (threshold={threshold:.0%})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenML benchmark suite for default validation"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Sample large datasets for fast feedback"
    )
    parser.add_argument(
        "--sweep-config", type=str, default=None, help="YAML file with parameter sweep variants"
    )
    parser.add_argument(
        "--check-regression",
        type=str,
        default=None,
        help="Path to baseline JSON to compare against",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=REGRESSION_THRESHOLD,
        help="Allowed score regression fraction",
    )
    parser.add_argument(
        "--save-baseline", action="store_true", help="Copy summary.json to baseline_summary.json"
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if any dataset/model run returns an error entry",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  tabular-blueprint OpenML Benchmark Suite")
    print("=" * 60)
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    if args.sweep_config:
        print(f"  Sweep: {args.sweep_config}")
    print()

    t0 = time.perf_counter()
    results = run_all(quick=args.quick, sweep_config_path=args.sweep_config)
    total = time.perf_counter() - t0

    print(f"\nDone: {len(results)} benchmark results in {total:.1f}s")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nFound {len(errors)} benchmark execution errors:")
        for e in errors[:20]:
            print(f"  {e.get('dataset')}/{e.get('model')}: {e.get('error')}")
        if args.fail_on_errors:
            sys.exit(1)

    if args.save_baseline:
        baseline_dest = RESULTS_DIR / "baseline_summary.json"
        baseline_dest.write_text(json.dumps(results, indent=2))
        print(f"Baseline saved to {baseline_dest}")

    if args.check_regression:
        ok = check_regression(results, args.check_regression, threshold=args.regression_threshold)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
