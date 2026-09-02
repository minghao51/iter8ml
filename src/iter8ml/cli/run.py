"""Run command — execute a tabular ML experiment."""

from pathlib import Path

import polars as pl
import typer

from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.data.loader import load_data
from iter8ml.exceptions import DataLoadError, TabularBlueprintError
from iter8ml.session import ExperimentSession

from .main import app


@app.command()
def run(
    config: str = typer.Option(
        None, "--config", "-c", help="Config file (.yaml/.yml/.toml/.json/.py)"
    ),
    data_path: str = typer.Option(None, "--data", "-d", help="Path to data file"),
    target_col: str = typer.Option(None, "--target", "-t", help="Target column name"),
    task: str = typer.Option("classification", "--task", help="classification or regression"),
    models: list[str] | None = typer.Option(None, "--models", "-m", help="Model names to run"),
    quick: bool = typer.Option(False, "--quick", help="Fast mode: 2 folds and a 20% data sample"),
    resume: bool = typer.Option(
        False, "--resume", help="Resume previous run, skip completed models"
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Validate config against the data and exit without training "
        "(target sanity, CV feasibility, leakage hints).",
    ),
    allow_unsafe_config: bool = typer.Option(
        False,
        "--allow-unsafe-config",
        help="Allow loading .py config files (unsafe, executes code).",
    ),
) -> None:
    """Run an experiment (or validate it with --check)."""
    experiment_config = None
    if config:
        config_path = Path(config)
        if not config_path.exists() or not config_path.is_file():
            typer.echo(f"Error: config file not found: {config}")
            raise typer.Exit(1)

        try:
            experiment_config = ExperimentConfig.from_file(
                config_path, allow_unsafe_python=allow_unsafe_config
            )
        except (ValueError, FileNotFoundError) as e:
            typer.echo(f"Error loading config: {e}")
            raise typer.Exit(1) from e

    if experiment_config is None:
        try:
            task_type = TaskType(task)
        except ValueError:
            valid = [t.value for t in TaskType]
            typer.echo(f"Error: unknown --task '{task}'. Valid tasks: {valid}")
            raise typer.Exit(1) from None
        experiment_config = ExperimentConfig(
            name="experiment",
            task=task_type,
            target_col=target_col or "target",
            data_path=data_path or "",
        )

    if models is not None and models:
        from iter8ml.config import _raise_if_unknown_model_names

        # Validate BEFORE overwriting: assigning post-parse bypasses the
        # pydantic field validator, which previously let unknown names through
        # to a confusing per-model failure mid-run.
        try:
            _raise_if_unknown_model_names(set(models), "model names")
        except ValueError as e:
            typer.echo(f"Error: {e}")
            raise typer.Exit(1) from e
        experiment_config.models = models

    if not data_path:
        data_path = experiment_config.data_path
    if not data_path:
        typer.echo("Error: --data or config.data_path required")
        raise typer.Exit(1)

    try:
        df = load_data(data_path)
    except (ValueError, DataLoadError) as e:
        typer.echo(f"Error loading data: {e}")
        raise typer.Exit(code=1) from e

    if quick:
        experiment_config.cv_folds = 2
        experiment_config.data_sample = 0.2
        typer.echo("[quick mode] 2 folds, 20% data sample")

    typer.echo(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    typer.echo(f"Task: {experiment_config.task}, Target: {experiment_config.target_col}")

    if check:
        _run_check(experiment_config, df)
        return

    session = ExperimentSession()
    try:
        results = session.run(experiment_config, df)
    except TabularBlueprintError as e:
        typer.echo(f"\nRun failed: {e}")
        if getattr(e, "context", None):
            typer.echo(f"  context: {e.context}")
        raise typer.Exit(1) from e

    _print_results(results)

    succeeded = sum(
        1
        for entry in results.values()
        if isinstance(entry, dict) and "cv_scores" in entry and entry["cv_scores"]
    )
    if succeeded == 0:
        typer.echo(
            "\nError: no model trained successfully. "
            "Run `iter8 run --check` for a pre-flight diagnosis."
        )
        raise typer.Exit(1)


def _run_check(experiment_config: ExperimentConfig, df: pl.DataFrame) -> None:
    from iter8ml.verification.preflight import has_errors, run_preflight

    typer.echo("\nPre-flight checks (config <-> data):")
    issues = run_preflight(experiment_config, df)
    if not issues:
        typer.echo("  All checks passed.")
        typer.echo(
            f"Would train models={experiment_config.models} with "
            f"metrics={experiment_config.metrics} "
            f"(primary={experiment_config.primary_metric}), "
            f"cv={experiment_config.cv_strategy.value}/{experiment_config.cv_folds} folds."
        )
        return
    for issue in issues:
        typer.echo(f"  {issue.format()}")
    if has_errors(issues):
        typer.echo("\nCheck failed: fix the errors above before running.")
        raise typer.Exit(1)
    typer.echo("\nCheck passed with warnings (see above).")


def _print_results(results: dict) -> None:
    typer.echo("\nResults:")
    for model, entry in results.items():
        if isinstance(entry, dict) and "error" in entry:
            typer.echo(f"  {model}: ERROR - {entry['error']}")
        elif isinstance(entry, dict) and "cv_scores" in entry:
            scores = entry["cv_scores"]
            stds = entry.get("cv_std") or {}
            parts = []
            for k, v in scores.items():
                std = stds.get(k)
                parts.append(f"{k}={v:.4f}" + (f" ±{std:.4f}" if std else ""))
            score_str = ", ".join(parts) if scores else "N/A"
            typer.echo(f"  {model}: {score_str}")
        else:
            typer.echo(f"  {model}: {entry}")
