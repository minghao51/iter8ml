"""Run command — execute a tabular ML experiment."""

from pathlib import Path

import typer

from iter8ml.config import ExperimentConfig
from iter8ml.constants import from_task_type
from iter8ml.data.loader import load_data
from iter8ml.session import ExperimentSession
from iter8ml.utils.io import load_events

from .main import app


def _find_last_run_id(config: ExperimentConfig) -> str | None:
    from iter8ml.workspace import Workspace

    ws = Workspace()
    events = load_events(ws.experiments_path)
    run_ids = [
        e.get("run_id")
        for e in events
        if e.get("event") == "experiment_started" and e.get("run_id")
    ]
    return run_ids[-1] if run_ids else None


@app.command()
def run(
    config: str = typer.Option(
        None, "--config", "-c", help="Config file (.yaml/.yml/.toml/.json/.py)"
    ),
    data_path: str = typer.Option(None, "--data", "-d", help="Path to data file"),
    target_col: str = typer.Option(None, "--target", "-t", help="Target column name"),
    task: str = typer.Option("classification", "--task", help="classification or regression"),
    models: list[str] | None = typer.Option(None, "--models", "-m", help="Model names to run"),
    quick: bool = typer.Option(False, "--quick", help="Fast mode: 2 folds, 20% data, skip SHAP"),
    resume: bool = typer.Option(
        False, "--resume", help="Resume previous run, skip completed models"
    ),
    allow_unsafe_config: bool = typer.Option(
        False,
        "--allow-unsafe-config",
        help="Allow loading .py config files (unsafe, executes code).",
    ),
) -> None:
    """Run an experiment."""
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
        experiment_config = ExperimentConfig(
            name="experiment",
            task=from_task_type(task),
            target_col=target_col or "target",
            data_path=data_path or "",
        )

    if models is not None and models:
        experiment_config.models = models

    if not data_path:
        data_path = experiment_config.data_path
    if not data_path:
        typer.echo("Error: --data or config.data_path required")
        raise typer.Exit(1)

    try:
        df = load_data(data_path)
    except ValueError as e:
        typer.echo(f"Error loading data: {e}")
        raise typer.Exit(code=1) from e

    if quick:
        experiment_config.cv_folds = 2
        experiment_config.shap_enabled = False
        experiment_config.calibration = "none"
        experiment_config.data_sample = 0.2
        typer.echo("[quick mode] 2 folds, 20% data, SHAP/calibration disabled")

    typer.echo(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    typer.echo(f"Task: {experiment_config.task}, Target: {experiment_config.target_col}")

    session = ExperimentSession()
    results = session.run(experiment_config, df)

    typer.echo("\nResults:")
    for model, scores in results.items():
        typer.echo(f"  {model}: {scores}")
