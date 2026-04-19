"""CLI entry point for tabular-blueprint."""

import importlib
import json
from pathlib import Path

import typer

from configs.experiment import ExperimentConfig
from configs.hardware import HardwareProfile
from core.constants import from_task_type
from core.data.loaders import load_data
from core.engine.trainer import Trainer
from core.models.factory import get_model_class
from core.services.report_service import ReportService

app = typer.Typer(name="tabblueprint", help="A high-velocity iteration framework for tabular ML")


@app.command()
def init(data: str | None = None):
    """Initialize workspace and optionally load data."""
    workspace = Path("workspace")
    workspace.mkdir(exist_ok=True)
    (workspace / "artifacts").mkdir(exist_ok=True)

    (workspace / "experiments.jsonl").touch(exist_ok=True)
    (workspace / "registry.json").write_text("{}")

    typer.echo("Workspace initialized.")
    if data:
        typer.echo(f"Data path set to: {data}")


@app.command()
def run(
    config: str = typer.Option(None, "--config", "-c", help="Path to experiment config module"),
    data_path: str = typer.Option(None, "--data", "-d", help="Path to data file"),
    target_col: str = typer.Option(None, "--target", "-t", help="Target column name"),
    task: str = typer.Option("classification", "--task", help="classification or regression"),
    models: list[str] | None = typer.Option(None, "--models", "-m", help="Model names to run"),
):
    """Run an experiment."""
    experiment_config = None
    if config:
        spec = importlib.util.spec_from_file_location("experiment_config", config)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            experiment_config = getattr(module, "config", None)

    if experiment_config is None:
        experiment_config = ExperimentConfig(
            name="experiment",
            task=from_task_type(task),
            target_col=target_col or "target",
            data_path=data_path or "",
        )

    if models:
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

    typer.echo(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    typer.echo(f"Task: {experiment_config.task}, Target: {experiment_config.target_col}")

    trainer = Trainer(experiment_config)
    results = trainer.run(df)

    typer.echo("\nResults:")
    for model, scores in results.items():
        typer.echo(f"  {model}: {scores}")


@app.command()
def leaderboard(
    top: int = typer.Option(10, "--top", "-n", help="Number of top results to show"),
    metric: str = typer.Option(None, "--metric", help="Sort by this metric"),
):
    """Show experiment leaderboard."""
    report = ReportService().format_leaderboard_console(metric=metric, limit=top)
    typer.echo(report)


@app.command()
def registry(action: str = typer.Argument("show", help="show or promote")):
    """Show or manage model registry."""
    registry_path = Path("workspace/registry.json")
    if not registry_path.exists():
        typer.echo("Registry is empty.")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    if action == "show":
        typer.echo("\n# Model Registry\n")
        for key, entry in registry.items():
            typer.echo(f"**{key}**")
            typer.echo(f"  Model: {entry.get('model')}")
            typer.echo(f"  Run ID: {entry.get('run_id')}")
            typer.echo(f"  Score: {entry.get('score')}")
            typer.echo(f"  Registered: {entry.get('registered_at')}")
            typer.echo("")
    else:
        typer.echo(f"Unknown action: {action}")


@app.command()
def hardware():
    """Show detected hardware profile."""
    profile = HardwareProfile.detect()
    typer.echo("\n# Hardware Profile")
    typer.echo(f"GPU: {profile.gpu_name or 'None'}")
    typer.echo(f"VRAM: {profile.vram_gb} GB")
    typer.echo(f"RAM: {profile.system_ram_gb} GB")
    typer.echo(f"CPU Cores: {profile.cpu_cores}")


@app.command()
def drift(
    reference: str = typer.Option(..., "--reference", "-r"),
    new: str = typer.Option(..., "--new", "-n"),
):
    """Detect distribution drift between two datasets."""
    from core.monitoring.drift import DriftDetector

    ref_df = load_data(reference)
    new_df = load_data(new)

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    typer.echo("\n# Drift Detection Report")
    typer.echo(f"Drift detected: {report.drift_detected}")
    typer.echo(f"Columns tested: {report.n_columns_tested}")
    typer.echo(f"Columns drifted: {report.n_drifted}\n")

    for col_result in report.column_results:
        status = "DRIFT" if col_result.drift_detected else "OK"
        typer.echo(
            f"{status} | {col_result.column} | p={col_result.p_value:.6f} | {col_result.test_used}"
        )


@app.command()
def state():
    """Generate and display current experiment state."""
    from core.engine.state_observer import StateObserver

    observer = StateObserver()
    content = observer.generate()
    typer.echo(content)


@app.command()
def hpo(
    data_path: str = typer.Option(..., "--data", "-d"),
    target_col: str = typer.Option(..., "--target", "-t"),
    model: str = typer.Option("catboost", "--model", "-m"),
    task: str = typer.Option("classification", "--task"),
    trials: int = typer.Option(50, "--trials", "-n"),
):
    """Run hyperparameter optimization for a model."""
    from core.engine.hpo import optimize_model, setup_hpo_components

    try:
        X, y, evaluator, search_space = setup_hpo_components(data_path, target_col, task, model)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

    model_cls = get_model_class(model)

    typer.echo(f"Running HPO for {model} ({trials} trials)...")
    result = optimize_model(
        model_cls,
        X,
        y,
        evaluator,
        model,
        n_trials=trials,
        search_space=search_space,
        task=task,
    )

    typer.echo(f"\nBest params: {result['best_params']}")
    typer.echo(f"Best value: {result['best_value']:.4f}")
    typer.echo(f"Trials completed: {result['n_trials']}")


if __name__ == "__main__":
    app()
