"""Export and registry commands."""

import typer

from iter8ml.services.registry import RegistryService
from iter8ml.session import ExperimentSession

from .main import app


@app.command()
def export(
    key: str = typer.Argument(..., help="Registry key (e.g. experiment:classification)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Target column name (recorded in metadata; predictor drops it before inference)",
    ),
    positive_class: str | None = typer.Option(
        None,
        "--positive-class",
        help="Positive class used in training (recorded in metadata so consumers "
        "can interpret predict_proba column orientation)",
    ),
) -> None:
    """Export champion model as a portable prediction package."""
    session = ExperimentSession()
    try:
        export_path = session.export(
            key,
            output_dir=output,
            target_col=target,
            positive_class=positive_class,
        )
        typer.echo(f"Exported to: {export_path}")
        typer.echo("  Model artifact: model.artifact")
        typer.echo("  Preprocessing:  pipelines/preprocessing.py")
        typer.echo("  Predictor:      predictor.py")
        typer.echo("\nUsage:")
        typer.echo(f"  cd {export_path}")
        typer.echo("  python predictor.py data.csv")
    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e


@app.command()
def registry(
    action: str = typer.Argument("show", help="'show', or 'promote <run_id> <key>'"),
    run_id: str | None = typer.Argument(None, help="Run ID to promote (action 'promote')"),
    key: str | None = typer.Argument(None, help="Registry key to promote into (action 'promote')"),
) -> None:
    """Show or manage model registry."""
    session = ExperimentSession()
    service = RegistryService(workspace=session.workspace)

    if action == "show":
        data = service.get_all()
        if not data:
            typer.echo("Registry is empty.")
            return
        typer.echo("\n# Model Registry\n")
        for registry_key, entry in data.items():
            typer.echo(f"**{registry_key}**")
            typer.echo(f"  Model: {entry.get('model')}")
            typer.echo(f"  Run ID: {entry.get('run_id')}")
            typer.echo(f"  Score: {entry.get('score')}")
            typer.echo(f"  Registered: {entry.get('registered_at')}")
            typer.echo("")
        return

    if action == "promote":
        if not run_id or not key:
            typer.echo("Usage: iter8 registry promote <run_id> <key>")
            raise typer.Exit(1)
        result = service.promote_run(run_id, key, session.workspace.experiments_path)
        typer.echo(result.message)
        if result.status == "not_found":
            raise typer.Exit(1)
        return

    typer.echo(f"Unknown action '{action}'. Usage: iter8 registry [show | promote <run_id> <key>]")
    raise typer.Exit(1)
