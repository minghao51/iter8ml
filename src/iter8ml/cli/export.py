"""Export and registry commands."""

import typer

from iter8ml.session import ExperimentSession

from .main import app


@app.command()
def export(
    key: str = typer.Argument(..., help="Registry key (e.g. experiment:classification)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column name"),
) -> None:
    """Export champion model as a portable prediction package."""
    session = ExperimentSession()
    try:
        export_path = session.export(key, output_dir=output)
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
def registry(action: str = typer.Argument("show", help="show or promote")) -> None:
    """Show or manage model registry."""
    ws = ExperimentSession().workspace

    import json

    data = json.loads(ws.registry_path.read_text()) if ws.registry_path.exists() else {}
    if not data:
        typer.echo("Registry is empty.")
        return

    if action == "show":
        typer.echo("\n# Model Registry\n")
        for key, entry in data.items():
            typer.echo(f"**{key}**")
            typer.echo(f"  Model: {entry.get('model')}")
            typer.echo(f"  Run ID: {entry.get('run_id')}")
            typer.echo(f"  Score: {entry.get('score')}")
            typer.echo(f"  Registered: {entry.get('registered_at')}")
            typer.echo("")
