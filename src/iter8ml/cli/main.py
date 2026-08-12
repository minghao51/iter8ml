"""CLI app assembly — init and hardware commands."""

from pathlib import Path

import typer

from iter8ml.config import HardwareProfile
from iter8ml.workspace import Workspace

app = typer.Typer(name="iter8", help="A high-velocity iteration framework for tabular ML")


@app.command()
def init(
    data: str | None = None,
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Seed the workspace with a bundled demo dataset (Telco Churn).",
    ),
    force_reset_registry: bool = typer.Option(
        False,
        "--force-reset-registry",
        help="Reset registry.json even if it already exists (destructive).",
    ),
) -> None:
    """Initialize workspace and optionally load data."""
    workspace = Workspace()
    workspace.init()
    if force_reset_registry:
        workspace.registry_path.write_text("{}")
    typer.echo("Workspace initialized.")
    if demo:
        demo_path = workspace.seed_demo_data()
        try:
            display = demo_path.relative_to(Path.cwd())
        except ValueError:
            display = demo_path
        typer.echo(f"Demo dataset seeded: {demo_path}")
        typer.echo(f"Next: iter8 run --data {display} --target Churn")
    if data:
        typer.echo(f"Data path set to: {data}")


@app.command()
def hardware() -> None:
    """Show detected hardware profile."""
    profile = HardwareProfile.detect()
    typer.echo("\n# Hardware Profile")
    typer.echo(f"GPU: {profile.gpu_name or 'None'}")
    typer.echo(f"VRAM: {profile.vram_gb} GB")
    typer.echo(f"RAM: {profile.system_ram_gb} GB")
    typer.echo(f"CPU Cores: {profile.cpu_cores}")
