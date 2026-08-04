"""CLI commands for plans, medallion products, catalog, and docs projections."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from iter8ml.config import ExperimentConfig
from iter8ml.data.loader import load_data
from iter8ml.dataflows.bronze import materialize_bronze
from iter8ml.domain.manifests import SourceSpec
from iter8ml.runtime.plan import compile_run_plan
from iter8ml.services.docs_export import DocsExporter
from iter8ml.services.retention import garbage_collect
from iter8ml.storage.catalog import LocalCatalogStore
from iter8ml.storage.local import LocalArtifactStore
from iter8ml.verification.schema import verify_product
from iter8ml.workspace import Workspace

from .main import app

data_app = typer.Typer(help="Medallion data product commands")
catalog_app = typer.Typer(help="Catalog commands")
docs_app = typer.Typer(help="Static documentation projections")
app.add_typer(data_app, name="data")
app.add_typer(catalog_app, name="catalog")
app.add_typer(docs_app, name="docs")


@app.command()
def plan(
    config: str = typer.Option(..., "--config", "-c"),
    graph: bool = typer.Option(False, "--graph"),
) -> None:
    """Compile a config into a redacted, deterministic RunPlan."""
    resolved = compile_run_plan(ExperimentConfig.from_file(config))
    typer.echo(resolved.model_dump_json(indent=2))
    if graph:
        typer.echo("graph TD\n    bronze --> silver --> gold --> platinum")


@app.command()
def verify(product_id: str, deep: bool = typer.Option(False, "--deep")) -> None:
    """Verify a committed product and its artifact checksums."""
    result = verify_product(LocalArtifactStore(Workspace().root), product_id, deep=deep)
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result.get("ok"):
        raise typer.Exit(1)


@data_app.command("ingest")
def data_ingest(
    data_path: str = typer.Option(..., "--data"),
    name: str = typer.Option("dataset", "--name"),
) -> None:
    """Snapshot CSV or Parquet input as a Bronze product."""
    suffix = Path(data_path).suffix.lower()
    source_types = {".csv": "csv", ".parquet": "parquet"}
    if suffix not in source_types:
        raise typer.BadParameter("--data must be a CSV or Parquet file")
    workspace = Workspace().init()
    frame = load_data(data_path)
    source = SourceSpec(
        name=name,
        source_type=source_types[suffix],  # type: ignore[arg-type]
        uri=data_path,
    )
    store = LocalArtifactStore(workspace.root)
    manifest = materialize_bronze(frame, source, store)
    LocalCatalogStore(workspace.root).register_product(manifest)
    typer.echo(manifest.product_id)


@catalog_app.command("rebuild")
def catalog_rebuild() -> None:
    """Rebuild the catalog from committed manifests."""
    workspace = Workspace().init()
    result = LocalCatalogStore(workspace.root).rebuild(LocalArtifactStore(workspace.root))
    typer.echo(json.dumps(result, sort_keys=True))


@catalog_app.command("query")
def catalog_query(sql: str) -> None:
    """Run a read-only SQL query against the local catalog."""
    frame = LocalCatalogStore(Workspace().root).query(sql)
    typer.echo(frame.write_json())


@docs_app.command("export")
def docs_export(limit: int = typer.Option(100, "--limit", min=1, max=1000)) -> None:
    """Emit bounded site-data JSON projections."""
    output = DocsExporter(Workspace().init()).export(limit=limit)
    typer.echo(str(output))


@app.command()
def lineage(product_id: str) -> None:
    """Show upstream product identities for a committed product."""
    manifest = LocalArtifactStore(Workspace().root).read_manifest(product_id)
    typer.echo(json.dumps({"product_id": product_id, "inputs": manifest.inputs}, indent=2))


@app.command()
def gc(dry_run: bool = typer.Option(True, "--dry-run/--apply")) -> None:
    """Report or remove abandoned temporary product directories."""
    paths = garbage_collect(LocalArtifactStore(Workspace().root), dry_run=dry_run)
    for path in paths:
        typer.echo(path)
