"""CLI command to serve the iter8ml MCP server for LLM agents."""

from __future__ import annotations

import typer

from .main import app


@app.command()
def mcp(
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="FastMCP transport: stdio, sse, or streamable-http."
    ),
) -> None:
    """Serve the iter8ml MCP server (requires the `mcp` package)."""
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        typer.echo("The 'mcp' package is required: uv add mcp", err=True)
        raise typer.Exit(1) from None

    from iter8ml.services import mcp as mcp_module

    server = mcp_module.mcp
    server.run(transport=transport)  # type: ignore[arg-type]
