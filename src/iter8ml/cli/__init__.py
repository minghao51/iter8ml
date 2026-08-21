"""CLI module — imports subcommands for side-effect registration on the shared app."""

import iter8ml.cli.analyze
import iter8ml.cli.export
import iter8ml.cli.mcp
import iter8ml.cli.medallion
import iter8ml.cli.optimize
import iter8ml.cli.run  # noqa: F401
from iter8ml.cli.main import app as app
