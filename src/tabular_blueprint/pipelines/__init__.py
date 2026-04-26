"""Data pipelines and Hamilton orchestration."""

from tabular_blueprint.pipelines.hamilton_executor import HamiltonExecutor


def visualize_pipeline(output_format: str = "mermaid") -> str:
    """Generate visual representation of the data pipeline."""
    executor = HamiltonExecutor()
    if output_format == "mermaid":
        return executor.get_mermaid_graph()
    return ""
