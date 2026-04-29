from tabular_blueprint.pipelines.executor import PipelineExecutor, PipelineMode
from tabular_blueprint.pipelines.hamilton_executor import HamiltonExecutor


def visualize_pipeline(output_format: str = "mermaid") -> str:
    executor = PipelineExecutor()
    if output_format == "mermaid":
        return executor.get_mermaid_graph()
    return ""


__all__ = ["HamiltonExecutor", "PipelineExecutor", "PipelineMode", "visualize_pipeline"]
