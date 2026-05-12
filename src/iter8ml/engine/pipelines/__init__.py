from iter8ml.engine.pipelines.executor import PipelineExecutor, PipelineMode


def visualize_pipeline(output_format: str = "mermaid") -> str:
    executor = PipelineExecutor()
    if output_format == "mermaid":
        return executor.get_mermaid_graph()
    return ""


__all__ = ["PipelineExecutor", "PipelineMode", "visualize_pipeline"]
