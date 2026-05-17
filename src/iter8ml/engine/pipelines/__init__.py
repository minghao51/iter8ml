from __future__ import annotations

from typing import TYPE_CHECKING, Any

from iter8ml.engine.pipelines.executor import PipelineExecutor, PipelineMode

if TYPE_CHECKING:
    from iter8ml.config import PipelineSpec


def describe_pipeline(spec: PipelineSpec) -> list[dict[str, Any]]:
    return PipelineExecutor().describe_pipeline(spec)


def visualize_pipeline(output_format: str = "mermaid", spec: PipelineSpec | None = None) -> str:
    executor = PipelineExecutor()
    if output_format == "mermaid":
        return executor.get_mermaid_graph(spec=spec)
    return ""


__all__ = [
    "PipelineExecutor",
    "PipelineMode",
    "describe_pipeline",
    "visualize_pipeline",
]
