from iter8ml.config import PipelineSpec
from iter8ml.engine.pipelines import (
    PipelineExecutor,
    PipelineMode,
    describe_pipeline,
    visualize_pipeline,
)


def test_describe_pipeline_delegates_to_executor():
    spec = PipelineSpec()
    result = describe_pipeline(spec)
    assert isinstance(result, list)
    assert result
    assert result[0]["step"]


def test_visualize_pipeline_non_mermaid_returns_empty():
    assert visualize_pipeline(output_format="json") == ""


def test_public_exports():
    assert PipelineExecutor is not None
    assert PipelineMode.TRAINING.value == "training"
