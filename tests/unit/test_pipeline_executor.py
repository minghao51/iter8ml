import polars as pl
import pytest

from iter8ml.engine.pipelines.executor import PipelineExecutor, PipelineMode
from iter8ml.exceptions import HamiltonUnavailableError


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "num": [1.0, None, 3.0, 4.0],
            "cat": pl.Series(["x", "y", "x", None], dtype=pl.Categorical),
            "date": pl.Series(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ).str.to_datetime(),
        }
    )


class TestPipelineExecutor:
    def test_available_with_hamilton(self):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor()
        assert executor.available is True

    def test_run_preprocessing_fills_nulls(self, sample_df):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor()
        result = executor.run_preprocessing(sample_df)
        assert result["num"].null_count() == 0
        assert result["cat"].null_count() == 0

    def test_run_preprocessing_decomposes_dates(self, sample_df):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor()
        result = executor.run_preprocessing(sample_df)
        assert "date_year" in result.columns
        assert "date" not in result.columns

    def test_execute_returns_dict(self, sample_df):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor()
        result = executor.execute(inputs={"df": sample_df})
        assert "processed_dataframe" in result
        assert isinstance(result["processed_dataframe"], pl.DataFrame)

    def test_execute_with_custom_final_vars(self, sample_df):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor()
        result = executor.execute(inputs={"df": sample_df}, final_vars=["fill_nulls_numeric"])
        assert "fill_nulls_numeric" in result

    def test_get_mermaid_graph(self):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor()
        graph = executor.get_mermaid_graph()
        assert isinstance(graph, str)
        assert len(graph) > 0

    def test_export_mode(self, sample_df):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor(mode=PipelineMode.EXPORT)
        result = executor.run_preprocessing(sample_df)
        assert result["num"].null_count() == 0

    def test_config_passed_to_driver(self, sample_df):
        pytest.importorskip("hamilton")
        executor = PipelineExecutor(config={"some_key": "some_value"})
        assert executor.available is True


class TestPipelineExecutorFallback:
    def test_fallback_without_hamilton(self, monkeypatch):
        import iter8ml.engine.pipelines.executor as executor_mod

        def mock_import():
            return None

        monkeypatch.setattr(executor_mod, "_try_import_hamilton", mock_import)
        executor = PipelineExecutor()
        assert executor.available is False

    def test_run_preprocessing_fallback_raises_actionable_error(self, monkeypatch, sample_df):
        import iter8ml.engine.pipelines.executor as executor_mod

        monkeypatch.setattr(executor_mod, "_try_import_hamilton", lambda: None)
        executor = PipelineExecutor()
        with pytest.raises(HamiltonUnavailableError, match="uv sync --extra train"):
            executor.run_preprocessing(sample_df)

    def test_execute_fallback_raises_actionable_error(self, monkeypatch, sample_df):
        import iter8ml.engine.pipelines.executor as executor_mod

        monkeypatch.setattr(executor_mod, "_try_import_hamilton", lambda: None)
        executor = PipelineExecutor()
        with pytest.raises(HamiltonUnavailableError, match="uv sync --extra train"):
            executor.execute(inputs={"df": sample_df})

    def test_mermaid_fallback(self, monkeypatch):
        import iter8ml.engine.pipelines.executor as executor_mod

        monkeypatch.setattr(executor_mod, "_try_import_hamilton", lambda: None)
        executor = PipelineExecutor()
        graph = executor.get_mermaid_graph()
        assert "Raw Data" in graph


def test_direct_fields_sync_with_config():
    """Ensure _DIRECT_FIELDS doesn't drift from ExperimentConfig fields."""
    from iter8ml.config import _FLAT_DELEGATES, ExperimentConfig
    from iter8ml.engine.pipelines.executor import _DIRECT_FIELDS

    direct = set(_DIRECT_FIELDS)
    field_names = set(ExperimentConfig.model_fields.keys())
    delegate_names = set(_FLAT_DELEGATES.keys())
    resolvable = field_names | delegate_names

    missing = direct - resolvable
    assert not missing, f"_DIRECT_FIELDS references non-existent fields: {missing}"

    for field in direct:
        try:
            config = ExperimentConfig(
                name="test", task="classification", target_col="t", data_path="d.csv"
            )
            getattr(config, field)
        except AttributeError as e:
            raise AssertionError(
                f"_DIRECT_FIELDS entry '{field}' not accessible on ExperimentConfig: {e}"
            ) from e
