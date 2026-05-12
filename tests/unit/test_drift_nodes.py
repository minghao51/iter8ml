import numpy as np
import polars as pl
import pytest

from iter8ml.engine.pipelines.nodes.drift_detection import (
    DriftReport,
    live_features,
    reference_features,
)


@pytest.fixture
def ref_df():
    rng = np.random.default_rng(42)
    return pl.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(3)})


@pytest.fixture
def live_df():
    rng = np.random.default_rng(99)
    return pl.DataFrame({f"f{i}": rng.normal(3, 1, 200) for i in range(3)})


class TestDriftNodes:
    def test_reference_features_selects_numeric(self, ref_df):
        result = reference_features(ref_df)
        assert result.shape == (200, 3)
        assert all(result[c].dtype.is_numeric() for c in result.columns)

    def test_live_features_selects_numeric(self, live_df):
        result = live_features(live_df)
        assert result.shape == (200, 3)

    def test_features_excludes_non_numeric(self):
        df = pl.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"], "val": [3.0, 4.0]})
        result = reference_features(df)
        assert "cat" not in result.columns
        assert "num" in result.columns
        assert "val" in result.columns


class TestDriftReport:
    def test_dataclass(self):
        report = DriftReport(drift_detected=True, psi_report=None, domain_report=None)
        assert report.drift_detected is True
        assert report.psi_report is None


class TestDriftDAGIntegration:
    def test_psi_drift_via_executor(self, ref_df, live_df):
        pytest.importorskip("hamilton")
        from iter8ml.engine.pipelines.executor import PipelineExecutor

        executor = PipelineExecutor()
        report = executor.run_drift(ref_df, live_df, drift_method="psi")
        assert report is not None
        assert isinstance(report.drift_detected, bool)
        assert report.psi_report is not None

    def test_domain_drift_via_executor(self, ref_df, live_df):
        pytest.importorskip("hamilton")
        from iter8ml.engine.pipelines.executor import PipelineExecutor

        executor = PipelineExecutor()
        report = executor.run_drift(ref_df, live_df, drift_method="domain_classifier")
        assert report is not None
        assert isinstance(report.drift_detected, bool)
        assert report.domain_report is not None

    def test_both_drift_via_executor(self, ref_df, live_df):
        pytest.importorskip("hamilton")
        from iter8ml.engine.pipelines.executor import PipelineExecutor

        executor = PipelineExecutor()
        report = executor.run_drift(ref_df, live_df, drift_method="both")
        assert report is not None
        assert report.psi_report is not None
        assert report.domain_report is not None

    def test_detects_shifted_data(self, ref_df, live_df):
        pytest.importorskip("hamilton")
        from iter8ml.engine.pipelines.executor import PipelineExecutor

        executor = PipelineExecutor()
        report = executor.run_drift(ref_df, live_df, drift_method="psi")
        assert report.drift_detected is True

    def test_no_drift_identical_data(self, ref_df):
        pytest.importorskip("hamilton")
        from iter8ml.engine.pipelines.executor import PipelineExecutor

        executor = PipelineExecutor()
        report = executor.run_drift(ref_df, ref_df, drift_method="psi")
        assert report.drift_detected is False
