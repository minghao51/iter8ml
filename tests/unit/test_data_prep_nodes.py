import numpy as np
import polars as pl
import pytest

BASE_CONFIG = {
    "run_quality_audit": False,
    "auto_clean_noise": False,
    "noise_quality_threshold": 0.5,
    "run_leakage_audit": False,
    "target_transform": "none",
    "target_skewness_threshold": 1.0,
}

BASE_INPUTS = {
    "target_col": "target",
    "task": "classification",
    "leakage_n_jobs": 1,
}


def _build_driver(config_overrides: dict | None = None):
    from hamilton import driver

    from iter8ml.engine.pipelines.nodes import prep

    cfg = {**BASE_CONFIG, **(config_overrides or {})}
    return driver.Builder().with_modules(prep).with_config(cfg).build()


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": pl.Series(["x", "y", "x", "y", "x"], dtype=pl.Categorical),
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def dr():
    pytest.importorskip("hamilton")
    return _build_driver()


class TestValidateTarget:
    def test_valid_target(self, sample_df):
        dr = _build_driver()
        result = dr.execute(
            ["validate_target"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        assert result["validate_target"] is not None
        assert "target" in result["validate_target"].columns
        assert len(result["validate_target"]) == 5

    def test_invalid_target(self, sample_df):
        dr = _build_driver()
        inputs = {**BASE_INPUTS, "df": sample_df, "target_col": "missing_col"}
        with pytest.raises(ValueError, match="target_col"):
            dr.execute(["validate_target"], inputs=inputs)


class TestAdapterResult:
    def test_produces_numpy_arrays(self, sample_df):
        dr = _build_driver()
        result = dr.execute(
            ["adapter_result"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        X, y = result["adapter_result"]
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 5
        assert len(y) == 5

    def test_excludes_target_from_X(self, sample_df):
        dr = _build_driver()
        result = dr.execute(
            ["adapter_result"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        X, _ = result["adapter_result"]
        assert X.shape[1] == 3


class TestFeatureNames:
    def test_excludes_target(self, sample_df):
        dr = _build_driver()
        result = dr.execute(
            ["feature_names"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        names = result["feature_names"]
        assert "target" not in names
        assert "num_a" in names
        assert "num_b" in names
        assert "cat" in names


class TestLeakageReport:
    def test_skipped_when_disabled(self, sample_df):
        dr = _build_driver({"run_leakage_audit": False})
        result = dr.execute(
            ["leakage_report"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        assert result["leakage_report"] is None

    def test_run_when_enabled(self, sample_df):
        dr = _build_driver({"run_leakage_audit": True})
        result = dr.execute(
            ["leakage_report"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        report = result["leakage_report"]
        assert report is not None
        assert hasattr(report, "n_flagged")
        assert hasattr(report, "baseline_score")


class TestTargetTransform:
    def test_none_passthrough(self, sample_df):
        dr = _build_driver({"target_transform": "none"})
        result = dr.execute(
            ["target_transform_result"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        _, _, method, _, _, applied = result["target_transform_result"]
        assert method == "none"
        assert applied is False

    def test_log1p_transform(self, sample_df):
        dr = _build_driver({"target_transform": "log1p"})
        result = dr.execute(
            ["target_transform_result"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        _, transformer, method, _, _, applied = result["target_transform_result"]
        assert method == "log1p"
        assert applied is True
        assert transformer is not None


class TestDataPrepResult:
    def test_full_pipeline(self, sample_df):
        dr = _build_driver()
        result = dr.execute(
            ["data_prep_result"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        prep = result["data_prep_result"]
        assert isinstance(prep.X, np.ndarray)
        assert isinstance(prep.y, np.ndarray)
        assert prep.n_rows == 5
        assert prep.n_features == 3
        assert len(prep.feature_names) == 3
        assert prep.target_transform_method == "none"
        assert prep.noise_cleaned is False

    def test_with_leakage(self, sample_df):
        dr = _build_driver({"run_leakage_audit": True})
        result = dr.execute(
            ["data_prep_result"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        prep = result["data_prep_result"]
        assert prep.leakage_report is not None

    def test_with_target_transform(self, sample_df):
        dr = _build_driver({"target_transform": "log1p"})
        result = dr.execute(
            ["data_prep_result"],
            inputs={"df": sample_df, **BASE_INPUTS},
        )
        prep = result["data_prep_result"]
        assert prep.target_transform_applied is True
        assert prep.target_transformer is not None


class TestTargetOrientation:
    """positive_class maps the positive class to 1 (explicit roc_auc orientation)."""

    def test_none_passthrough(self):
        from iter8ml.engine.pipelines.nodes.prep import target_oriented_df

        df = pl.DataFrame({"t": ["bad", "good"]})
        assert target_oriented_df(df, "t").equals(df)

    def test_positive_class_maps_to_one(self):
        from iter8ml.engine.pipelines.nodes.prep import target_oriented_df

        df = pl.DataFrame({"t": ["bad", "good", "bad"]})
        out = target_oriented_df(df, "t", positive_class="bad")
        assert out["t"].to_list() == [1, 0, 1]

    def test_orientation_is_value_based_not_order_based(self):
        """Mapping keys off the VALUE, not appearance order (which is arbitrary
        and subject to Polars' global string cache — don't assert its codes)."""
        from iter8ml.engine.pipelines.nodes.prep import target_oriented_df

        df = pl.DataFrame({"t": ["b_first", "a_second", "b_first"]})
        oriented = target_oriented_df(df, "t", positive_class="a_second")["t"].to_list()
        assert oriented == [0, 1, 0]

    def test_numeric_positive_class(self):
        from iter8ml.engine.pipelines.nodes.prep import target_oriented_df

        df = pl.DataFrame({"t": [0, 1, 1]})
        out = target_oriented_df(df, "t", positive_class=1)
        assert out["t"].to_list() == [0, 1, 1]

    def test_unknown_positive_class_raises_with_observed_values(self):
        from iter8ml.engine.pipelines.nodes.prep import target_oriented_df

        df = pl.DataFrame({"t": ["bad", "good"]})
        with pytest.raises(ValueError, match=r"not found in target column 't'.*Observed"):
            target_oriented_df(df, "t", positive_class="excellent")

    def test_multiclass_target_raises(self):
        from iter8ml.engine.pipelines.nodes.prep import target_oriented_df

        df = pl.DataFrame({"t": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="binary target"):
            target_oriented_df(df, "t", positive_class="a")


class TestPositiveClassDagWiring:
    """BLOCKER regression: end-to-end prep with positive_class on a string target."""

    def test_string_target_with_positive_class_survives_prep(self):
        from iter8ml.config import ExperimentConfig
        from iter8ml.constants import TaskType
        from iter8ml.engine.pipelines.executor import (
            PipelineExecutor,
            PipelineMode,
            _resolve_hamilton_config,
        )

        df = pl.DataFrame(
            {
                "num": [float(i) for i in range(20)],
                "target": ["bad", "good"] * 10,
            }
        )
        cfg = ExperimentConfig(
            name="t",
            task=TaskType.CLASSIFICATION,
            target_col="target",
            data_path="unused.csv",
            positive_class="good",
            metrics=["roc_auc"],
        )
        executor = PipelineExecutor(mode=PipelineMode.HPO, config=_resolve_hamilton_config(cfg))
        out = executor.run_prep(cfg, df)
        # Oriented target: "good" (positive) → 1, "bad" → 0; and the integer
        # target did NOT re-enter the categorical cast (which used to raise).
        assert out["target"].to_list() == [0, 1] * 10

    def test_without_positive_class_legacy_codes_unchanged(self):
        from iter8ml.config import ExperimentConfig
        from iter8ml.constants import TaskType
        from iter8ml.engine.pipelines.executor import (
            PipelineExecutor,
            PipelineMode,
            _resolve_hamilton_config,
        )

        df = pl.DataFrame(
            {
                "num": [float(i) for i in range(20)],
                "target": ["bad", "good"] * 10,
            }
        )
        cfg = ExperimentConfig(
            name="t",
            task=TaskType.CLASSIFICATION,
            target_col="target",
            data_path="unused.csv",
            metrics=["roc_auc"],
        )
        executor = PipelineExecutor(mode=PipelineMode.HPO, config=_resolve_hamilton_config(cfg))
        out = executor.run_prep(cfg, df)
        codes = out["target"].unique().to_list()
        # Appearance-order physical codes (values may be offset by Polars'
        # global string cache when the suite runs together) — still two classes.
        assert len(codes) == 2
        assert all(isinstance(c, int) for c in codes)
