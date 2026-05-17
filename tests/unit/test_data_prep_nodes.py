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
