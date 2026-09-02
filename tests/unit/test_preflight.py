"""Tests for pre-run preflight checks (verification/preflight.py)."""

import numpy as np
import polars as pl

from iter8ml.config import ExperimentConfig
from iter8ml.verification.preflight import IssueLevel, has_errors, run_preflight


def make_config(task: str = "classification", **overrides) -> ExperimentConfig:
    return ExperimentConfig(
        name="t",
        task=task,  # type: ignore[arg-type]
        target_col="target",
        data_path="unused.csv",
        **overrides,
    )


def clean_classification_df(n: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": pl.Series((rng.normal(size=n) > 0).astype(int)),
        }
    )


class TestTargetChecks:
    def test_missing_target_is_error(self):
        df = clean_classification_df().drop("target")
        issues = run_preflight(make_config(), df)
        assert has_errors(issues)
        assert any("not found" in i.message for i in issues)

    def test_null_target_is_error(self):
        df = clean_classification_df()
        df = df.with_columns(
            pl.when(pl.arange(0, df.height) == 0)
            .then(None)
            .otherwise(pl.col("target"))
            .alias("target")
        )
        issues = run_preflight(make_config(), df)
        assert any(i.level == IssueLevel.ERROR and i.check == "target" for i in issues)

    def test_constant_target_is_error(self):
        df = clean_classification_df()
        df = df.with_columns(pl.lit(1).alias("target"))
        issues = run_preflight(make_config(), df)
        assert has_errors(issues)
        assert any("constant" in i.message for i in issues)

    def test_regression_with_string_target_is_error(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "target": ["x", "y", "z"]})
        issues = run_preflight(make_config(task="regression"), df)
        assert has_errors(issues)

    def test_classification_with_many_float_values_warns_regression(self):
        df = pl.DataFrame(
            {
                "a": [float(i) for i in range(30)],
                "target": [float(i) for i in range(30)],
            }
        )
        config = make_config(task="classification", cv_strategy="kfold")
        issues = run_preflight(config, df)
        assert any(i.level == IssueLevel.WARNING and "regression" in i.message for i in issues)
        assert not has_errors(issues)


class TestCVFeasibility:
    def test_too_few_rows_for_folds_is_error(self):
        df = clean_classification_df(6)
        issues = run_preflight(make_config(cv_folds=5), df)
        assert any(i.level == IssueLevel.ERROR and i.check == "cv" for i in issues)

    def test_stratified_folds_exceeding_rarest_class_is_error(self):
        df = pl.DataFrame(
            {
                "a": list(range(30)),
                "target": [1] * 28 + [0, 0],
            }
        )
        issues = run_preflight(make_config(cv_folds=5, cv_strategy="stratified"), df)
        assert any(i.level == IssueLevel.ERROR and "rarest class" in i.message for i in issues)

    def test_timeseries_without_date_column_warns(self):
        df = clean_classification_df()
        config = make_config(cv_strategy="timeseries")
        issues = run_preflight(config, df)
        assert any(i.level == IssueLevel.WARNING and i.check == "cv" for i in issues)


class TestColumnChecks:
    def test_unknown_ignore_col_is_error(self):
        df = clean_classification_df()
        issues = run_preflight(make_config(ignore_cols=["nope"]), df)
        assert has_errors(issues)
        assert any(i.check == "ignore_cols" for i in issues)

    def test_target_in_ignore_cols_is_error(self):
        df = clean_classification_df()
        issues = run_preflight(make_config(ignore_cols=["target"]), df)
        assert has_errors(issues)

    def test_id_like_unique_column_warns_leakage(self):
        n = 50
        df = clean_classification_df(n).with_columns(pl.arange(0, n).alias("customer_id"))
        issues = run_preflight(make_config(), df)
        assert any(
            i.level == IssueLevel.WARNING and i.check == "leakage" and "customer_id" in i.message
            for i in issues
        )

    def test_all_null_column_warns(self):
        df = clean_classification_df().with_columns(pl.lit(None).alias("empty"))
        issues = run_preflight(make_config(), df)
        assert any("entirely null" in i.message for i in issues)


class TestCleanData:
    def test_clean_data_passes_without_errors(self):
        issues = run_preflight(make_config(), clean_classification_df())
        assert not has_errors(issues)

    def test_regression_clean_data_passes(self):
        rng = np.random.default_rng(0)
        df = pl.DataFrame(
            {
                "a": rng.normal(size=100),
                "target": rng.normal(size=100),
            }
        )
        issues = run_preflight(make_config(task="regression"), df)
        assert not has_errors(issues)


class TestPositiveClassChecks:
    def test_positive_class_present_is_noted_not_error(self):
        config = make_config(positive_class=1)
        df = clean_classification_df()
        issues = run_preflight(config, df)
        assert not has_errors(issues)
        notes = [i for i in issues if "positive_class" in i.message]
        assert notes
        assert notes[0].level == IssueLevel.WARNING

    def test_positive_class_absent_is_error(self):
        config = make_config(positive_class="never_there")
        df = clean_classification_df()
        issues = run_preflight(config, df)
        errors = [
            i for i in issues if i.level == IssueLevel.ERROR and "positive_class" in i.message
        ]
        assert errors
        assert "Observed values" in errors[0].message

    def test_regression_config_rejects_positive_class_at_parse(self):
        import pytest

        with pytest.raises(ValueError, match="only valid for classification"):
            make_config(task="regression", positive_class=1)


class TestPositiveClassChecksMulticlass:
    def test_multiclass_target_with_positive_class_is_error(self):
        config = make_config(positive_class="a")
        df = clean_classification_df().with_columns(
            pl.Series("target", (["a", "b", "c"] * 34)[:100])
        )
        issues = run_preflight(config, df)
        errors = [i for i in issues if i.level == IssueLevel.ERROR and "binary target" in i.message]
        assert errors
