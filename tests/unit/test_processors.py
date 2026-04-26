"""Tests for Hamilton-based preprocessing nodes."""

import polars as pl
import pytest

from tabular_blueprint.pipelines import preprocessing


@pytest.fixture
def dr():
    pytest.importorskip("hamilton")
    from hamilton import driver

    return driver.Builder().with_modules(preprocessing).build()


def test_fill_nulls_numeric(dr):
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    # We want to test fill_nulls_numeric, but we need to provide the 'df' for raw_dataframe
    result = dr.execute(["fill_nulls_numeric"], inputs={"df": df})
    out_df = result["fill_nulls_numeric"]
    assert out_df["a"].null_count() == 0
    assert out_df["b"].null_count() == 0
    assert out_df["a"].to_list() == [1.0, 2.0, 3.0]


def test_fill_nulls_categorical(dr):
    df = pl.DataFrame({"cat": pl.Series(["a", "a", "b", None], dtype=pl.Categorical)})
    # Test through the chain
    result = dr.execute(["fill_nulls_categorical"], inputs={"df": df})
    out_df = result["fill_nulls_categorical"]
    assert out_df["cat"].null_count() == 0


def test_decompose_dates(dr):
    df = pl.DataFrame(
        {
            "event_date": pl.Series(["2023-01-15", "2023-06-20"]).str.to_datetime(),
        }
    )
    result = dr.execute(["decomposed_dates_df"], inputs={"df": df})
    out_df = result["decomposed_dates_df"]
    assert "event_year" in out_df.columns
    assert "event_month" in out_df.columns


def test_full_pipeline(dr):
    # Fix shape error: all columns must have 4 rows
    df = pl.DataFrame(
        {
            "a": [1.0, None, 3.0, 4.0],
            "cat": pl.Series(["x", "y", "x", None], dtype=pl.Categorical),
            "date": pl.Series(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ).str.to_datetime(),
        }
    )
    result = dr.execute(["processed_dataframe"], inputs={"df": df})
    out_df = result["processed_dataframe"]

    assert out_df["a"].null_count() == 0
    assert out_df["cat"].null_count() == 0
    assert "date_year" in out_df.columns
    assert out_df["cat"].dtype.is_integer()
