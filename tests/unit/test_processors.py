"""Tests for feature engineering processors."""

import polars as pl

from core.data.processors import (
    cast_types,
    decompose_dates,
    encode_categoricals,
    fill_nulls,
    pipeline,
)


def test_fill_nulls_median():
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    result = fill_nulls(df, numeric_strategy="median")
    assert result["a"].null_count() == 0
    assert result["b"].null_count() == 0
    assert result["a"].to_list() == [1.0, 2.0, 3.0]


def test_fill_nulls_mean():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    result = fill_nulls(df, numeric_strategy="mean")
    assert result["a"].to_list() == [1.0, 2.0, 3.0]


def test_fill_nulls_zero():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    result = fill_nulls(df, numeric_strategy="zero")
    assert result["a"].to_list() == [1.0, 0.0, 3.0]


def test_fill_nulls_categorical_mode():
    df = pl.DataFrame({"cat": pl.Series(["a", "a", "b", None], dtype=pl.Categorical)})
    result = fill_nulls(df, categorical_strategy="mode")
    assert result["cat"].null_count() == 0


def test_fill_nulls_categorical_unknown():
    df = pl.DataFrame({"cat": pl.Series(["a", None, "b", None], dtype=pl.Categorical)})
    result = fill_nulls(df, categorical_strategy="unknown")
    assert result["cat"].null_count() == 0
    assert "unknown" in result["cat"].to_list()


def test_fill_nulls_no_nulls():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = fill_nulls(df)
    assert result["a"].to_list() == [1.0, 2.0, 3.0]


def test_decompose_dates():
    df = pl.DataFrame(
        {
            "event_date": pl.Series(["2023-01-15", "2023-06-20"]).str.to_datetime(),
        }
    )
    result = decompose_dates(df)
    assert "event_year" in result.columns
    assert "event_month" in result.columns
    assert "event_day" in result.columns
    assert "event_day_of_week" in result.columns


def test_decompose_dates_custom_cols():
    df = pl.DataFrame(
        {
            "my_date": pl.Series(["2023-01-15"]).str.to_datetime(),
        }
    )
    result = decompose_dates(df, date_cols=["my_date"])
    assert "my_year" in result.columns
    assert "my_month" in result.columns


def test_decompose_dates_no_date_cols():
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = decompose_dates(df)
    assert result.columns == ["a"]


def test_encode_categoricals_label():
    df = pl.DataFrame({"cat": pl.Series(["a", "b", "c"], dtype=pl.Categorical)})
    result = encode_categoricals(df, strategy="label")
    assert result["cat"].dtype.is_integer()
    assert set(result["cat"].to_list()) == {0, 1, 2}


def test_encode_categoricals_no_cat_cols():
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = encode_categoricals(df)
    assert result.columns == ["a"]


def test_encode_categoricals_unknown_strategy():
    df = pl.DataFrame({"cat": pl.Series(["a", "b"], dtype=pl.Categorical)})
    result = encode_categoricals(df, strategy="onehot")
    assert result["cat"].dtype == pl.Categorical


def test_cast_types():
    df = pl.DataFrame({"a": ["1", "2", "3"]})
    result = cast_types(df, {"a": pl.Int64})
    assert result["a"].dtype == pl.Int64


def test_pipeline_all_steps():
    df = pl.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "cat": pl.Series(["x", "y", "z"], dtype=pl.Categorical),
        }
    )
    result = pipeline(
        df,
        do_fill_nulls=True,
        do_decompose_dates=False,
        do_encode_categoricals=True,
    )
    assert result["a"].null_count() == 0
    assert result["cat"].dtype.is_integer()


def test_pipeline_skip_all():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    result = pipeline(
        df,
        do_fill_nulls=False,
        do_decompose_dates=False,
        do_encode_categoricals=False,
    )
    assert result["a"].null_count() == 1
