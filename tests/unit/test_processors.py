import polars as pl
import pytest

from tabular_blueprint.pipelines.nodes import preprocessing


@pytest.fixture
def dr():
    pytest.importorskip("hamilton")
    from hamilton import driver

    return driver.Builder().with_modules(preprocessing).build()


def test_fill_nulls_numeric(dr):
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    result = dr.execute(["fill_nulls_numeric"], inputs={"df": df})
    out_df = result["fill_nulls_numeric"]
    assert out_df["a"].null_count() == 0
    assert out_df["b"].null_count() == 0
    assert out_df["a"].to_list() == [1.0, 2.0, 3.0]


def test_fill_nulls_categorical(dr):
    df = pl.DataFrame({"cat": pl.Series(["a", "a", "b", None], dtype=pl.Categorical)})
    result = dr.execute(["fill_nulls_categorical"], inputs={"df": df})
    out_df = result["fill_nulls_categorical"]
    assert out_df["cat"].null_count() == 0


def test_fill_nulls_string_columns(dr):
    df = pl.DataFrame({"cat": ["a", "a", "b", None]})
    result = dr.execute(["categorical_columns"], inputs={"df": df})
    assert "cat" in result["categorical_columns"]


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


def test_decompose_dates_drops_original(dr):
    df = pl.DataFrame(
        {
            "event_date": pl.Series(["2023-01-15", "2023-06-20"]).str.to_datetime(),
        }
    )
    result = dr.execute(["decomposed_dates_df"], inputs={"df": df})
    out_df = result["decomposed_dates_df"]
    assert "event_date" not in out_df.columns


def test_null_filled_df_merges_columns(dr):
    df = pl.DataFrame(
        {
            "num": [1.0, None, 3.0, 4.0],
            "cat": pl.Series(["x", "y", "x", None], dtype=pl.Categorical),
        }
    )
    result = dr.execute(["null_filled_df"], inputs={"df": df})
    out_df = result["null_filled_df"]
    assert out_df["num"].null_count() == 0
    assert out_df["cat"].null_count() == 0
    assert set(out_df.columns) == {"num", "cat"}


def test_full_pipeline(dr):
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
    assert "date" not in out_df.columns
    assert out_df["cat"].dtype.is_integer()


def test_full_pipeline_with_strings(dr):
    df = pl.DataFrame(
        {
            "num": [1.0, None, 3.0, 4.0],
            "cat": ["x", "y", "x", None],
            "date": pl.Series(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ).str.to_datetime(),
        }
    )
    result = dr.execute(["processed_dataframe"], inputs={"df": df})
    out_df = result["processed_dataframe"]

    assert out_df["num"].null_count() == 0
    assert out_df["cat"].null_count() == 0
    assert out_df["cat"].dtype.is_integer()
    assert "date_year" in out_df.columns


def test_no_dates_passthrough(dr):
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "cat": pl.Series(["x", "y", "z"], dtype=pl.Categorical),
        }
    )
    result = dr.execute(["processed_dataframe"], inputs={"df": df})
    out_df = result["processed_dataframe"]
    assert "a" in out_df.columns
    assert "cat" in out_df.columns
