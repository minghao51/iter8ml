"""Shared hypothesis strategy generators for property-based tests."""

import json
import pickle

import numpy as np
import polars as pl
from hypothesis import strategies as st


@st.composite
def dataframes(draw, min_rows=1, max_rows=50, min_cols=1, max_cols=8):
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    cols = {}
    for i in range(n_cols):
        dtype = draw(st.sampled_from(["float", "int", "str", "null", "bool"]))
        if dtype == "float":
            cols[f"f_{i}"] = draw(
                st.lists(
                    st.floats(allow_nan=False, allow_infinity=False),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        elif dtype == "int":
            cols[f"f_{i}"] = draw(
                st.lists(
                    st.integers(min_value=-1000, max_value=1000), min_size=n_rows, max_size=n_rows
                )
            )
        elif dtype == "str":
            cols[f"f_{i}"] = draw(st.lists(st.text(max_size=5), min_size=n_rows, max_size=n_rows))
        elif dtype == "bool":
            cols[f"f_{i}"] = draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))
        else:
            cols[f"f_{i}"] = [None] * n_rows
    return pl.DataFrame(cols)


@st.composite
def polars_with_all_dtypes(draw, min_rows=1, max_rows=30):
    n_rows = draw(st.integers(min_rows, max_rows))
    cols = {
        "float_col": draw(
            st.lists(
                st.floats(allow_nan=True, allow_infinity=True), min_size=n_rows, max_size=n_rows
            )
        ),
        "int_col": draw(
            st.lists(st.integers(min_value=-1000, max_value=1000), min_size=n_rows, max_size=n_rows)
        ),
        "str_col": draw(st.lists(st.text(max_size=5), min_size=n_rows, max_size=n_rows)),
        "bool_col": draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows)),
        "null_col": [None] * n_rows,
    }
    return pl.DataFrame(cols)


@st.composite
def numpy_arrays(draw, min_rows=5, max_rows=100, min_cols=1, max_cols=10):
    n = draw(st.integers(min_rows, max_rows))
    m = draw(st.integers(min_cols, max_cols))
    return draw(
        st.lists(
            st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=m, max_size=m),
            min_size=n,
            max_size=n,
        ).map(lambda rows: np.array(rows))
    )


@st.composite
def classification_problem(draw, min_rows=20, max_rows=200):
    n = draw(st.integers(min_rows, max_rows))
    m = draw(st.integers(1, 6))
    X = draw(
        st.lists(
            st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=m, max_size=m),
            min_size=n,
            max_size=n,
        ).map(lambda rows: np.array(rows))
    )
    y = draw(st.lists(st.integers(0, 1), min_size=n, max_size=n).map(lambda v: np.array(v)))
    return X, y


@st.composite
def multiclass_problem(draw, min_rows=30, max_rows=200, max_classes=5):
    n = draw(st.integers(min_rows, max_rows))
    m = draw(st.integers(1, 6))
    n_classes = draw(st.integers(3, max_classes))
    X = draw(
        st.lists(
            st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=m, max_size=m),
            min_size=n,
            max_size=n,
        ).map(lambda rows: np.array(rows))
    )
    y = draw(
        st.lists(st.integers(0, n_classes - 1), min_size=n, max_size=n).map(lambda v: np.array(v))
    )
    return X, y


@st.composite
def regression_problem(draw, min_rows=20, max_rows=200):
    n = draw(st.integers(min_rows, max_rows))
    m = draw(st.integers(1, 6))
    X = draw(
        st.lists(
            st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=m, max_size=m),
            min_size=n,
            max_size=n,
        ).map(lambda rows: np.array(rows))
    )
    coeffs = draw(st.lists(st.floats(-2, 2), min_size=m, max_size=m).map(lambda v: np.array(v)))
    noise = draw(st.floats(0.01, 1.0))
    y = X @ coeffs + np.random.RandomState(42).randn(n) * noise
    return X, y


@st.composite
def jsonl_events(draw, min_events=0, max_events=10):
    n = draw(st.integers(min_events, max_events))
    events = []
    for _ in range(n):
        event = {
            "event": draw(st.sampled_from(["metrics", "params", "artifact", "run_completed"])),
            "run_id": draw(st.uuids()).hex,
        }
        if draw(st.booleans()):
            event["timestamp"] = draw(st.datetimes()).isoformat()
        events.append(event)
    return events


@st.composite
def picklable_object(draw):
    return draw(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-10000, max_value=10000),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=20),
            st.lists(st.integers(min_value=-100, max_value=100), max_size=5),
            st.dictionaries(
                st.text(max_size=5),
                st.integers(min_value=-100, max_value=100),
                max_size=3,
            ),
            st.tuples(st.integers(), st.text(max_size=5)),
        )
    )


def jsonl_bytes(events: list[dict]) -> bytes:
    return "\n".join(json.dumps(e) for e in events).encode()


@st.composite
def whitelisted_pickle_bytes(draw):
    obj = draw(picklable_object())
    return pickle.dumps(obj)


@st.composite
def config_dict(draw):
    return (
        st.fixed_dictionaries(
            {
                "name": st.text(min_size=1, max_size=20),
                "task": st.sampled_from(["classification", "regression"]),
                "target_col": st.text(min_size=1, max_size=10),
                "data_path": st.text(min_size=1, max_size=50),
            }
        )
        .and_then(
            lambda base: st.fixed_dictionaries(
                {
                    **base,
                    "cv_folds": st.integers(2, 10),
                    "random_seed": st.integers(0, 1000),
                    "data_sample": st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
                }
            )
        )()
        .filter(lambda d: isinstance(d, dict))
    )
