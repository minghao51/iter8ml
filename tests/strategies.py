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
        dtype = draw(st.sampled_from(["float", "int", "str", "null"]))
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
        else:
            cols[f"f_{i}"] = [None] * n_rows
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
        )
    )


def jsonl_bytes(events: list[dict]) -> bytes:
    return "\n".join(json.dumps(e) for e in events).encode()


@st.composite
def whitelisted_pickle_bytes(draw):
    obj = draw(picklable_object())
    return pickle.dumps(obj)


@st.composite
def blocked_pickle_bytes(draw):
    class _Arbitrary:
        def __init__(self, x):
            self.x = x

    obj = _Arbitrary(draw(st.integers()))
    return pickle.dumps(obj)
