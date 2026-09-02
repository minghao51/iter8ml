"""Tests for IO utilities: JSONL loading + safe pickle."""

import io
import logging
import pickle
from pathlib import Path

import pytest

from iter8ml.utils.io import (
    iter_events,
    load_events,
    safe_dump,
    safe_load,
    safe_load_file,
)

# --- JSONL helpers ---


def test_load_events_empty_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.touch()
    assert load_events(path) == []


def test_load_events_nonexistent(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    assert load_events(path) == []


def test_load_events_single_event(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "test", "value": 1}\n')
    events = load_events(path)
    assert len(events) == 1
    assert events[0]["event"] == "test"


def test_load_events_multiple(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\n{"event": "b"}\n{"event": "c"}\n')
    events = load_events(path)
    assert len(events) == 3
    assert [e["event"] for e in events] == ["a", "b", "c"]


def test_load_events_skips_empty_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\n\n\n{"event": "b"}\n')
    events = load_events(path)
    assert len(events) == 2


def test_load_events_malformed_json(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\nnot json\n')
    with pytest.raises(ValueError, match="Invalid JSON at line 2"):
        load_events(path)


def test_iter_events_streaming(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\n{"event": "b"}\n')
    events = list(iter_events(path))
    assert len(events) == 2


def test_iter_events_nonexistent(tmp_path):
    events = list(iter_events(tmp_path / "nope.jsonl"))
    assert events == []


def test_iter_events_malformed(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\nbad\n')
    with pytest.raises(ValueError, match="Invalid JSON at line 2"):
        list(iter_events(path))


# --- Malformed-line policies (torn-tail recovery) ---


def test_iter_events_torn_trailing_line_skipped(tmp_path, caplog):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\n{"event": "b"}\n{"event": "trun')
    with caplog.at_level(logging.WARNING):
        events = list(iter_events(path, on_error="skip_trailing"))
    assert [e["event"] for e in events] == ["a", "b"]
    assert any("line 3" in record.getMessage() for record in caplog.records)


def test_iter_events_mid_file_corruption_still_raises_under_skip_trailing(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\nbad\n{"event": "c"}\n')
    with pytest.raises(ValueError, match="Invalid JSON at line 2"):
        list(iter_events(path, on_error="skip_trailing"))


def test_iter_events_skip_mode_drops_any_malformed_with_warning(tmp_path, caplog):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\nbad\n{"event": "c"}\n')
    with caplog.at_level(logging.WARNING):
        events = list(iter_events(path, on_error="skip"))
    assert [e["event"] for e in events] == ["a", "c"]
    assert any("line 2" in record.getMessage() for record in caplog.records)


def test_load_events_on_error_passthrough(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event": "a"}\n{"event": "trun')
    assert [e["event"] for e in load_events(path, on_error="skip_trailing")] == ["a"]
    with pytest.raises(ValueError, match="Invalid JSON at line 2"):
        load_events(path)  # default stays strict


def test_iter_events_rejects_unknown_on_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown on_error policy"):
        list(iter_events(tmp_path / "events.jsonl", on_error="yolo"))  # type: ignore[arg-type]


# --- Safe pickle ---


def test_safe_load_bytes():
    data = {"key": "value"}
    payload = pickle.dumps(data)
    result = safe_load(payload)
    assert result == data


def test_safe_load_buffered_reader():
    data = [1, 2, 3]
    payload = pickle.dumps(data)
    buf = io.BytesIO(payload)
    result = safe_load(buf)
    assert result == data


def test_safe_dump_and_load_file(tmp_path):
    obj = {"model": "catboost", "params": {"depth": 6}}
    path = str(tmp_path / "model.pkl")
    safe_dump(obj, path)
    assert Path(path).exists()
    loaded = safe_load_file(path)
    assert loaded == obj


def test_safe_dump_creates_parent_dir(tmp_path):
    obj = {"a": 1}
    path = str(tmp_path / "sub" / "nested" / "model.pkl")
    safe_dump(obj, path)
    assert Path(path).exists()


def test_restricted_unpickler_allows_sklearn():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    payload = pickle.dumps(model)
    result = safe_load(payload)
    assert isinstance(result, LogisticRegression)


class _EvilForPickleTest:
    pass


def test_restricted_unpickler_blocks_arbitrary():
    payload = pickle.dumps(_EvilForPickleTest())
    with pytest.raises(pickle.UnpicklingError, match="Blocked"):
        safe_load(payload)


def test_restricted_unpickler_allows_builtins():
    for obj in [42, "hello", [1, 2], {"a": 1}, (1, 2), {1, 2}, 3.14, True, None, b"bytes"]:
        payload = pickle.dumps(obj)
        result = safe_load(payload)
        assert result == obj or result == type(obj)(obj)


def test_restricted_unpickler_allows_numpy():
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0])
    payload = pickle.dumps(arr)
    result = safe_load(payload)
    assert np.array_equal(result, arr)


def test_safe_load_file_nonexistent(tmp_path):
    path = str(tmp_path / "nope.pkl")
    with pytest.raises(FileNotFoundError):
        safe_load_file(path)
