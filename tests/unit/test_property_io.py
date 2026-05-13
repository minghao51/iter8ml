"""Property-based and differential tests for IO utilities."""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from iter8ml.utils.io import (
    iter_events,
    load_events,
    safe_dump,
    safe_load,
    safe_load_file,
)
from tests.strategies import (
    jsonl_events,
    picklable_object,
    whitelisted_pickle_bytes,
)

pytestmark = [pytest.mark.property, pytest.mark.differential]


class _EvilForPropertyTest:
    pass


class TestPropertyRoundTrip:
    """Property: safe_dump + safe_load_file is an identity for picklable objects."""

    @given(obj=picklable_object())
    @settings(max_examples=50)
    def test_safe_dump_load_roundtrip(self, obj):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            safe_dump(obj, path)
            loaded = safe_load_file(path)
            assert loaded == obj
        finally:
            Path(path).unlink(missing_ok=True)

    @given(obj=picklable_object())
    @settings(max_examples=30)
    def test_safe_load_equals_pickle_loads(self, obj):
        payload = pickle.dumps(obj)
        safe_result = safe_load(payload)
        pickle_result = pickle.loads(payload)
        assert safe_result == pickle_result


class TestPropertyRestrictedUnpickler:
    """Property: non-whitelisted FQNs always raise UnpicklingError."""

    def test_blocked_arbitrary_class_raises(self):
        payload = pickle.dumps(_EvilForPropertyTest())
        with pytest.raises(pickle.UnpicklingError, match="Blocked"):
            safe_load(payload)

    def test_custom_module_blocked(self):
        import io as _io

        payload = pickle.dumps(_io.BytesIO(b"test"))
        with pytest.raises(pickle.UnpicklingError, match="Blocked"):
            safe_load(payload)

    @given(data=whitelisted_pickle_bytes())
    @settings(max_examples=30)
    def test_whitelisted_classes_always_pass(self, data):
        result = safe_load(data)
        assert result is not None or data == pickle.dumps(None)


class TestPropertyJSONL:
    """Property: JSONL operations are idempotent and consistent."""

    @given(events=jsonl_events(min_events=1, max_events=20))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_load_events_preserves_order(self, events, tmp_path):
        path = tmp_path / "events.jsonl"
        lines = "\n".join(json.dumps(e) for e in events) + "\n"
        path.write_text(lines)
        loaded = load_events(path)
        assert len(loaded) == len(events)
        for orig, loaded_event in zip(events, loaded, strict=True):
            assert loaded_event["event"] == orig["event"]

    @given(events=jsonl_events(min_events=1, max_events=20))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_load_events_equals_list_iter_events(self, events, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        assert load_events(path) == list(iter_events(path))

    @given(events=jsonl_events(min_events=0, max_events=10))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_iter_events_streams_idempotent(self, events, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        first = list(iter_events(path))
        second = list(iter_events(path))
        assert first == second


class TestPropertyNumpyPickle:
    """Property: numpy arrays survive pickle round-trip through safe_load."""

    def test_numpy_array_roundtrip(self, tmp_path):
        arr = np.random.RandomState(42).randn(10, 3)
        path = str(tmp_path / "arr.pkl")
        safe_dump(arr, path)
        loaded = safe_load_file(path)
        assert np.array_equal(loaded, arr)
