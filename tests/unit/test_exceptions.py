"""Tests for custom exception hierarchy."""

import pytest

from tabular_blueprint.exceptions import (
    DataLoadError,
    ModelFitError,
    RegistryError,
    TabularBlueprintError,
    track_errors,
)


class DummyTracker:
    def __init__(self):
        self.events = []

    def log_event(self, event):
        self.events.append(event)


class DummyTrainer:
    def __init__(self):
        self.tracker = DummyTracker()

    @track_errors()
    def good_method(self):
        return 42

    @track_errors()
    def value_error_method(self):
        raise ValueError("bad data")

    @track_errors()
    def runtime_error_method(self):
        raise RuntimeError("model crashed")

    @track_errors()
    def generic_error_method(self):
        raise TypeError("unexpected")

    @track_errors()
    def already_typed_error(self):
        raise ModelFitError("already typed")


def test_exception_hierarchy():
    assert issubclass(DataLoadError, TabularBlueprintError)
    assert issubclass(ModelFitError, TabularBlueprintError)
    assert issubclass(RegistryError, TabularBlueprintError)


def test_exception_context():
    e = DataLoadError("test", context={"file": "data.csv"})
    assert e.context == {"file": "data.csv"}
    assert str(e) == "test"


def test_track_errors_success():
    t = DummyTrainer()
    result = t.good_method()
    assert result == 42
    assert len(t.tracker.events) == 0


def test_track_errors_value_error():
    t = DummyTrainer()
    with pytest.raises(DataLoadError, match="bad data"):
        t.value_error_method()
    assert len(t.tracker.events) == 1
    assert t.tracker.events[0]["error_type"] == "DataLoadError"


def test_track_errors_runtime_error():
    t = DummyTrainer()
    with pytest.raises(ModelFitError, match="model crashed"):
        t.runtime_error_method()
    assert len(t.tracker.events) == 1
    assert t.tracker.events[0]["error_type"] == "ModelFitError"


def test_track_errors_generic():
    t = DummyTrainer()
    with pytest.raises(ModelFitError, match="unexpected"):
        t.generic_error_method()
    assert len(t.tracker.events) == 1


def test_track_errors_already_typed():
    t = DummyTrainer()
    with pytest.raises(ModelFitError, match="already typed"):
        t.already_typed_error()
    assert len(t.tracker.events) == 0
