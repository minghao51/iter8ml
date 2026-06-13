"""Tests for custom exception hierarchy."""

import pytest

from iter8ml.exceptions import (
    DataLoadError,
    Iter8MLError,
    ModelFitError,
    RegistryError,
    TabularBlueprintError,
    track_errors,
)


class DummyService:
    @track_errors(ModelFitError)
    def good_method(self):
        return 42

    @track_errors(DataLoadError)
    def value_error_method(self):
        raise ValueError("file not found: data.csv")

    @track_errors(ModelFitError)
    def runtime_error_method(self):
        raise RuntimeError("model crashed")

    @track_errors(ModelFitError)
    def generic_error_method(self):
        raise TypeError("unexpected")

    @track_errors(ModelFitError)
    def already_typed_error(self):
        raise ModelFitError("already typed")


@track_errors(DataLoadError)
def standalone_bad():
    raise FileNotFoundError("missing.csv")


@track_errors(RegistryError)
def standalone_ok():
    return "ok"


def test_exception_hierarchy():
    assert issubclass(DataLoadError, Iter8MLError)
    assert issubclass(ModelFitError, Iter8MLError)
    assert issubclass(RegistryError, Iter8MLError)
    assert TabularBlueprintError is Iter8MLError


def test_exception_context():
    e = DataLoadError("test", context={"file": "data.csv"})
    assert e.context == {"file": "data.csv"}
    assert str(e) == "test"


def test_track_errors_success():
    svc = DummyService()
    assert svc.good_method() == 42


def test_track_errors_wraps_as_specified_type():
    svc = DummyService()
    with pytest.raises(DataLoadError, match="file not found"):
        svc.value_error_method()


def test_track_errors_wraps_runtime_as_model_fit():
    svc = DummyService()
    with pytest.raises(ModelFitError, match="model crashed"):
        svc.runtime_error_method()


def test_track_errors_wraps_generic():
    svc = DummyService()
    with pytest.raises(ModelFitError, match="unexpected"):
        svc.generic_error_method()


def test_track_errors_already_typed_passes_through():
    svc = DummyService()
    with pytest.raises(ModelFitError, match="already typed"):
        svc.already_typed_error()


def test_track_errors_preserves_cause():
    svc = DummyService()
    with pytest.raises(ModelFitError) as exc_info:
        svc.runtime_error_method()
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "model crashed"


def test_track_errors_context_includes_original_type():
    svc = DummyService()
    with pytest.raises(ModelFitError) as exc_info:
        svc.generic_error_method()
    assert exc_info.value.context["original_type"] == "TypeError"


def test_track_errors_standalone_function():
    with pytest.raises(DataLoadError, match=r"missing\.csv"):
        standalone_bad()


def test_track_errors_standalone_success():
    assert standalone_ok() == "ok"
