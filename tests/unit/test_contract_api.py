"""Contract tests: public API surface verification."""

import inspect
from pathlib import Path

import pytest

import iter8ml as iml
from iter8ml.exceptions import DataLoadError, ModelFitError, RegistryError, TabularBlueprintError

pytestmark = pytest.mark.contract


class TestPublicExports:
    def test_all_exports_importable(self):
        for name in iml.__all__:
            obj = getattr(iml, name, None)
            assert obj is not None, f"{name} not found in iter8ml"

    def test_all_exports_are_classes_or_functions(self):
        for name in iml.__all__:
            obj = getattr(iml, name)
            assert inspect.isclass(obj) or inspect.isfunction(obj), (
                f"{name} is {type(obj)}, not a class or function"
            )

    def test_exceptions_are_tabular_blueprint_errors(self):
        for exc_cls in [DataLoadError, ModelFitError, RegistryError]:
            assert issubclass(exc_cls, TabularBlueprintError), (
                f"{exc_cls.__name__} is not a TabularBlueprintError"
            )


class TestWorkspacePathContract:
    def test_all_path_properties_return_path(self):
        from iter8ml.workspace import Workspace

        ws = Workspace(root="/tmp/test_ws")
        for attr in [
            "experiments_path",
            "registry_path",
            "artifacts_dir",
            "exports_dir",
            "state_path",
            "leaderboard_path",
        ]:
            val = getattr(ws, attr)
            assert isinstance(val, Path), f"{attr} is {type(val)}, not Path"


class TestTrackerProtocolContract:
    def test_tracker_protocol_methods_match(self):
        from iter8ml.engine.tracker import JSONLTracker, MLflowTracker, WandbTracker

        required = {"log_metrics", "log_params", "log_artifact", "log_event", "finish"}
        for cls in [JSONLTracker, WandbTracker, MLflowTracker]:
            methods = {
                name
                for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
                if not name.startswith("_")
            }
            assert required.issubset(methods), f"{cls.__name__} missing: {required - methods}"

    def test_tracker_protocol_has_run_id_attribute(self):
        import typing

        from iter8ml.engine.tracker import Tracker

        attrs = typing.get_type_hints(Tracker)
        assert "current_run_id" in attrs, "Tracker protocol missing current_run_id"


class TestSessionSignatureContract:
    def test_session_run_signature(self):
        from iter8ml.session import ExperimentSession

        sig = inspect.signature(ExperimentSession.run)
        assert "config" in sig.parameters
        assert "df" in sig.parameters
        assert "resume_run_id" in sig.parameters

    def test_session_init_signature(self):
        from iter8ml.session import ExperimentSession

        sig = inspect.signature(ExperimentSession.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "workspace" in params or "root" in params
