"""Contract tests: public API surface verification."""

import inspect
from pathlib import Path

import pytest

import iter8ml as iml
from iter8ml.exceptions import DataLoadError, Iter8MLError, ModelFitError, RegistryError

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

    def test_exceptions_are_iter8ml_errors(self):
        for exc_cls in [DataLoadError, ModelFitError, RegistryError]:
            assert issubclass(exc_cls, Iter8MLError), f"{exc_cls.__name__} is not an Iter8MLError"

    def test_all_exports_match_all(self):
        for name in iml.__all__:
            obj = getattr(iml, name)
            if inspect.isclass(obj):
                assert obj.__module__.startswith("iter8ml."), (
                    f"{name} module {obj.__module__} not in iter8ml"
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
        assert "workspace" in params


class TestTrainerContract:
    def test_trainer_init_signature(self):
        from iter8ml.engine.trainer import Trainer

        sig = inspect.signature(Trainer.__init__)
        params = list(sig.parameters.keys())
        assert "config" in params
        assert "workspace" in params
        assert "tracker" in params
        assert "resume_run_id" in params

    def test_trainer_run_signature(self):
        from iter8ml.engine.trainer import Trainer

        sig = inspect.signature(Trainer.run)
        assert "df" in sig.parameters


class TestEvaluatorContract:
    def test_evaluator_init_signature(self):
        from iter8ml.engine.evaluator import Evaluator

        sig = inspect.signature(Evaluator.__init__)
        assert "config" in sig.parameters

    def test_evaluate_signature(self):
        from iter8ml.engine.evaluator import Evaluator

        sig = inspect.signature(Evaluator.evaluate)
        assert "model_cls" in sig.parameters
        assert "X" in sig.parameters
        assert "y" in sig.parameters
        assert "task" in sig.parameters

    def test_compute_lift_signature(self):
        from iter8ml.engine.evaluator import Evaluator

        sig = inspect.signature(Evaluator.compute_lift)
        assert "model_scores" in sig.parameters
        assert "baseline_scores" in sig.parameters
        assert "metric_name" in sig.parameters


class TestDriftDetectorContract:
    def test_drift_detector_init_signature(self):
        from iter8ml.analysis.drift import DriftDetector

        sig = inspect.signature(DriftDetector.__init__)
        assert "reference_df" in sig.parameters
        assert "alpha" in sig.parameters

    def test_drift_detect_signature(self):
        from iter8ml.analysis.drift import DriftDetector

        sig = inspect.signature(DriftDetector.detect)
        assert "new_df" in sig.parameters

    def test_drift_report_fields(self):
        from iter8ml.analysis.drift import DriftReport

        assert "drift_detected" in DriftReport.model_fields
        assert "n_columns_tested" in DriftReport.model_fields
        assert "n_drifted" in DriftReport.model_fields
        assert "column_results" in DriftReport.model_fields


class TestPSIDriftDetectorContract:
    def test_psi_init_signature(self):
        from iter8ml.analysis.psi import PSIDriftDetector

        sig = inspect.signature(PSIDriftDetector.__init__)
        assert "reference_df" in sig.parameters
        assert "n_bins" in sig.parameters

    def test_psi_detect_signature(self):
        from iter8ml.analysis.psi import PSIDriftDetector

        sig = inspect.signature(PSIDriftDetector.detect)
        assert "live_df" in sig.parameters

    def test_psi_report_fields(self):
        from iter8ml.analysis.psi import PSIDriftReport

        assert "drift_detected" in PSIDriftReport.model_fields
        assert "n_features_tested" in PSIDriftReport.model_fields
        assert "n_moderate" in PSIDriftReport.model_fields
        assert "n_severe" in PSIDriftReport.model_fields
        assert "feature_psi" in PSIDriftReport.model_fields


class TestConfigContract:
    def test_experiment_config_init_signature(self):
        from iter8ml.config import ExperimentConfig

        assert "name" in ExperimentConfig.model_fields
        assert "task" in ExperimentConfig.model_fields
        assert "target_col" in ExperimentConfig.model_fields
        assert "data_path" in ExperimentConfig.model_fields

    def test_config_has_from_file_method(self):
        from iter8ml.config import ExperimentConfig

        assert hasattr(ExperimentConfig, "from_file")
        assert callable(ExperimentConfig.from_file)
