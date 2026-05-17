import numpy as np
import pytest

from iter8ml.engine.pipelines.nodes.train import (
    ModelResult,
    _effective_training_workers,
    _train_one,
    models_to_run,
    training_state,
)
from iter8ml.workspace import Workspace


class MockDataPrepResult:
    def __init__(self, n_rows=100):
        self.n_rows = n_rows
        self.X = np.random.rand(n_rows, 5)
        self.y = np.random.randint(0, 2, n_rows)
        self.feature_names = [f"f{i}" for i in range(5)]


class TestModelSelection:
    def test_auto_returns_list(self):
        result = models_to_run(
            data_prep_result=MockDataPrepResult(100),
            task="classification",
            vram_gb=0.0,
            config_models="auto",
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_auto_includes_baselines(self):
        result = models_to_run(
            data_prep_result=MockDataPrepResult(100),
            task="classification",
            vram_gb=0.0,
            config_models="auto",
            include_baselines=True,
        )
        assert "naive_baseline" in result
        assert "linear_baseline" in result

    def test_auto_excludes_baselines(self):
        result = models_to_run(
            data_prep_result=MockDataPrepResult(100),
            task="classification",
            vram_gb=0.0,
            config_models="auto",
            include_baselines=False,
        )
        assert "naive_baseline" not in result

    def test_explicit_model_list(self):
        result = models_to_run(
            data_prep_result=MockDataPrepResult(100),
            task="classification",
            vram_gb=0.0,
            config_models=["lightgbm", "xgboost"],
        )
        assert result == ["lightgbm", "xgboost"]


class TestModelResult:
    def test_dataclass_fields(self):
        r = ModelResult(
            model_name="test",
            input_name="test",
            cv_scores={"accuracy": 0.9},
            artifact_path="/tmp/test",
            duration_seconds=1.5,
        )
        assert r.model_name == "test"
        assert r.input_name == "test"
        assert r.error is None
        assert r.lift_over_baselines is None

    def test_error_result(self):
        r = ModelResult(
            model_name="fail",
            input_name="fail",
            cv_scores={},
            artifact_path="",
            duration_seconds=0.1,
            error="boom",
        )
        assert r.error == "boom"


class TestStateGeneration:
    def _make_results(self):
        return [
            ModelResult(
                model_name="Model A",
                input_name="model_a",
                cv_scores={"roc_auc": 0.85, "f1_macro": 0.80},
                artifact_path="/tmp/a",
                duration_seconds=1.0,
            ),
            ModelResult(
                model_name="Model B",
                input_name="model_b",
                cv_scores={"roc_auc": 0.90, "f1_macro": 0.88},
                artifact_path="/tmp/b",
                duration_seconds=2.0,
            ),
        ]

    def test_training_state_picks_best(self, tmp_path):
        state = training_state(
            training_results=self._make_results(),
            baseline_scores={},
            metrics=["roc_auc", "f1_macro"],
            run_id="test_run",
            experiment_name="exp",
            task="classification",
            workspace=Workspace(root=tmp_path),
        )
        assert state.best_model == "model_b"
        assert state.best_score == 0.90
        assert state.best_metric == "roc_auc"
        assert len(state.leaderboard) == 2
        assert "model_a" in state.results
        assert "model_b" in state.results

    def test_training_state_with_errors(self, tmp_path):
        results = self._make_results()
        results.append(
            ModelResult(
                model_name="fail_model",
                input_name="fail_model",
                cv_scores={},
                artifact_path="",
                duration_seconds=0.1,
                error="something broke",
            )
        )
        state = training_state(
            training_results=results,
            baseline_scores={},
            metrics=["roc_auc"],
            run_id="test_run",
            experiment_name="exp",
            task="classification",
            workspace=Workspace(root=tmp_path),
        )
        assert "fail_model" in state.results
        assert "error" in state.results["fail_model"]

    def test_training_state_with_baselines(self, tmp_path):
        baselines = {
            "naive_baseline": {"roc_auc": 0.5},
            "linear_baseline": {"roc_auc": 0.7},
        }
        state = training_state(
            training_results=self._make_results(),
            baseline_scores=baselines,
            metrics=["roc_auc"],
            run_id="test_run",
            experiment_name="exp",
            task="classification",
            workspace=Workspace(root=tmp_path),
        )
        assert "naive_baseline" in state.results
        assert state.results["naive_baseline"]["is_baseline"] is True

    def test_empty_results(self, tmp_path):
        state = training_state(
            training_results=[],
            baseline_scores={},
            metrics=["roc_auc"],
            run_id="test_run",
            experiment_name="exp",
            task="classification",
            workspace=Workspace(root=tmp_path),
        )
        assert state.best_model is None
        assert state.best_score is None


class _OverrideAwareModel:
    def __init__(self, task="classification"):
        self.task = task
        self.params = {}

    @property
    def model_name(self):
        return "OverrideAwareModel"

    def apply_overrides(self, overrides):
        allowed = {"depth", "learning_rate"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unsupported override keys: {sorted(unknown)}")
        self.params.update(overrides)

    def fit(self, X, y):
        return None

    def save(self, path):
        return None


class TestModelOverrides:
    @pytest.fixture(autouse=True)
    def _patch_evaluator(self, monkeypatch):
        from iter8ml.engine.pipelines.nodes import train

        monkeypatch.setattr(
            train,
            "_evaluate_model",
            lambda *args, **kwargs: {"roc_auc": 0.9},
        )

    @pytest.fixture(autouse=True)
    def _patch_model_factory(self, monkeypatch):
        monkeypatch.setattr(
            "iter8ml.engine.models.factory.get_model_class",
            lambda _name: _OverrideAwareModel,
        )

    def test_training_applies_model_overrides(self, tmp_path):
        result = _train_one(
            name="catboost",
            X=np.random.rand(20, 4),
            y=np.random.randint(0, 2, 20),
            task="classification",
            cv_folds=3,
            cv_strategy="stratified",
            metrics=["roc_auc"],
            calibration="none",
            workspace=Workspace(root=tmp_path),
            run_id="override_ok",
            baseline_scores={},
            model_overrides={"catboost": {"depth": 8}},
        )
        assert result.error is None
        assert result.params == {"depth": 8}

    def test_training_returns_explicit_error_on_invalid_override(self, tmp_path):
        result = _train_one(
            name="catboost",
            X=np.random.rand(20, 4),
            y=np.random.randint(0, 2, 20),
            task="classification",
            cv_folds=3,
            cv_strategy="stratified",
            metrics=["roc_auc"],
            calibration="none",
            workspace=Workspace(root=tmp_path),
            run_id="override_bad",
            baseline_scores={},
            model_overrides={"catboost": {"bad_key": 8}},
        )
        assert result.error is not None
        assert "Unsupported override keys" in result.error


class TestWorkerSizing:
    def test_caps_parallelism_for_gbdt_models(self):
        workers = _effective_training_workers(4, ["catboost", "tabnet"])
        assert workers == 1

    def test_respects_requested_workers_for_non_gbdt(self):
        workers = _effective_training_workers(4, ["tabnet", "ft_transformer"])
        assert workers == 2

    def test_can_opt_out_of_strict_thread_safety(self):
        workers = _effective_training_workers(
            4,
            ["catboost", "tabnet"],
            strict_thread_safety=False,
        )
        assert workers == 2
