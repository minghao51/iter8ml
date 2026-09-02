"""Tests for pre-warmed HPO from historical JSONL."""

import json
from pathlib import Path

import optuna
import pytest

from iter8ml.engine.hpo_warmstart import (
    WarmstartInjection,
    _build_trial_data,
    _infer_distribution,
    _parse_model_completed_events,
    create_warmstarted_study,
)


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "experiments.jsonl"
    events = [
        {
            "event": "experiment_started",
            "run_id": "exp_001",
            "model": "catboost",
        },
        {
            "event": "model_completed",
            "run_id": "exp_001",
            "model": "catboost",
            "cv_scores": {"roc_auc": 0.85},
            "params": {"depth": 6, "learning_rate": 0.05},
        },
        {
            "event": "model_completed",
            "run_id": "exp_002",
            "model": "catboost",
            "cv_scores": {"roc_auc": 0.87},
            "params": {"depth": 8, "learning_rate": 0.03},
        },
        {
            "event": "baseline_completed",
            "run_id": "exp_001",
            "model": "naive_baseline",
            "cv_scores": {"roc_auc": 0.72},
        },
        {
            "event": "model_completed",
            "run_id": "exp_003",
            "model": "lightgbm",
            "cv_scores": {"roc_auc": 0.84},
            "params": {"num_leaves": 31},
        },
        {
            "event": "model_completed",
            "run_id": "exp_004",
            "model": "catboost",
            "cv_scores": {},
            "params": {"depth": 7},
        },
    ]
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


class TestParseModelCompletedEvents:
    def test_yields_only_model_completed(self, sample_jsonl):
        from iter8ml.utils.io import load_events

        events = load_events(sample_jsonl)
        catboost_events = list(_parse_model_completed_events(events, "catboost"))
        assert len(catboost_events) == 3

    def test_filters_by_model_name(self, sample_jsonl):
        from iter8ml.utils.io import load_events

        events = load_events(sample_jsonl)
        lgb_events = list(_parse_model_completed_events(events, "lightgbm"))
        assert len(lgb_events) == 1
        assert lgb_events[0]["run_id"] == "exp_003"

    def test_empty_when_no_matching_model(self, sample_jsonl):
        from iter8ml.utils.io import load_events

        events = load_events(sample_jsonl)
        xgb_events = list(_parse_model_completed_events(events, "xgboost"))
        assert len(xgb_events) == 0

    def test_includes_hpo_trial_completed_events(self):
        events = [
            {
                "event": "hpo_trial_completed",
                "model": "catboost",
                "cv_scores": {"roc_auc": 0.81},
                "params": {"depth": 6},
            },
            {"event": "model_completed", "model": "lightgbm"},
        ]
        parsed = list(_parse_model_completed_events(events, "catboost"))
        assert len(parsed) == 1
        assert parsed[0]["event"] == "hpo_trial_completed"


class TestInferDistribution:
    def test_boolean_returns_categorical(self):
        dist = _infer_distribution("use_gpu", True)
        assert isinstance(dist, optuna.distributions.CategoricalDistribution)

    def test_int_n_estimators_returns_int_distribution(self):
        dist = _infer_distribution("n_estimators", 500)
        assert isinstance(dist, optuna.distributions.IntDistribution)

    def test_int_depth_returns_int_distribution(self):
        dist = _infer_distribution("max_depth", 6)
        assert isinstance(dist, optuna.distributions.IntDistribution)

    def test_float_lr_returns_log_float_distribution(self):
        dist = _infer_distribution("learning_rate", 0.05)
        assert isinstance(dist, optuna.distributions.FloatDistribution)
        assert dist.log is True

    def test_float_dropout_returns_uniform_float_distribution(self):
        dist = _infer_distribution("dropout", 0.1)
        assert isinstance(dist, optuna.distributions.FloatDistribution)
        assert dist.log is False

    def test_float_subsample_returns_uniform_float_distribution(self):
        dist = _infer_distribution("subsample", 0.8)
        assert isinstance(dist, optuna.distributions.FloatDistribution)
        assert dist.log is False

    def test_fallback_for_unknown_float(self):
        dist = _infer_distribution("unknown_float_param", 1.5)
        assert isinstance(dist, optuna.distributions.FloatDistribution)

    def test_fallback_for_unknown_int(self):
        dist = _infer_distribution("unknown_int_param", 42)
        assert isinstance(dist, optuna.distributions.IntDistribution)


class TestBuildTrialData:
    def test_creates_params_and_distributions(self):
        params = {"depth": 6, "learning_rate": 0.05}
        out_params, distributions = _build_trial_data(params, 0.85)
        assert out_params == params
        assert "depth" in distributions
        assert "learning_rate" in distributions


class TestCreateWarmstartedStudy:
    def test_returns_study_and_injection_metadata(self, sample_jsonl):
        study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="maximize",
            log_path=str(sample_jsonl),
            n_trials=50,
            pruner="median",
        )
        assert isinstance(study, optuna.Study)
        assert isinstance(injection, WarmstartInjection)
        assert injection.model_name == "catboost"
        assert injection.n_runs_scanned == 3

    def test_injects_completed_trials(self, sample_jsonl):
        study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="maximize",
            log_path=str(sample_jsonl),
            primary_metric="roc_auc",
        )
        assert injection.n_trials_injected == 2
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed_trials) == 2

    def test_filters_out_empty_cv_scores(self, sample_jsonl):
        _study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="maximize",
            log_path=str(sample_jsonl),
            primary_metric="roc_auc",
        )
        assert injection.n_trials_injected == 2
        assert injection.n_skipped_missing_scores == 1

    def test_skips_malformed_params(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        bad_event = {
            "event": "hpo_trial_completed",
            "model": "catboost",
            "cv_scores": {"roc_auc": 0.8},
            "params": "not-a-dict",
        }
        path.write_text(json.dumps(bad_event) + "\n")

        _study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="maximize",
            log_path=str(path),
            primary_metric="roc_auc",
        )
        assert injection.n_trials_injected == 0
        assert injection.n_skipped_missing_params == 1

    def test_counts_invalid_trial_payloads(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        bad_event = {
            "event": "hpo_trial_completed",
            "model": "catboost",
            "cv_scores": {"roc_auc": 0.8},
            "params": {"depth": None},
        }
        path.write_text(json.dumps(bad_event) + "\n")

        _study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="maximize",
            log_path=str(path),
            primary_metric="roc_auc",
        )
        assert injection.n_trials_injected == 0
        assert injection.n_skipped_invalid_trials == 1

    def test_does_not_include_wrong_model(self, sample_jsonl):
        _study, injection = create_warmstarted_study(
            model_name="xgboost",
            direction="maximize",
            log_path=str(sample_jsonl),
        )
        assert injection.n_trials_injected == 0
        assert injection.n_runs_scanned == 0

    def test_nop_pruner_when_unknown_pruner(self, sample_jsonl):
        study, _ = create_warmstarted_study(
            model_name="catboost",
            pruner="unknown",
            log_path=str(sample_jsonl),
        )
        assert isinstance(study.pruner, optuna.pruners.NopPruner)

    def test_median_pruner(self, sample_jsonl):
        study, _ = create_warmstarted_study(
            model_name="catboost",
            pruner="median",
            log_path=str(sample_jsonl),
        )
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)


class TestMetricCompatibility:
    """Direction safety: inject only values scored on the current primary metric."""

    @staticmethod
    def _write_log(tmp_path: Path, events: list[dict]) -> Path:
        path = tmp_path / "events.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        return path

    def test_injects_primary_metric_value_not_first_metric(self, tmp_path: Path):
        """Event scored [r2, rmse] with primary rmse: the rmse value is injected."""
        path = self._write_log(
            tmp_path,
            [
                {
                    "event": "model_completed",
                    "run_id": "r1",
                    "model": "catboost",
                    "cv_scores": {"r2": 0.9, "rmse": 2.0},
                    "params": {"depth": 6},
                }
            ],
        )
        study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="minimize",
            log_path=str(path),
            primary_metric="rmse",
        )
        assert injection.n_trials_injected == 1
        assert injection.n_skipped_metric_mismatch == 0
        # The maximize-oriented r2 score (0.9) must NOT be injected into the
        # minimize study — the injected value is the rmse score.
        assert study.trials[0].value == 2.0
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_skips_events_not_scored_on_primary_metric(self, tmp_path: Path):
        """Event lacks the current primary metric entirely: skip and count it."""
        path = self._write_log(
            tmp_path,
            [
                {
                    "event": "model_completed",
                    "run_id": "r1",
                    "model": "catboost",
                    "cv_scores": {"r2": 0.9, "mae": 1.0},
                    "params": {"depth": 6},
                }
            ],
        )
        study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="minimize",
            log_path=str(path),
            primary_metric="rmse",
        )
        assert injection.n_trials_injected == 0
        assert injection.n_skipped_metric_mismatch == 1
        assert len(study.trials) == 0
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_unknown_primary_metric_skips_all_injection(self, tmp_path: Path):
        """Without a known primary metric, event metric identity is unverifiable."""
        path = self._write_log(
            tmp_path,
            [
                {
                    "event": "model_completed",
                    "run_id": "r1",
                    "model": "catboost",
                    "cv_scores": {"roc_auc": 0.85},
                    "params": {"depth": 6},
                }
            ],
        )
        _study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="maximize",
            log_path=str(path),
            primary_metric=None,
        )
        assert injection.n_trials_injected == 0
        assert injection.n_skipped_metric_mismatch == 1

    def test_compatible_metric_still_warms(self, tmp_path: Path):
        path = self._write_log(
            tmp_path,
            [
                {
                    "event": "model_completed",
                    "run_id": "r1",
                    "model": "catboost",
                    "cv_scores": {"rmse": 2.0},
                    "params": {"depth": 6},
                }
            ],
        )
        study, injection = create_warmstarted_study(
            model_name="catboost",
            direction="minimize",
            log_path=str(path),
            primary_metric="rmse",
        )
        assert injection.n_trials_injected == 1
        assert study.trials[0].value == 2.0
