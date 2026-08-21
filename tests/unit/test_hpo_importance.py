"""Tests for PedAnova hyperparameter importance analysis."""

import optuna
import pytest
from pydantic import ValidationError

from iter8ml.engine.hpo_importance import (
    ImportanceReport,
    ParamImportance,
    compute_param_importance,
)


@pytest.fixture
def study_with_trials():
    study = optuna.create_study(direction="maximize")

    search_space = {
        "depth": (2, 10),
        "learning_rate": (0.01, 0.3, "log"),
        "subsample": (0.5, 1.0),
    }

    for _ in range(10):
        study.optimize(
            lambda t: (
                (
                    t.suggest_int("depth", 2, 10)
                    + t.suggest_float("learning_rate", 0.01, 0.3, log=True)
                    + t.suggest_float("subsample", 0.5, 1.0)
                )
                / 3
            ),
            n_trials=1,
        )

    return study, search_space


class TestComputeParamImportance:
    def test_returns_importance_report(self, study_with_trials):
        study, _ = study_with_trials
        report = compute_param_importance(study)
        assert isinstance(report, ImportanceReport)
        assert report.model_name is not None
        assert report.n_trials == 10
        assert len(report.importances) > 0

    def test_importances_sorted_by_importance(self, study_with_trials):
        study, _ = study_with_trials
        report = compute_param_importance(study)
        importances = [p.importance for p in report.importances]
        assert importances == sorted(importances, reverse=True)

    def test_each_param_has_name_and_score(self, study_with_trials):
        study, _ = study_with_trials
        report = compute_param_importance(study)
        for p in report.importances:
            assert isinstance(p.param_name, str)
            assert isinstance(p.importance, float)
            assert p.importance >= 0.0

    def test_evaluator_name_in_report(self, study_with_trials):
        study, _ = study_with_trials
        report = compute_param_importance(study)
        assert "PedAnova" in report.evaluator

    def test_timestamp_is_iso_format(self, study_with_trials):
        study, _ = study_with_trials
        report = compute_param_importance(study)
        assert "T" in report.timestamp

    def test_empty_study_returns_empty_importances(self):
        study = optuna.create_study(direction="maximize")
        report = compute_param_importance(study)
        assert len(report.importances) == 0


class TestParamImportanceDataclass:
    def test_frozen(self):
        p = ParamImportance(param_name="depth", importance=0.5)
        with pytest.raises(ValidationError, match="Instance is frozen"):
            p.importance = 0.6

    def test_fields(self):
        p = ParamImportance(param_name="lr", importance=0.42)
        assert p.param_name == "lr"
        assert p.importance == 0.42


class TestImportanceReportDataclass:
    def test_frozen(self):
        report = ImportanceReport(
            model_name="test",
            n_trials=5,
            importances=[],
            evaluator="PedAnovaImportanceEvaluator",
            timestamp="2025-01-01T00:00:00",
        )
        with pytest.raises(ValidationError, match="Instance is frozen"):
            report.n_trials = 10
