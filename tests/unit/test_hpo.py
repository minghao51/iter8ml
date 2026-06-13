"""Tests for HPO _validate_bounds and create_study edge cases."""

import optuna
import pytest

from iter8ml.engine.hpo import _validate_bounds, create_study


class TestValidateBounds:
    def test_valid_int_bounds(self):
        _validate_bounds("n_estimators", 10, 500)

    def test_valid_float_bounds(self):
        _validate_bounds("lr", 0.001, 0.1)

    def test_rejects_non_numeric_low(self):
        with pytest.raises(ValueError, match="bounds must be numeric"):
            _validate_bounds("p", "low", 1.0)

    def test_rejects_non_numeric_high(self):
        with pytest.raises(ValueError, match="bounds must be numeric"):
            _validate_bounds("p", 0.0, None)

    def test_rejects_low_geq_high(self):
        with pytest.raises(ValueError, match="lower bound"):
            _validate_bounds("p", 10, 10)

    def test_rejects_low_gt_high(self):
        with pytest.raises(ValueError, match="lower bound"):
            _validate_bounds("p", 100, 1)


class TestCreateStudy:
    def test_creates_maximize_study(self):
        study = create_study("test", direction="maximize")
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_creates_minimize_study(self):
        study = create_study("test", direction="minimize")
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_default_pruner_is_median(self):
        study = create_study("test", pruner="median")
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)
