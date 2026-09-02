"""Tests for the shared model factory."""

import pytest

from iter8ml.engine.models.factory import get_model_class, validate_model_name


def test_get_model_class_known_model():
    cls = get_model_class("catboost")
    assert cls.__name__ == "CatBoostModel"


def test_get_model_class_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_class("not_a_model")


def test_get_model_class_reuses_cached_class():
    first = get_model_class("catboost")
    second = get_model_class("catboost")
    assert first is second


def test_get_model_class_caps_omp_threads(monkeypatch):
    """get_model_class must apply the OMP cap before GBDT imports (ADR-0004/0006)."""
    from iter8ml.config import HardwareProfile

    calls: list[int] = []

    def _recorder(*args: object, **kwargs: object) -> int:
        calls.append(1)
        return 8

    monkeypatch.setattr(HardwareProfile, "configure_omp_threads", _recorder)

    cls = get_model_class("lightgbm")

    assert cls.__name__ == "LightGBMModel"
    assert calls == [1]


def test_validate_model_name_returns_name():
    assert validate_model_name("catboost") == "catboost"
