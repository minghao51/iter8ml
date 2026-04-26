"""Tests for the shared model factory."""

import pytest

from tabular_blueprint.models.factory import get_model_class, validate_model_name


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


def test_validate_model_name_returns_name():
    assert validate_model_name("catboost") == "catboost"
