"""Thread-pinning defaults for GBDT wrappers (ADR-0004/0006, hybrid-CPU safety).

LightGBM/XGBoost/CatBoost fall back to their own core detection when their
thread parameter is unset; these tests pin the default to the OMP cap while
keeping user overrides authoritative.
"""

from __future__ import annotations

import os

import pytest

from iter8ml.config import HardwareProfile
from iter8ml.engine.models.catboost_model import CatBoostModel
from iter8ml.engine.models.lightgbm_model import LightGBMModel
from iter8ml.engine.models.xgboost_model import XGBoostModel


@pytest.mark.parametrize(
    ("model_cls", "param_name"),
    [
        (LightGBMModel, "num_threads"),
        (XGBoostModel, "nthread"),
        (CatBoostModel, "thread_count"),
    ],
)
def test_gbdt_params_default_thread_count(model_cls, param_name):
    model = model_cls(task="classification")
    params = model._build_params()
    assert params[param_name] == HardwareProfile.configure_omp_threads()
    assert int(params[param_name]) >= 1


@pytest.mark.parametrize(
    ("model_cls", "param_name"),
    [
        (LightGBMModel, "num_threads"),
        (XGBoostModel, "nthread"),
        (CatBoostModel, "thread_count"),
    ],
)
def test_gbdt_params_user_override_wins(model_cls, param_name):
    model = model_cls(task="classification", **{param_name: 3})
    params = model._build_params()
    assert params[param_name] == 3


def test_configure_omp_threads_sets_passive_wait_policy(monkeypatch):
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    HardwareProfile.configure_omp_threads()
    assert os.environ.get("OMP_WAIT_POLICY") == "passive"
