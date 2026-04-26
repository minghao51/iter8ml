"""Tests for ModelSelector routing logic."""

from unittest.mock import patch

from tabular_blueprint.models.selector import ModelSelector


def test_small_dataset_no_gpu_routing():
    selector = ModelSelector()
    models = selector.select(n_rows=5000, task="classification", vram_gb=0.0)
    assert models[:2] == ["naive_baseline", "linear_baseline"]
    assert "catboost" in models
    assert "lightgbm" in models


def test_small_dataset_with_gpu_routing():
    selector = ModelSelector()
    with patch.object(selector, "_has_gpu", return_value=True):
        models = selector.select(n_rows=5000, task="classification", vram_gb=0.0)
        assert models[:2] == ["naive_baseline", "linear_baseline"]
        assert models[2:] == ["tabpfn", "catboost", "lightgbm", "xgboost"]


def test_medium_dataset_routing():
    selector = ModelSelector()
    models = selector.select(n_rows=100_000, task="classification", vram_gb=0.0)
    assert models[2:] == ["catboost", "lightgbm", "xgboost"]


def test_large_dataset_routing():
    selector = ModelSelector()
    models = selector.select(n_rows=600_000, task="classification", vram_gb=0.0)
    assert models[2:] == ["lightgbm", "xgboost"]


def test_no_baselines_when_disabled():
    selector = ModelSelector()
    with patch.object(selector, "_has_gpu", return_value=True):
        models = selector.select(
            n_rows=5000, task="classification", vram_gb=0.0, include_baselines=False
        )
        assert models == ["tabpfn", "catboost", "lightgbm", "xgboost"]


def test_gpu_routing_ft_transformer():
    selector = ModelSelector()
    models = selector.select(n_rows=100_000, task="classification", vram_gb=16.0)
    assert "ft_transformer" in models
    assert "catboost" in models
    assert "lightgbm" in models
    assert "xgboost" in models


def test_ft_transformer_with_gpu():
    selector = ModelSelector()
    models = selector.select(n_rows=100_000, task="classification", vram_gb=16.0)
    assert "ft_transformer" in models


def test_no_ft_transformer_without_enough_vram():
    selector = ModelSelector()
    models = selector.select(n_rows=100_000, task="classification", vram_gb=8.0)
    assert "ft_transformer" not in models


def test_no_ft_transformer_with_small_dataset():
    selector = ModelSelector()
    models = selector.select(n_rows=10_000, task="classification", vram_gb=16.0)
    assert "ft_transformer" not in models


def test_boundary_50k_rows_no_gpu():
    selector = ModelSelector()
    models = selector.select(n_rows=50_000, task="classification", vram_gb=0.0)
    assert "tabpfn" not in models
    assert "catboost" in models


def test_boundary_500k_rows():
    selector = ModelSelector()
    models = selector.select(n_rows=500_000, task="classification", vram_gb=0.0)
    assert "catboost" not in models
    assert "lightgbm" in models
    assert "xgboost" in models


def test_tabnet_added_with_enough_vram():
    selector = ModelSelector()
    models = selector.select(n_rows=100_000, task="classification", vram_gb=10.0)
    assert "tabnet" in models
