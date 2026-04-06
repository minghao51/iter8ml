"""Tests for ModelSelector routing logic."""

from core.models.selector import ModelSelector


def test_small_dataset_routing():
    selector = ModelSelector()
    models = selector.select(n_rows=5000, task="classification", vram_gb=0.0)
    assert models == ["tabpfn", "catboost", "lightgbm"]


def test_medium_dataset_routing():
    selector = ModelSelector()
    models = selector.select(n_rows=100_000, task="classification", vram_gb=0.0)
    assert models == ["catboost", "lightgbm", "xgboost"]


def test_large_dataset_routing():
    selector = ModelSelector()
    models = selector.select(n_rows=600_000, task="classification", vram_gb=0.0)
    assert models == ["lightgbm", "xgboost"]


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


def test_boundary_10k_rows():
    selector = ModelSelector()
    models = selector.select(n_rows=10_000, task="classification", vram_gb=0.0)
    assert "tabpfn" not in models
    assert "catboost" in models


def test_boundary_500k_rows():
    selector = ModelSelector()
    models = selector.select(n_rows=500_000, task="classification", vram_gb=0.0)
    assert "catboost" not in models
    assert "lightgbm" in models
    assert "xgboost" in models
