"""Tests for MCP server tools."""

import json
import os
from pathlib import Path

import polars as pl
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("mcp.server.fastmcp")

from iter8ml.services.mcp import (
    __getattr__,
    _get_workspace,
    detect_drift,
    export_champion,
    get_column_stats,
    get_event_log,
    get_experiment_state,
    registry_promote,
    registry_show,
    run_baseline,
    run_hpo,
)

_HAS_TABPFN_TOKEN = bool(os.environ.get("TABPFN_TOKEN"))


@pytest.fixture
def sample_csv(tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(5)})
    df = df.with_columns(target=pl.Series(y))
    path = tmp_path / "sample.csv"
    df.write_csv(str(path))
    return str(path)


@pytest.fixture
def sample_parquet(tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(5)})
    df = df.with_columns(target=pl.Series(y))
    path = tmp_path / "sample.parquet"
    df.write_parquet(str(path))
    return str(path)


def test_get_experiment_state(tmp_path, monkeypatch):
    """Test get_experiment_state in a clean environment."""

    # Change to tmp_path to avoid reading existing experiments
    monkeypatch.chdir(str(tmp_path))

    result = get_experiment_state()
    assert isinstance(result, str)
    assert "No experiments run" in result


def test_get_column_stats_csv(sample_csv):
    result = get_column_stats(sample_csv)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_column_stats_parquet(sample_parquet):
    result = get_column_stats(sample_parquet)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_column_stats_unsupported_format(tmp_path):
    bad_path = str(tmp_path / "data.json")
    Path(bad_path).write_text("{}")
    with pytest.raises(ValueError, match="Unsupported file format"):
        get_column_stats(bad_path)


@pytest.mark.skipif(
    not _HAS_TABPFN_TOKEN,
    reason="TABPFN_TOKEN not set — TabPFN needs license in CI",
)
def test_run_baseline_csv(sample_csv):
    result = run_baseline(sample_csv, "target", task="classification")
    data = json.loads(result)
    assert "catboost" in data


@pytest.mark.skipif(
    not _HAS_TABPFN_TOKEN,
    reason="TABPFN_TOKEN not set — TabPFN needs license in CI",
)
def test_run_baseline_parquet(sample_parquet):
    result = run_baseline(sample_parquet, "target", task="classification")
    data = json.loads(result)
    assert "catboost" in data


def test_run_hpo(sample_csv):
    result = run_hpo(sample_csv, "target", model="catboost", trials=2)
    data = json.loads(result)
    assert "best_params" in data
    assert "best_value" in data


def test_get_event_log_empty(tmp_path):
    import os

    orig_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        result = get_event_log()
        assert "No events" in result
    finally:
        os.chdir(orig_cwd)


def test_registry_show_empty(tmp_path):
    import os

    orig_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        result = registry_show()
        assert "empty" in result
    finally:
        os.chdir(orig_cwd)


def test_registry_promote_missing_run(tmp_path):
    import os

    orig_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        Path("workspace").mkdir()
        Path("workspace/experiments.jsonl").write_text("")
        Path("workspace/registry.json").write_text("{}")
        result = registry_promote("nonexistent_run", "test_key")
        data = json.loads(result)
        assert data["status"] == "not_found"
        assert "not found" in data["message"]
    finally:
        os.chdir(orig_cwd)


def test_registry_promote_regression_uses_r2_score(tmp_path):
    import os

    orig_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        workspace = Path("workspace")
        workspace.mkdir()
        event = {
            "event": "model_completed",
            "run_id": "run_123",
            "model": "CatBoost",
            "task": "regression",
            "cv_scores": {"rmse": 1.2, "r2": 0.81},
            "artifact_path": "workspace/artifacts/model",
        }
        Path("workspace/experiments.jsonl").write_text(json.dumps(event) + "\n")
        Path("workspace/registry.json").write_text("{}")

        result = registry_promote("run_123", "regression:test")
        data = json.loads(result)

        registry = json.loads(Path("workspace/registry.json").read_text())
        assert data["status"] == "promoted"
        assert data["selected_metric"] == "r2"
        assert data["selected_score"] == 0.81
        assert registry["regression:test"]["score"] == 0.81
    finally:
        os.chdir(orig_cwd)


def test_detect_drift(sample_parquet, tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=99)
    df_shifted = pl.DataFrame({f"feat_{i}": X[:, i] + 5 for i in range(5)})
    df_shifted = df_shifted.with_columns(target=pl.Series(y))
    new_path = str(tmp_path / "shifted.parquet")
    df_shifted.write_parquet(new_path)

    result = detect_drift(sample_parquet, new_path)
    data = json.loads(result)
    assert "drift_detected" in data
    assert "n_columns_tested" in data
    assert "n_drifted" in data


def test_run_hpo_forwards_task(sample_csv, monkeypatch):
    captured = {}

    def fake_optimize_model(model_cls, X, y, evaluator, model_name, **kwargs):
        captured["task"] = kwargs.get("task")
        return {"best_params": {}, "best_value": 0.0, "n_trials": 1}

    monkeypatch.setattr("iter8ml.engine.hpo.optimize_model", fake_optimize_model)

    result = run_hpo(sample_csv, "target", model="catboost", task="regression", trials=1)

    data = json.loads(result)
    assert data["n_trials"] == 1
    assert captured["task"] == "regression"


def test_run_hpo_uses_shared_model_factory(sample_csv, monkeypatch):
    captured = {}

    class FakeModel:
        pass

    def fake_get_model_class(model_name):
        captured["model_name"] = model_name
        return FakeModel

    def fake_optimize_model(model_cls, X, y, evaluator, model_name, **kwargs):
        captured["model_cls"] = model_cls
        return {"best_params": {}, "best_value": 0.0, "n_trials": 1}

    monkeypatch.setattr("iter8ml.services.mcp.get_model_class", fake_get_model_class)
    monkeypatch.setattr("iter8ml.engine.hpo.optimize_model", fake_optimize_model)

    result = run_hpo(sample_csv, "target", model="catboost", trials=1)

    data = json.loads(result)
    assert data["n_trials"] == 1
    assert captured["model_name"] == "catboost"
    assert captured["model_cls"] is FakeModel


def test_get_column_stats_uses_centralized_loader(monkeypatch):
    """Verify get_column_stats uses load_data from core.data.loaders."""
    from unittest.mock import patch

    with patch("iter8ml.services.mcp.load_data") as mock_load:
        mock_load.return_value = pl.DataFrame({"a": [1, 2, 3]})
        get_column_stats("test.csv")
        mock_load.assert_called_once()


def test_registry_tools_use_service(monkeypatch, tmp_path):
    """Verify registry tools use RegistryService."""
    from unittest.mock import Mock, patch

    from iter8ml.services.mcp import registry_show

    mock_registry = Mock()
    mock_registry.get_all.return_value = {"key1": {"model": "catboost"}}

    with patch("iter8ml.services.mcp.RegistryService", return_value=mock_registry):
        result = registry_show()
        assert "catboost" in result


def test_getattr_invalid_name_raises_attribute_error():
    with pytest.raises(AttributeError):
        __getattr__("not_a_real_attribute")


def test_getattr_mcp_uses_lazy_initializer(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("iter8ml.services.mcp._init_mcp", lambda: sentinel)
    assert __getattr__("mcp") is sentinel


def test_get_workspace_is_cached(monkeypatch):
    import iter8ml.services.mcp as mcp_mod

    monkeypatch.setattr(mcp_mod, "_WORKSPACE", None)
    ws1 = _get_workspace()
    ws2 = _get_workspace()
    assert ws1 is ws2


def test_get_event_log_respects_n(monkeypatch):
    import iter8ml.services.mcp as mcp_mod

    class _DummyWorkspace:
        experiments_path = Path("/tmp/fake-experiments.jsonl")

    monkeypatch.setattr(mcp_mod, "_get_workspace", lambda: _DummyWorkspace())
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(
        "iter8ml.services.mcp.load_events",
        lambda _: [{"id": 1}, {"id": 2}, {"id": 3}],
    )
    result = json.loads(get_event_log(n=2))
    assert result == [{"id": 2}, {"id": 3}]


def test_export_champion_success(monkeypatch):
    class _FakeExportService:
        def __init__(self, workspace):
            self.workspace = workspace

        def export(self, key, target_col=None):
            assert key == "clf:demo"
            assert target_col == "target"
            return Path("/tmp/pkg")

    monkeypatch.setattr("iter8ml.services.export.ExportService", _FakeExportService)
    result = json.loads(export_champion("clf:demo", target_col="target"))
    assert result["status"] == "ok"
    assert result["export_path"] == "/tmp/pkg"
    assert "predictor.py" in result["files"]


def test_export_champion_error(monkeypatch):
    class _FakeExportService:
        def __init__(self, workspace):
            self.workspace = workspace

        def export(self, key, target_col=None):
            raise RuntimeError("boom")

    monkeypatch.setattr("iter8ml.services.export.ExportService", _FakeExportService)
    result = json.loads(export_champion("clf:demo"))
    assert result["status"] == "error"
    assert "boom" in result["message"]
