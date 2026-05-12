"""Tests for CLI commands using typer CliRunner."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest
from sklearn.datasets import make_classification
from typer.testing import CliRunner

from iter8ml.cli import app

runner = CliRunner()


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


def test_init_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = os.getcwd()
        os.chdir(tmpdir)
        try:
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert "Workspace initialized" in result.stdout
            assert Path(tmpdir, "workspace").exists()
            assert Path(tmpdir, "workspace", "artifacts").exists()
            assert Path(tmpdir, "workspace", "experiments.jsonl").exists()
            assert Path(tmpdir, "workspace", "registry.json").exists()
        finally:
            os.chdir(orig)


def test_init_preserves_existing_registry_without_force_reset():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = os.getcwd()
        os.chdir(tmpdir)
        try:
            workspace = Path(tmpdir, "workspace")
            workspace.mkdir(parents=True, exist_ok=True)
            registry_path = workspace / "registry.json"
            registry_path.write_text('{"existing":"champion"}')

            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert registry_path.read_text() == '{"existing":"champion"}'
        finally:
            os.chdir(orig)


def test_init_force_reset_registry_overwrites_existing_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = os.getcwd()
        os.chdir(tmpdir)
        try:
            workspace = Path(tmpdir, "workspace")
            workspace.mkdir(parents=True, exist_ok=True)
            registry_path = workspace / "registry.json"
            registry_path.write_text('{"existing":"champion"}')

            result = runner.invoke(app, ["init", "--force-reset-registry"])
            assert result.exit_code == 0
            assert registry_path.read_text() == "{}"
        finally:
            os.chdir(orig)


def test_init_with_data(sample_csv):
    result = runner.invoke(app, ["init", "--data", sample_csv])
    assert result.exit_code == 0
    assert "Data path set to" in result.stdout


def test_hardware_command():
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Hardware Profile" in result.stdout
    assert "RAM" in result.stdout
    assert "CPU Cores" in result.stdout


def test_leaderboard_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = os.getcwd()
        os.chdir(tmpdir)
        try:
            result = runner.invoke(app, ["leaderboard"])
            assert result.exit_code == 0
            assert "No experiments" in result.stdout
        finally:
            os.chdir(orig)


def test_registry_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = os.getcwd()
        os.chdir(tmpdir)
        try:
            result = runner.invoke(app, ["registry", "show"])
            assert result.exit_code == 0
            assert "Registry is empty" in result.stdout
        finally:
            os.chdir(orig)


def test_state_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = os.getcwd()
        os.chdir(tmpdir)
        try:
            result = runner.invoke(app, ["state"])
            assert result.exit_code == 0
            assert "No experiments run" in result.stdout
        finally:
            os.chdir(orig)


def test_run_missing_data():
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_run_unsupported_format():
    result = runner.invoke(app, ["run", "--data", "data.json"])
    assert result.exit_code == 1
    assert "Unsupported file format" in result.stdout


def test_run_invalid_config_path_exits_with_error(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--config", str(tmp_path / "missing.py"), "--data", "data.csv"],
    )
    assert result.exit_code == 1
    assert "config file not found" in result.stdout


def test_run_non_python_config_exits_with_error(tmp_path):
    config_path = tmp_path / "config.txt"
    config_path.write_text("not python")

    result = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--allow-unsafe-config", "--data", "data.csv"],
    )
    assert result.exit_code == 1
    assert "Unsupported config format" in result.stdout


def test_run_config_missing_config_object_exits_with_error(tmp_path):
    config_path = tmp_path / "config.py"
    config_path.write_text("x = 1\n")

    result = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--allow-unsafe-config", "--data", "data.csv"],
    )
    assert result.exit_code == 1
    assert "must define `config`" in result.stdout


def test_run_python_config_rejected_without_unsafe_flag(tmp_path):
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "from iter8ml.config import ExperimentConfig\n"
        "from iter8ml.constants import TaskType\n"
        "config = ExperimentConfig("
        "name='x', task=TaskType.CLASSIFICATION, target_col='target', data_path='d.csv')\n"
    )
    result = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--data", "data.csv"],
    )
    assert result.exit_code == 1
    assert "disabled by default for safety" in result.stdout


def test_run_python_config_allowed_with_unsafe_flag(tmp_path):
    config_path = tmp_path / "config.py"
    data_path = tmp_path / "data.txt"
    data_path.write_text("x")
    config_path.write_text(
        "from iter8ml.config import ExperimentConfig\n"
        "from iter8ml.constants import TaskType\n"
        f"config = ExperimentConfig("
        f"name='x', task=TaskType.CLASSIFICATION, target_col='target', data_path='{data_path}')\n"
    )
    result = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--allow-unsafe-config"],
    )
    assert result.exit_code == 1
    assert "Unsupported file format: .txt" in result.stdout


@pytest.mark.slow
def test_run_with_csv(sample_csv):
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            sample_csv,
            "--target",
            "target",
            "--models",
            "catboost",
        ],
    )
    assert result.exit_code == 0
    assert "Loaded 100 rows" in result.stdout
    assert "Results" in result.stdout
    assert "catboost" in result.stdout.lower()


@pytest.mark.slow
def test_run_with_parquet(sample_parquet):
    result = runner.invoke(
        app,
        [
            "run",
            "--data",
            sample_parquet,
            "--target",
            "target",
            "--models",
            "catboost",
        ],
    )
    assert result.exit_code == 0
    assert "Loaded 100 rows" in result.stdout


@pytest.mark.slow
def test_leaderboard_after_run(sample_csv, tmp_path):
    orig = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        runner.invoke(app, ["init"])
        runner.invoke(
            app,
            [
                "run",
                "--data",
                sample_csv,
                "--target",
                "target",
                "--models",
                "catboost",
            ],
        )

        result = runner.invoke(app, ["leaderboard"])
        assert result.exit_code == 0
        assert "Leaderboard" in result.stdout
        assert "CatBoost" in result.stdout
        assert "Primary Metric" in result.stdout
    finally:
        os.chdir(orig)


def test_drift_detection(sample_parquet, tmp_path):
    X, y = make_classification(n_samples=100, n_features=5, random_state=99)
    df_shifted = pl.DataFrame({f"feat_{i}": X[:, i] + 5 for i in range(5)})
    df_shifted = df_shifted.with_columns(target=pl.Series(y))
    new_path = tmp_path / "shifted.parquet"
    df_shifted.write_parquet(str(new_path))

    result = runner.invoke(
        app,
        [
            "drift",
            "--reference",
            sample_parquet,
            "--new",
            str(new_path),
        ],
    )
    assert result.exit_code == 0
    assert "Drift Detection Report" in result.stdout
    assert "Drift detected" in result.stdout


@pytest.mark.slow
def test_hpo_command(sample_csv):
    result = runner.invoke(
        app,
        [
            "hpo",
            "--data",
            sample_csv,
            "--target",
            "target",
            "--model",
            "catboost",
            "--trials",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "HPO" in result.stdout
    assert "Best params" in result.stdout
    assert "Best value" in result.stdout


def test_hpo_unknown_model_exits_gracefully(tmp_path, monkeypatch):
    """Test that unknown model name produces clear error."""
    # Create a dummy data file
    data_file = tmp_path / "test.csv"
    data_file.write_text("a,b,target\n1,2,0\n3,4,1")

    result = runner.invoke(
        app,
        [
            "hpo",
            "--data",
            str(data_file),
            "--target",
            "target",
            "--model",
            "unknown_model_x",
        ],
    )

    assert result.exit_code == 1
    assert "Unknown model" in result.stdout
