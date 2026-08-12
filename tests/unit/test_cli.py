"""Tests for CLI commands using typer CliRunner."""

import os
from pathlib import Path

import polars as pl
import pytest
from sklearn.datasets import make_classification
from typer.testing import CliRunner

from iter8ml.cli import app
from iter8ml.workspace import Workspace

runner = CliRunner()


@pytest.fixture
def isolated_cwd(tmp_path):
    """Change to a temp directory for CLI commands that depend on CWD."""
    orig = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        yield tmp_path
    finally:
        os.chdir(orig)


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


def test_init_command(isolated_cwd):
    tmpdir = isolated_cwd
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Workspace initialized" in result.stdout
    assert Path(tmpdir, "workspace").exists()
    assert Path(tmpdir, "workspace", "artifacts").exists()
    assert Path(tmpdir, "workspace", "experiments.jsonl").exists()
    assert Path(tmpdir, "workspace", "registry.json").exists()


def test_init_preserves_existing_registry_without_force_reset(isolated_cwd):
    tmpdir = isolated_cwd
    workspace = Path(tmpdir, "workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    registry_path = workspace / "registry.json"
    registry_path.write_text('{"existing":"champion"}')

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert registry_path.read_text() == '{"existing":"champion"}'


def test_init_force_reset_registry_overwrites_existing_registry(isolated_cwd):
    tmpdir = isolated_cwd
    workspace = Path(tmpdir, "workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    registry_path = workspace / "registry.json"
    registry_path.write_text('{"existing":"champion"}')

    result = runner.invoke(app, ["init", "--force-reset-registry"])
    assert result.exit_code == 0
    assert registry_path.read_text() == "{}"


def test_init_with_data(sample_csv):
    result = runner.invoke(app, ["init", "--data", sample_csv])
    assert result.exit_code == 0
    assert "Data path set to" in result.stdout


# --- Demo dataset seeding ---


def test_init_demo_seeds_telco_churn(isolated_cwd):
    tmpdir = isolated_cwd
    result = runner.invoke(app, ["init", "--demo"])
    assert result.exit_code == 0
    assert "Workspace initialized" in result.stdout
    assert "Demo dataset seeded" in result.stdout
    assert "Churn" in result.stdout
    assert "iter8 run" in result.stdout
    seeded = Path(tmpdir, "workspace", "data", "telco_churn.parquet")
    assert seeded.exists()
    assert seeded.stat().st_size > 0


def test_init_demo_and_data_are_independent(isolated_cwd):
    result = runner.invoke(app, ["init", "--demo", "--data", "foo.csv"])
    assert result.exit_code == 0
    assert "Demo dataset seeded" in result.stdout
    assert "Data path set to: foo.csv" in result.stdout


def test_init_creates_data_dir(isolated_cwd):
    tmpdir = isolated_cwd
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert Path(tmpdir, "workspace", "data").is_dir()


def test_workspace_seed_demo_data_unknown_name(tmp_path):
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    with pytest.raises(KeyError, match="Unknown bundled dataset"):
        ws.seed_demo_data("does_not_exist")


def test_hardware_command():
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "Hardware Profile" in result.stdout
    assert "RAM" in result.stdout
    assert "CPU Cores" in result.stdout


def test_leaderboard_empty(isolated_cwd):
    result = runner.invoke(app, ["leaderboard"])
    assert result.exit_code == 0
    assert "No experiments" in result.stdout


def test_registry_empty(isolated_cwd):
    result = runner.invoke(app, ["registry", "show"])
    assert result.exit_code == 0
    assert "Registry is empty" in result.stdout


def test_state_empty(isolated_cwd):
    result = runner.invoke(app, ["state"])
    assert result.exit_code == 0
    assert "No experiments run" in result.stdout


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
def test_leaderboard_after_run(sample_csv, isolated_cwd):
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


# --- Export & Registry ---


def test_registry_show_empty(isolated_cwd):
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["registry", "show"])
    assert result.exit_code == 0
    assert "Registry is empty" in result.stdout


def test_registry_show_with_data(isolated_cwd):
    tmpdir = isolated_cwd
    runner.invoke(app, ["init"])
    ws_path = Path(tmpdir) / "workspace"
    registry_path = ws_path / "registry.json"
    registry_path.write_text(
        '{"best": {"model": "CatBoost", "run_id": "exp_1",'
        ' "score": 0.85, "registered_at": "2026-01-01T00:00:00Z"}}'
    )
    result = runner.invoke(app, ["registry", "show"])
    assert result.exit_code == 0
    assert "CatBoost" in result.stdout
    assert "exp_1" in result.stdout


def test_registry_unknown_action(isolated_cwd):
    tmpdir = isolated_cwd
    runner.invoke(app, ["init"])
    ws_path = Path(tmpdir) / "workspace"
    registry_path = ws_path / "registry.json"
    registry_path.write_text('{"best": {"model": "CatBoost", "run_id": "exp_1", "score": 0.85}}')
    result = runner.invoke(app, ["registry", "invalid_action"])
    assert "Unknown action" in result.stdout


def test_export_missing_key(isolated_cwd):
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["export", "nonexistent:key"])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_state_with_events(isolated_cwd):
    tmpdir = isolated_cwd
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0

    ws_path = Path(tmpdir) / "workspace"
    exp_path = ws_path / "experiments.jsonl"
    event = (
        '{"event": "model_completed", "model": "CatBoost", "task": "classification",'
        ' "dataset": "test", "n_rows": 100, "n_features": 5,'
        ' "cv_scores": {"roc_auc": 0.85}, "duration_seconds": 3.0,'
        ' "hardware": {"device": "cpu", "vram_used_gb": 0.0},'
        ' "timestamp": "2026-01-01T00:00:00Z"}\n'
    )
    exp_path.write_text(event)

    result = runner.invoke(app, ["state"])
    assert result.exit_code == 0
    assert "CatBoost" in result.stdout
    assert "roc_auc" in result.stdout


def test_diff_command(isolated_cwd):
    tmpdir = isolated_cwd
    runner.invoke(app, ["init"])
    ws_path = Path(tmpdir) / "workspace"
    exp_path = ws_path / "experiments.jsonl"
    events = (
        '{"event": "experiment_started", "run_id": "run_a", '
        '"config": {"task": "classification", "models": "auto",'
        ' "cv_folds": 5, "metrics": ["roc_auc"]}}\n'
        '{"event": "model_completed", "run_id": "run_a",'
        ' "model": "CatBoost",'
        ' "cv_scores": {"roc_auc": 0.85}, "duration_seconds": 3.0}\n'
        '{"event": "experiment_started", "run_id": "run_b", '
        '"config": {"task": "classification", "models": "auto",'
        ' "cv_folds": 5, "metrics": ["roc_auc"]}}\n'
        '{"event": "model_completed", "run_id": "run_b",'
        ' "model": "CatBoost",'
        ' "cv_scores": {"roc_auc": 0.90}, "duration_seconds": 2.5}\n'
    )
    exp_path.write_text(events)

    result = runner.invoke(app, ["diff", "run_a", "run_b"])
    assert result.exit_code == 0
    assert "Experiment Diff" in result.stdout
    assert "run_a" in result.stdout
    assert "run_b" in result.stdout


def test_diff_missing_run(isolated_cwd):
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["diff", "nonexistent", "other"])
    assert result.exit_code == 1
    assert "Run ID not found" in result.stdout
