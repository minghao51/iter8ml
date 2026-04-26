"""Tests for the champion export service."""

import json
from pathlib import Path

import pytest

from tabular_blueprint.services.export_service import ExportService


@pytest.fixture
def export_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = workspace / "artifacts"
    artifacts.mkdir()

    registry = {
        "credit_risk:classification": {
            "model": "CatBoost",
            "run_id": "exp_test_001",
            "score": 0.91,
            "metric_name": "roc_auc",
            "artifact_path": str(artifacts / "catboost_exp_test_001"),
            "registered_at": "2026-04-23T12:00:00Z",
        }
    }
    (workspace / "registry.json").write_text(json.dumps(registry))

    artifact = artifacts / "catboost_exp_test_001"
    artifact.write_text("fake_model_bytes")

    return workspace


def test_export_creates_portable_directory(export_workspace):
    service = ExportService(workspace_dir=export_workspace)
    export_path = service.export("credit_risk:classification")

    assert export_path.exists()
    assert (export_path / "model.artifact").exists()
    assert (export_path / "predictor.py").exists()
    assert (export_path / "metadata.json").exists()
    assert (export_path / "pipelines" / "preprocessing.py").exists()


def test_export_metadata_is_valid_json(export_workspace):
    service = ExportService(workspace_dir=export_workspace)
    service.export("credit_risk:classification")

    exports_dir = export_workspace / "exports" / "credit_risk_classification"
    metadata = json.loads((exports_dir / "metadata.json").read_text())

    assert metadata["model_name"] == "CatBoost"
    assert metadata["score"] == 0.91
    assert "model_class" in metadata


def test_export_copies_model_artifact(export_workspace):
    service = ExportService(workspace_dir=export_workspace)
    export_path = service.export("credit_risk:classification")

    content = (export_path / "model.artifact").read_text()
    assert content == "fake_model_bytes"


def test_export_raises_for_missing_key(export_workspace):
    service = ExportService(workspace_dir=export_workspace)
    with pytest.raises(ValueError, match="No champion registered"):
        service.export("nonexistent:key")


def test_export_raises_for_missing_artifact(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "artifacts").mkdir()

    registry = {
        "test:classification": {
            "model": "CatBoost",
            "run_id": "exp_001",
            "score": 0.9,
            "artifact_path": str(workspace / "artifacts" / "nonexistent_model"),
            "registered_at": "2026-04-23T12:00:00Z",
        }
    }
    (workspace / "registry.json").write_text(json.dumps(registry))

    service = ExportService(workspace_dir=workspace)
    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        service.export("test:classification")


def test_export_custom_output_dir(export_workspace, tmp_path: Path):
    custom_output = tmp_path / "my_export"
    service = ExportService(workspace_dir=export_workspace)
    export_path = service.export("credit_risk:classification", output_dir=custom_output)

    assert export_path == custom_output
    assert (custom_output / "model.artifact").exists()


def test_export_predictor_script_contains_class(export_workspace):
    service = ExportService(workspace_dir=export_workspace)
    export_path = service.export("credit_risk:classification")

    script = (export_path / "predictor.py").read_text()
    assert "class Predictor" in script
    assert "def predict(" in script
    assert "from pipelines.preprocessing import (" in script
    assert "from tabular_blueprint.pipelines.preprocessing import (" not in script
