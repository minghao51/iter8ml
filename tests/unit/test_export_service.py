"""Tests for the champion export service."""

import json
from pathlib import Path

import pytest

from iter8ml.services.export import ExportService
from iter8ml.workspace import Workspace


@pytest.fixture
def export_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "workspace")
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.artifacts_dir.mkdir(parents=True, exist_ok=True)

    registry = {
        "credit_risk:classification": {
            "model": "CatBoost",
            "run_id": "exp_test_001",
            "score": 0.91,
            "metric_name": "roc_auc",
            "artifact_path": str(ws.artifacts_dir / "catboost_exp_test_001"),
            "registered_at": "2026-04-23T12:00:00Z",
        }
    }
    ws.registry_path.write_text(json.dumps(registry))

    artifact = ws.artifacts_dir / "catboost_exp_test_001"
    artifact.write_text("fake_model_bytes")

    return ws


def test_export_creates_portable_directory(export_workspace):
    service = ExportService(workspace=export_workspace)
    export_path = service.export("credit_risk:classification")

    assert export_path.exists()
    assert (export_path / "model.artifact").exists()
    assert (export_path / "predictor.py").exists()
    assert (export_path / "metadata.json").exists()
    assert (export_path / "pipelines" / "preprocessing.py").exists()


def test_export_metadata_is_valid_json(export_workspace):
    service = ExportService(workspace=export_workspace)
    service.export("credit_risk:classification")

    exports_dir = export_workspace.exports_dir / "credit_risk_classification"
    metadata = json.loads((exports_dir / "metadata.json").read_text())

    assert metadata["model_name"] == "CatBoost"
    assert metadata["score"] == 0.91
    assert "model_class" in metadata
    assert "allowlisted_model_classes" in metadata


def test_export_copies_model_artifact(export_workspace):
    service = ExportService(workspace=export_workspace)
    export_path = service.export("credit_risk:classification")

    content = (export_path / "model.artifact").read_text()
    assert content == "fake_model_bytes"


def test_export_raises_for_missing_key(export_workspace):
    service = ExportService(workspace=export_workspace)
    with pytest.raises(ValueError, match="No champion registered"):
        service.export("nonexistent:key")


def test_export_raises_for_missing_artifact(tmp_path: Path):
    ws = Workspace(root=tmp_path / "workspace")
    ws.root.mkdir(parents=True)
    ws.artifacts_dir.mkdir(parents=True)

    registry = {
        "test:classification": {
            "model": "CatBoost",
            "run_id": "exp_001",
            "score": 0.9,
            "artifact_path": str(ws.artifacts_dir / "nonexistent_model"),
            "registered_at": "2026-04-23T12:00:00Z",
        }
    }
    ws.registry_path.write_text(json.dumps(registry))

    service = ExportService(workspace=ws)
    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        service.export("test:classification")


def test_export_custom_output_dir(export_workspace, tmp_path: Path):
    custom_output = tmp_path / "my_export"
    service = ExportService(workspace=export_workspace)
    export_path = service.export("credit_risk:classification", output_dir=custom_output)

    assert export_path == custom_output
    assert (custom_output / "model.artifact").exists()


def test_export_predictor_script_contains_class(export_workspace):
    service = ExportService(workspace=export_workspace)
    export_path = service.export("credit_risk:classification")

    script = (export_path / "predictor.py").read_text()
    assert "class Predictor" in script
    assert "def predict(" in script
    assert "_build_preprocessing_driver" in script
    assert "_preprocess" in script


def test_exported_predictor_rejects_non_allowlisted_model_class(export_workspace):
    import importlib.util

    service = ExportService(workspace=export_workspace)
    export_path = service.export("credit_risk:classification")
    metadata_path = export_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_class"] = ["os", "system"]
    metadata_path.write_text(json.dumps(metadata))

    spec = importlib.util.spec_from_file_location("predictor_module", export_path / "predictor.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="not allowlisted"):
        module.Predictor()
