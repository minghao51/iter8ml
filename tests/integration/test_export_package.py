"""Integration tests for exported prediction package execution."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification

from iter8ml.engine.models.catboost_model import CatBoostModel
from iter8ml.services.export import ExportService
from iter8ml.workspace import Workspace


def _setup_export_workspace(tmp_path: Path) -> tuple[Path, Path, object]:
    ws = Workspace(root=tmp_path / "workspace")
    ws.root.mkdir(parents=True)
    ws.artifacts_dir.mkdir(parents=True)

    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    model = CatBoostModel(task="classification")
    model.fit(X, y)

    artifact_path = ws.artifacts_dir / "catboost_exp_001"
    model.save(str(artifact_path))

    registry = {
        "export_test:classification": {
            "model": "CatBoost",
            "run_id": "exp_001",
            "score": 0.88,
            "metric_name": "roc_auc",
            "artifact_path": str(artifact_path),
            "registered_at": "2026-04-27T00:00:00Z",
        }
    }
    ws.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    service = ExportService(workspace=ws)
    export_path = service.export("export_test:classification")
    return ws.root, export_path, model


def _load_predictor(export_path: Path):
    sys.path.insert(0, str(export_path))
    try:
        spec = importlib.util.spec_from_file_location(
            "predictor_module", export_path / "predictor.py"
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Predictor()
    finally:
        sys.path.pop(0)


def test_exported_package_predictor_runs_end_to_end(tmp_path: Path):
    pytest.importorskip("hamilton")
    _, export_path, _ = _setup_export_workspace(tmp_path)

    X, _ = make_classification(n_samples=80, n_features=4, random_state=42)
    input_df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    input_path = tmp_path / "inference.csv"
    input_df.write_csv(input_path)

    predictor = _load_predictor(export_path)
    preds = predictor.predict(str(input_path))

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(input_df)


def test_exported_predictor_uses_hamilton_preprocessing(tmp_path: Path):
    pytest.importorskip("hamilton")
    _, export_path, _ = _setup_export_workspace(tmp_path)

    predictor = _load_predictor(export_path)
    assert predictor._dr is not None, "Hamilton driver should be available in export"

    df = pl.DataFrame({"a": [1.0, 2.0, None], "b": ["x", None, "z"]})
    result = predictor._preprocess(df)
    assert isinstance(result, pl.DataFrame)
    assert result.height == 3


def test_export_preprocessing_parity_with_training_pipeline(tmp_path: Path):
    pytest.importorskip("hamilton")
    from iter8ml.engine.pipelines.executor import PipelineExecutor

    _, export_path, _ = _setup_export_workspace(tmp_path)

    df = pl.DataFrame(
        {
            "num": [1.0, None, 3.0],
            "cat": ["a", "b", None],
            "val": [10.0, 20.0, 30.0],
        }
    )

    executor = PipelineExecutor()
    training_result = executor.run_preprocessing(df)

    predictor = _load_predictor(export_path)
    export_result = predictor._preprocess(df)

    assert training_result.columns == export_result.columns
    assert training_result.shape == export_result.shape
