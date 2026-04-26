"""Integration tests for exported prediction package execution."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.datasets import make_classification

from tabular_blueprint.models.conventional.catboost_model import CatBoostModel
from tabular_blueprint.services.export_service import ExportService


def test_exported_package_predictor_runs_end_to_end(tmp_path: Path):
    workspace = tmp_path / "workspace"
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True)

    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    model = CatBoostModel(task="classification")
    model.fit(X, y)

    artifact_path = artifacts / "catboost_exp_001"
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
    (workspace / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    service = ExportService(workspace_dir=workspace)
    export_path = service.export("export_test:classification")

    input_df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    input_path = tmp_path / "inference.csv"
    input_df.write_csv(input_path)

    sys.path.insert(0, str(export_path))
    try:
        spec = importlib.util.spec_from_file_location("predictor_module", export_path / "predictor.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        predictor = module.Predictor()
        preds = predictor.predict(str(input_path))
    finally:
        sys.path.pop(0)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(input_df)
