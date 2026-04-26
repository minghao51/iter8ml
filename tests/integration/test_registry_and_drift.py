"""Integration tests for model registry and drift detection."""

import json

import polars as pl
import pytest
from sklearn.datasets import make_classification

from tabular_blueprint.monitoring.drift import DriftDetector
from tabular_blueprint.services.registry_service import RegistryService


@pytest.fixture
def populated_workspace(tmp_path):
    """Workspace with sample experiment events."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    events = [
        {
            "event": "model_completed",
            "run_id": "run_001",
            "model": "CatBoost",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.85, "f1_macro": 0.72},
            "artifact_path": "workspace/artifacts/catboost_run001.cbm",
        },
        {
            "event": "model_completed",
            "run_id": "run_002",
            "model": "LightGBM",
            "task": "classification",
            "cv_scores": {"roc_auc": 0.83, "f1_macro": 0.70},
            "artifact_path": "workspace/artifacts/lightgbm_run002.cbm",
        },
        {
            "event": "model_completed",
            "run_id": "run_003",
            "model": "CatBoost",
            "task": "regression",
            "cv_scores": {"rmse": 0.45, "r2": 0.82},
            "artifact_path": "workspace/artifacts/catboost_run003.cbm",
        },
    ]

    with open(workspace / "experiments.jsonl", "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    return workspace


def test_registry_update_if_better(populated_workspace):
    """Registry should update when new model beats champion."""
    registry_path = populated_workspace / "registry.json"
    registry = RegistryService(str(registry_path))

    updated = registry.update_if_better(
        key="classification:test",
        model_name="CatBoost",
        run_id="run_001",
        score=0.85,
        artifact_path="workspace/artifacts/catboost_run001.cbm",
    )

    assert updated is True
    assert registry.get("classification:test")["score"] == 0.85


def test_registry_keeps_best_score(populated_workspace):
    """Registry should not update with worse score."""
    registry_path = populated_workspace / "registry.json"
    registry = RegistryService(str(registry_path))

    registry.update_if_better(
        key="classification:test",
        model_name="CatBoost",
        run_id="run_001",
        score=0.85,
        artifact_path="workspace/artifacts/catboost_run001.cbm",
    )

    updated = registry.update_if_better(
        key="classification:test",
        model_name="LightGBM",
        run_id="run_002",
        score=0.80,
        artifact_path="workspace/artifacts/lightgbm_run002.cbm",
    )

    assert updated is False
    assert registry.get("classification:test")["model"] == "CatBoost"


def test_registry_promote_run(populated_workspace):
    """Registry should promote a specific run to champion."""
    registry_path = populated_workspace / "registry.json"
    registry = RegistryService(str(registry_path))

    log_path = populated_workspace / "experiments.jsonl"

    result = registry.promote_run(
        run_id="run_002",
        key="classification:test",
        log_path=log_path,
    )

    assert "Promoted" in result.message
    champion = registry.get("classification:test")
    assert champion["run_id"] == "run_002"


def test_registry_regression_uses_r2(populated_workspace):
    """Registry should use R2 for regression tasks."""
    registry_path = populated_workspace / "registry.json"
    registry = RegistryService(str(registry_path))

    updated = registry.update_if_better(
        key="regression:test",
        model_name="CatBoost",
        run_id="run_003",
        score=0.82,
        artifact_path="workspace/artifacts/catboost_run003.cbm",
    )

    assert updated is True
    assert registry.get("regression:test")["score"] == 0.82


def test_drift_detector_no_drift_on_identical_data():
    """DriftDetector should detect no drift on identical data."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    ref_df = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(X.shape[1])})
    ref_df = ref_df.with_columns(target=pl.Series(y))

    new_df = ref_df.clone()

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    assert report.drift_detected is False
    assert report.n_drifted == 0


def test_drift_detector_detects_numeric_drift():
    """DriftDetector should detect drift in numeric columns."""
    ref_df = pl.DataFrame({"feature": list(range(100))})

    shifted_df = pl.DataFrame({"feature": [x + 50 for x in range(100)]})

    detector = DriftDetector(ref_df)
    report = detector.detect(shifted_df)

    assert report.n_drifted >= 1


def test_drift_detector_detects_categorical_drift():
    """DriftDetector should detect drift in categorical columns."""
    ref_df = pl.DataFrame({"category": ["A"] * 50 + ["B"] * 50})

    shifted_df = pl.DataFrame({"category": ["A"] * 20 + ["B"] * 30 + ["C"] * 50})

    detector = DriftDetector(ref_df)
    report = detector.detect(shifted_df)

    assert report.n_drifted >= 1


def test_drift_detector_respects_alpha():
    """DriftDetector should respect significance level."""
    ref_df = pl.DataFrame({"feature": list(range(100))})
    shifted_df = pl.DataFrame({"feature": [x + 2 for x in range(100)]})

    strict_detector = DriftDetector(ref_df, alpha=0.01)
    lenient_detector = DriftDetector(ref_df, alpha=0.10)

    strict_report = strict_detector.detect(shifted_df)
    lenient_report = lenient_detector.detect(shifted_df)

    assert lenient_report.n_drifted >= strict_report.n_drifted


def test_drift_report_pydantic_validation():
    """DriftReport should validate as Pydantic model."""
    ref_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    new_df = ref_df.with_columns(pl.col("a") * 10)

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    assert hasattr(report, "drift_detected")
    assert hasattr(report, "n_columns_tested")
    assert hasattr(report, "n_drifted")
    assert hasattr(report, "column_results")
