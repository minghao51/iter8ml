"""Contract tests for the local medallion runtime."""

import json

import polars as pl

from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.dataflows import materialize_bronze, materialize_gold, materialize_silver
from iter8ml.domain.events import EventEnvelope
from iter8ml.domain.hashing import dataframe_digest
from iter8ml.domain.manifests import SourceSpec
from iter8ml.orchestration import MedallionExecutionService
from iter8ml.storage import LocalArtifactStore
from iter8ml.workspace import Workspace


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"feature": list(range(12)), "target": [0, 1] * 6})


def test_atomic_product_store_and_deep_verification(tmp_path):
    store = LocalArtifactStore(tmp_path)
    bronze = materialize_bronze(
        _frame(), SourceSpec(name="fixture", source_type="memory", uri="memory://fixture"), store
    )

    result = store.verify(bronze.product_id, deep=True)

    assert result["ok"] is True
    assert (tmp_path / "lake" / "01_bronze").exists()


def test_gold_split_has_no_fold_overlap(tmp_path):
    store = LocalArtifactStore(tmp_path)
    source = SourceSpec(name="fixture", source_type="memory", uri="memory://fixture")
    bronze = materialize_bronze(_frame(), source, store)
    silver = materialize_silver(_frame(), bronze, store, target_col="target")
    gold, split = materialize_gold(_frame(), silver, store, target_col="target")

    split_path = next((tmp_path / "lake").glob(f"*/**/{gold.product_id}/splits.parquet"))
    split_frame = pl.read_parquet(split_path)
    for fold in split_frame["fold"].unique().to_list():
        fold_frame = split_frame.filter(pl.col("fold") == fold)
        train = set(fold_frame.filter(pl.col("role") == "train")["row_id"].to_list())
        validation = set(fold_frame.filter(pl.col("role") == "validation")["row_id"].to_list())
        assert train.isdisjoint(validation)
    assert split.overlap_checks_passed


def test_run_service_persists_products_manifest_and_event_archive(tmp_path):
    config = ExperimentConfig(
        name="fixture_run",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
    )
    service = MedallionExecutionService(Workspace(root=tmp_path))
    result = service.run(config, _frame(), execute_training=False)
    manifest = json.loads(result.manifest_path.read_text())

    assert result.status == "succeeded"
    assert len(result.products) == 4
    assert manifest["event_archive"].endswith(".events.jsonl.gz")
    assert (tmp_path / manifest["event_archive"]).exists()
    assert len(service.resume(result.run_id).products) == 4


def test_plan_and_dataframe_digests_are_stable():
    left = _frame()
    right = pl.DataFrame({"feature": list(range(12)), "target": [0, 1] * 6})
    assert dataframe_digest(left) == dataframe_digest(right)
    assert EventEnvelope(run_id="run_1", event_type="run.started").schema_version == 1
