"""Contract tests for the local medallion runtime."""

import gzip
import json
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest

from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.dataflows import materialize_bronze, materialize_gold, materialize_silver
from iter8ml.dataflows.gold import build_split_frame
from iter8ml.domain.events import EventEnvelope, JsonlEventSink
from iter8ml.domain.hashing import dataframe_digest
from iter8ml.domain.manifests import SourceSpec, SplitSpec
from iter8ml.exceptions import ArtifactError
from iter8ml.orchestration import LocalOrchestrator, MedallionExecutionService
from iter8ml.orchestration.protocol import RunHandle
from iter8ml.runtime.plan import compile_run_plan
from iter8ml.storage import LocalArtifactStore, LocalCatalogStore
from iter8ml.verification.split_validation import validate_split
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


def test_idempotent_materialization_rejects_corrupted_product(tmp_path):
    store = LocalArtifactStore(tmp_path)
    source = SourceSpec(name="fixture", source_type="memory", uri="memory://fixture")
    bronze = materialize_bronze(_frame(), source, store)
    data_path = next((tmp_path / "lake").glob(f"*/**/{bronze.product_id}/data/data.parquet"))
    data_path.write_bytes(b"corrupted")

    with pytest.raises(ArtifactError, match="failed verification"):
        materialize_bronze(_frame(), source, store)


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


def test_gold_materialization_is_idempotent(tmp_path):
    store = LocalArtifactStore(tmp_path)
    source = SourceSpec(name="fixture", source_type="memory", uri="memory://fixture")
    bronze = materialize_bronze(_frame(), source, store)
    silver = materialize_silver(_frame(), bronze, store, target_col="target")

    first, first_split = materialize_gold(_frame(), silver, store, target_col="target")
    second, second_split = materialize_gold(_frame(), silver, store, target_col="target")

    assert second == first
    assert second_split == first_split


def test_silver_contract_blocks_invalid_data(tmp_path):
    store = LocalArtifactStore(tmp_path)
    frame = pl.DataFrame({"record": [1, 1], "feature": [1.0, None], "target": [0, 1]})
    bronze = materialize_bronze(
        frame, SourceSpec(name="fixture", source_type="memory", uri="memory://fixture"), store
    )

    with pytest.raises(ValueError, match="missing required columns"):
        materialize_silver(
            frame,
            bronze,
            store,
            target_col="target",
            contract={"required_columns": {"missing": "Int64"}},
        )
    with pytest.raises(ValueError, match="dtype mismatch"):
        materialize_silver(
            frame,
            bronze,
            store,
            target_col="target",
            contract={"required_columns": {"feature": "String"}},
        )
    with pytest.raises(ValueError, match="uniqueness failed"):
        materialize_silver(
            frame,
            bronze,
            store,
            target_col="target",
            contract={"unique": ["record"]},
        )
    with pytest.raises(ValueError, match="null threshold exceeded"):
        materialize_silver(
            frame,
            bronze,
            store,
            target_col="target",
            contract={"null_thresholds": {"feature": 0.0}},
        )


def test_random_split_membership_is_stable_after_row_reordering():
    frame = pl.DataFrame(
        {"record": list(range(18)), "feature": list(range(100, 118)), "target": [0, 1] * 9}
    )
    spec = SplitSpec(strategy="stratified", folds=3, shuffle=True, random_seed=42)

    original = build_split_frame(frame, "target", spec)
    reordered = build_split_frame(frame.reverse(), "target", spec)

    assert original.sort(["fold", "role", "row_id"]).equals(
        reordered.sort(["fold", "role", "row_id"])
    )


def test_unshuffled_kfold_ignores_random_seed():
    split = build_split_frame(
        _frame(),
        "target",
        SplitSpec(strategy="kfold", folds=3, shuffle=False, random_seed=42),
    )

    assert validate_split(split)["ok"] is True


def test_purged_time_embargo_removes_nearest_training_rows():
    frame = pl.DataFrame(
        {"time": list(range(20)), "feature": list(range(20)), "target": [0, 1] * 10}
    )
    plain = build_split_frame(
        frame,
        "target",
        SplitSpec(strategy="time", folds=3, time_column="time"),
    )
    purged = build_split_frame(
        frame,
        "target",
        SplitSpec(strategy="purged_time", folds=3, time_column="time", embargo=2),
    )

    for fold in purged["fold"].unique().to_list():
        plain_train = plain.filter((pl.col("fold") == fold) & (pl.col("role") == "train")).height
        purged_train = purged.filter((pl.col("fold") == fold) & (pl.col("role") == "train")).height
        assert purged_train == plain_train - 2


def test_gold_rejects_non_strict_temporal_order_before_commit(tmp_path):
    frame = pl.DataFrame({"time": [1] * 12, "feature": list(range(12)), "target": [0, 1] * 6})
    store = LocalArtifactStore(tmp_path)
    bronze = materialize_bronze(
        frame, SourceSpec(name="fixture", source_type="memory", uri="memory://fixture"), store
    )
    silver = materialize_silver(frame, bronze, store, target_col="target")

    with pytest.raises(ValueError, match="temporal split ordering"):
        materialize_gold(
            frame,
            silver,
            store,
            target_col="target",
            split_spec=SplitSpec(strategy="time", folds=3, time_column="time"),
        )

    assert list(store.list_products("gold")) == []


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
    assert all(stage["status"] == "succeeded" for stage in manifest["stages"])


def test_run_service_reuses_same_data_and_separates_different_data(tmp_path):
    config = ExperimentConfig(
        name="fixture_run",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
    )
    service = MedallionExecutionService(Workspace(root=tmp_path))

    first = service.run(config, _frame(), execute_training=False)
    repeated = service.run(config, _frame(), execute_training=False)
    changed = service.run(
        config,
        _frame().with_columns((pl.col("feature") + 100).alias("feature")),
        execute_training=False,
    )

    assert repeated.run_id == first.run_id
    assert changed.run_id != first.run_id


def test_run_service_binds_feature_spec_to_gold_identity(tmp_path):
    config = ExperimentConfig(
        name="fixture_run",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
    )
    service = MedallionExecutionService(Workspace(root=tmp_path))
    first_plan = compile_run_plan(config)
    second_plan = first_plan.model_copy(update={"features": {"strategy": "different"}})

    first = service.run(config, _frame(), plan=first_plan, execute_training=False)
    second = service.run(config, _frame(), plan=second_plan, execute_training=False)

    assert first.products[2] != second.products[2]


def test_run_service_does_not_reuse_untrained_run_for_training(tmp_path, monkeypatch):
    config = ExperimentConfig(
        name="fixture_run",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
    )
    service = MedallionExecutionService(Workspace(root=tmp_path))
    untrained = service.run(config, _frame(), execute_training=False)
    monkeypatch.setattr(
        "iter8ml.engine.trainer.Trainer.run", lambda self, frame, split_frame=None: {}
    )

    trained = service.run(config, _frame(), execute_training=True)

    assert trained.run_id != untrained.run_id


def test_cancellation_request_is_honored_at_next_stage_boundary(tmp_path, monkeypatch):
    config = ExperimentConfig(
        name="fixture_run",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
    )
    workspace = Workspace(root=tmp_path)
    service = MedallionExecutionService(workspace)
    original_materialize_bronze = materialize_bronze

    def materialize_and_cancel(*args, **kwargs):
        product = original_materialize_bronze(*args, **kwargs)
        run_path = next(workspace.runs_dir.glob("*/run.json"))
        LocalOrchestrator(workspace).cancel(run_path.parent.name)
        return product

    monkeypatch.setattr("iter8ml.orchestration.service.materialize_bronze", materialize_and_cancel)

    result = service.run(config, _frame(), execute_training=False)
    manifest = json.loads(result.manifest_path.read_text())

    assert result.status == "cancelled"
    assert manifest["status"] == "cancelled"
    assert manifest["event_archive"].endswith(".events.jsonl.gz")
    assert (tmp_path / manifest["event_archive"]).exists()
    assert (result.manifest_path.parent / "CANCEL_REQUESTED").exists()

    with gzip.open(tmp_path / manifest["event_archive"], "rt", encoding="utf-8") as handle:
        events = [EventEnvelope.model_validate_json(line) for line in handle]
    assert any(event.event_type == "run.cancelled" for event in events)
    assert events[-1].payload == {"terminal_status": "cancelled"}


def test_failed_run_finalizes_event_archive(tmp_path):
    config = ExperimentConfig(
        name="fixture_run",
        task=TaskType.CLASSIFICATION,
        target_col="missing_target",
        data_path="",
    )
    workspace = Workspace(root=tmp_path)
    service = MedallionExecutionService(workspace)

    with pytest.raises(ValueError, match="target_col 'missing_target' is not present"):
        service.run(config, _frame(), execute_training=False)

    manifest_path = next(workspace.runs_dir.glob("*/run.json"))
    manifest = json.loads(manifest_path.read_text())
    archive_path = tmp_path / manifest["event_archive"]

    assert manifest["status"] == "failed"
    assert archive_path.exists()
    assert not (manifest_path.parent / f"{manifest_path.parent.name}.jsonl").exists()
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        events = [EventEnvelope.model_validate_json(line) for line in handle]
    assert any(event.event_type == "run.failed" for event in events)
    assert events[-1].payload == {"terminal_status": "failed"}


def test_concurrent_equivalent_product_commits_are_idempotent(tmp_path):
    store = LocalArtifactStore(tmp_path)
    source = SourceSpec(name="fixture", source_type="memory", uri="memory://fixture")

    def materialize() -> str:
        return materialize_bronze(_frame(), source, store).product_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        product_ids = list(executor.map(lambda _: materialize(), range(2)))

    assert product_ids[0] == product_ids[1]
    assert len(list(store.list_products("bronze"))) == 1
    assert store.verify(product_ids[0], deep=True)["ok"] is True


def test_event_archive_contains_only_valid_envelopes_and_closes_sink(tmp_path):
    sink = JsonlEventSink(tmp_path, "run_1")
    sink.append(EventEnvelope(run_id="run_1", event_type="run.started"))

    archive = sink.finalize("succeeded")

    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        events = [EventEnvelope.model_validate_json(line) for line in handle]
    assert events[-1].event_type == "run.finalized"
    assert events[-1].payload == {"terminal_status": "succeeded"}
    assert not sink.hot_path.exists()
    with pytest.raises(RuntimeError, match="already finalized"):
        sink.append(EventEnvelope(run_id="run_1", event_type="late.event"))


def test_incomplete_product_is_not_readable(tmp_path):
    store = LocalArtifactStore(tmp_path)
    bronze = materialize_bronze(
        _frame(), SourceSpec(name="fixture", source_type="memory", uri="memory://fixture"), store
    )
    product_dir = next((tmp_path / "lake").glob(f"*/**/{bronze.product_id}"))
    (product_dir / "_SUCCESS").unlink()

    assert store.exists(bronze.product_id) is False
    assert store.verify(bronze.product_id)["ok"] is False
    with pytest.raises(ArtifactError, match="Committed product"):
        store.read_manifest(bronze.product_id)


def test_artifact_store_rejects_product_path_traversal(tmp_path):
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactError, match="Invalid product_id"):
        store.begin("../escape", product_type="bronze", name="fixture")
    with pytest.raises(ArtifactError, match="Invalid product name"):
        store.begin("bronze_fixture_123", product_type="bronze", name="../escape")


def test_catalog_query_rejects_writes(tmp_path):
    catalog = LocalCatalogStore(tmp_path)

    with pytest.raises(ValueError, match="read-only"):
        catalog.query("DELETE FROM products")
    assert catalog.query("SELECT COUNT(*) AS count FROM products")["count"].item() == 0


def test_plan_and_dataframe_digests_are_stable():
    left = _frame()
    right = pl.DataFrame({"feature": list(range(12)), "target": [0, 1] * 6})
    assert dataframe_digest(left) == dataframe_digest(right)
    assert dataframe_digest(left) == dataframe_digest(right.reverse())
    assert EventEnvelope(run_id="run_1", event_type="run.started").schema_version == 1


def test_plan_sanitizes_names_to_source_contract():
    config = ExperimentConfig(
        name="123 Δ", task=TaskType.CLASSIFICATION, target_col="target", data_path=""
    )

    plan = compile_run_plan(config)

    assert plan.source.name == "experiment_123"


def _file_backed_plan(tmp_path, execute_training: bool):
    csv = tmp_path / "data.csv"
    pl.DataFrame({"f": list(range(12)), "target": [0, 1] * 6}).write_csv(csv)
    config = ExperimentConfig(
        name="orchestrator_run",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path=str(csv),
    )
    plan = compile_run_plan(config)
    return plan.model_copy(
        update={"documentation": {**plan.documentation, "execute_training": execute_training}}
    )


def test_orchestrator_seam_submit_and_status(tmp_path):
    service = MedallionExecutionService(Workspace(root=tmp_path))
    handle = service.submit(_file_backed_plan(tmp_path, execute_training=False))

    assert isinstance(handle, RunHandle)
    status = service.status(handle.run_id)
    assert status["status"] == "succeeded"
    assert status["cancellation_requested"] is False


def test_orchestrator_seam_cancel_is_honored(tmp_path, monkeypatch):
    workspace = Workspace(root=tmp_path)
    service = MedallionExecutionService(workspace)
    original_materialize_bronze = materialize_bronze

    def materialize_and_cancel(*args, **kwargs):
        product = original_materialize_bronze(*args, **kwargs)
        run_path = next(workspace.runs_dir.glob("*/run.json"))
        service.cancel(run_path.parent.name)
        return product

    monkeypatch.setattr("iter8ml.orchestration.service.materialize_bronze", materialize_and_cancel)

    handle = service.submit(_file_backed_plan(tmp_path, execute_training=False))
    manifest_path = workspace.runs_dir / handle.run_id / "run.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "cancelled"
    assert (manifest_path.parent / "CANCEL_REQUESTED").exists()
