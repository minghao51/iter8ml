"""Local run lifecycle around Bronze -> Silver -> Gold -> Platinum."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

import polars as pl

from iter8ml.config import CVStrategy, ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.data.loader import load_data
from iter8ml.dataflows.bronze import materialize_bronze
from iter8ml.dataflows.gold import materialize_gold
from iter8ml.dataflows.platinum_train import materialize_platinum
from iter8ml.dataflows.silver import materialize_silver
from iter8ml.domain.events import EventEnvelope, JsonlEventSink
from iter8ml.domain.hashing import dataframe_digest, digest
from iter8ml.domain.ids import run_id as make_run_id
from iter8ml.domain.manifests import RunManifest, RunPlan, StageRecord
from iter8ml.exceptions import CancellationRequested
from iter8ml.orchestration.protocol import RunHandle
from iter8ml.runtime.plan import compile_run_plan
from iter8ml.storage.catalog import LocalCatalogStore
from iter8ml.storage.local import LocalArtifactStore
from iter8ml.verification.split_validation import validate_split
from iter8ml.workspace import Workspace


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    status: str
    manifest_path: Path
    products: tuple[str, ...]


class MedallionExecutionService:
    """Reference local runner; legacy Trainer remains the Platinum engine."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace.init()
        self.store = LocalArtifactStore(self.workspace.root)
        self.catalog = LocalCatalogStore(self.workspace.root)

    def run(
        self,
        config: ExperimentConfig,
        frame: pl.DataFrame,
        *,
        plan: RunPlan | None = None,
        execute_training: bool = True,
    ) -> ExecutionResult:
        resolved_plan = plan or compile_run_plan(config)
        resolved_plan = resolved_plan.model_copy(
            update={
                "documentation": {
                    **resolved_plan.documentation,
                    "execute_training": execute_training,
                }
            }
        )
        observed_fingerprint = dataframe_digest(frame)
        if resolved_plan.source.fingerprint != observed_fingerprint:
            resolved_plan = resolved_plan.model_copy(
                update={
                    "source": resolved_plan.source.model_copy(
                        update={"fingerprint": observed_fingerprint}
                    )
                }
            )
        existing_run = self._find_successful_run(resolved_plan.run_key)
        if existing_run is not None:
            return self.resume(existing_run.run_id)
        run_id = self._allocate_run_id(resolved_plan.run_key)
        manifest = RunManifest(
            run_id=run_id,
            run_key=resolved_plan.run_key,
            plan=resolved_plan,
            graph_version=digest("iter8ml.medallion.graph.v1"),
            stages=[
                StageRecord(name=name)  # type: ignore[arg-type]
                for name in ("bronze", "silver", "gold", "platinum", "publish")
            ],
        )
        event_sink = JsonlEventSink(
            self.workspace.events_dir, run_id, graph_version=manifest.graph_version
        )
        products: list[str] = []
        self._save_manifest(manifest)
        try:
            manifest = self._transition(manifest, "running")
            self._event(
                event_sink, manifest, "run.started", "run", {"plan": resolved_plan.plan_name}
            )

            bronze = self._stage(
                manifest,
                event_sink,
                "bronze",
                lambda: materialize_bronze(
                    frame,
                    resolved_plan.source,
                    self.store,
                    specification=resolved_plan.source.model_dump(mode="json"),
                ),
            )
            products.append(bronze.product_id)
            self._record_stage(manifest, "bronze", bronze)
            self.catalog.register_product(bronze)
            silver = self._stage(
                manifest,
                event_sink,
                "silver",
                lambda: materialize_silver(
                    frame,
                    bronze,
                    self.store,
                    target_col=config.target_col,
                    contract=resolved_plan.contract,
                ),
            )
            products.append(silver.product_id)
            self._record_stage(manifest, "silver", silver)
            self.catalog.register_product(silver)
            gold, split = self._stage(
                manifest,
                event_sink,
                "gold",
                lambda: materialize_gold(
                    frame,
                    silver,
                    self.store,
                    target_col=config.target_col,
                    split_spec=resolved_plan.split,
                    feature_spec=resolved_plan.features,
                ),
            )
            split_frame = self._read_artifact(split.artifact.uri)
            split_result = validate_split(split_frame)
            if not split_result["ok"]:
                raise ValueError(f"Gold leakage gate failed: {split_result['errors']}")
            products.append(gold.product_id)
            self._record_stage(manifest, "gold", gold)
            self.catalog.register_product(gold)

            results: dict[str, Any] = {}
            if execute_training:
                from iter8ml.engine.trainer import Trainer

                results = Trainer(config=config, workspace=self.workspace).run(
                    frame, split_frame=split_frame
                )
            platinum = self._stage(
                manifest,
                event_sink,
                "platinum",
                lambda: materialize_platinum(
                    run_id,
                    gold,
                    results,
                    self.store,
                    experiment_name=resolved_plan.source.name,
                ),
            )
            products.append(platinum.product_id)
            self._record_stage(manifest, "platinum", platinum)
            self.catalog.register_product(platinum)
            self._complete_publish(manifest, event_sink, products)
            manifest = self._transition(manifest, "succeeded")
            self._event(event_sink, manifest, "run.succeeded", "publish", {"products": products})
        except CancellationRequested:
            manifest.status = "cancelled"
            manifest.ended_at = datetime.now(UTC)
            self._save_manifest(manifest)
            self._event(event_sink, manifest, "run.cancelled", "publish", {})
        except KeyboardInterrupt:
            manifest.status = "cancelled"
            manifest.ended_at = datetime.now(UTC)
            self._save_manifest(manifest)
            self._event(event_sink, manifest, "run.cancelled", "publish", {})
            raise
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = {"type": type(exc).__name__, "message": str(exc)}
            manifest.ended_at = datetime.now(UTC)
            self._save_manifest(manifest)
            self._event(event_sink, manifest, "run.failed", "publish", manifest.error)
            raise
        finally:
            archive = event_sink.finalize(manifest.status)
            manifest.event_archive = str(archive.relative_to(self.workspace.root))
            self._save_manifest(manifest)
            self.catalog.register_run(manifest)
        return ExecutionResult(
            run_id, manifest.status, self._manifest_path(run_id), tuple(products)
        )

    # ── Orchestrator protocol (single orchestration seam) ──────────────────

    @staticmethod
    def _config_from_plan(plan: RunPlan) -> ExperimentConfig:
        strategy_map = {
            "stratified": CVStrategy.STRATIFIED,
            "time": CVStrategy.TIMESERIES,
            "purged_time": CVStrategy.TIMESERIES,
        }
        cv_strategy = strategy_map.get(plan.split.strategy, CVStrategy.KFOLD)
        task_value = plan.target.get("task") or "classification"
        target_value = plan.target.get("column") or "target"
        return ExperimentConfig(
            name=plan.source.name,
            task=TaskType(str(task_value)),
            target_col=str(target_value),
            data_path=plan.source.uri,
            cv_folds=plan.split.folds,
            cv_strategy=cv_strategy,
            models=plan.models,
        )

    def submit(self, plan: RunPlan) -> RunHandle:
        """Run a compiled plan as the single orchestration entry point."""
        if plan.source.source_type == "memory":
            raise ValueError(
                "submit requires a file-backed source; use run() for in-memory frames"
            )
        config = self._config_from_plan(plan)
        frame = load_data(plan.source.uri)
        execute_training = bool(plan.documentation.get("execute_training", True))
        result = self.run(
            config,
            frame,
            plan=plan,
            execute_training=execute_training,
        )
        return RunHandle(run_id=result.run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        path = self.workspace.runs_dir / run_id / "run.json"
        if not path.exists():
            return {"run_id": run_id, "status": "unknown"}
        data = json.loads(path.read_text(encoding="utf-8"))
        data["cancellation_requested"] = (path.parent / "CANCEL_REQUESTED").exists()
        return data  # type: ignore[no-any-return]

    def cancel(self, run_id: str) -> None:
        path = self.workspace.runs_dir / run_id / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") in {"succeeded", "partial", "failed", "cancelled"}:
            return
        (path.parent / "CANCEL_REQUESTED").touch(exist_ok=True)

    def resume(self, run_id: str) -> ExecutionResult:
        """Return a verified completed run without trusting event history alone."""
        path = self._manifest_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.status != "succeeded":
            raise RuntimeError(
                "Only a run with a committed terminal manifest can be resumed; "
                f"{run_id} is {manifest.status}."
            )
        products: list[str] = []
        for stage in manifest.stages:
            for product_id in stage.output_products:
                verification = self.store.verify(product_id, deep=True)
                if not verification["ok"]:
                    raise ValueError(f"Committed checkpoint failed verification: {product_id}")
                products.append(product_id)
        return ExecutionResult(run_id, manifest.status, path, tuple(products))

    def _stage(self, manifest: RunManifest, sink: JsonlEventSink, name: str, operation: Any) -> Any:
        if self._cancellation_path(manifest.run_id).exists():
            raise CancellationRequested(f"cancellation requested for {manifest.run_id}")
        stage = next(item for item in manifest.stages if item.name == name)
        configured_attempts = manifest.plan.retry.get("max_attempts", 1)
        attempts = configured_attempts if isinstance(configured_attempts, int) else 1
        for attempt in range(1, max(attempts, 1) + 1):
            stage.status = "running"
            stage.attempt = attempt
            stage.started_at = datetime.now(UTC)
            stage.ended_at = None
            stage.error = None
            self._save_manifest(manifest)
            self._event(sink, manifest, "stage.started", name, {"attempt": attempt})
            try:
                result = operation()
                stage.status = "succeeded"
                stage.ended_at = datetime.now(UTC)
                self._save_manifest(manifest)
                self._event(sink, manifest, "stage.succeeded", name, {"attempt": attempt})
                return result
            except Exception as exc:
                stage.status = "failed"
                stage.ended_at = datetime.now(UTC)
                stage.error = {"type": type(exc).__name__, "message": str(exc)}
                self._save_manifest(manifest)
                self._event(
                    sink,
                    manifest,
                    "stage.failed",
                    name,
                    {"attempt": attempt, **stage.error},
                )
                if attempt >= max(attempts, 1) or not _is_retryable(exc):
                    raise
                backoff = manifest.plan.retry.get("backoff_seconds", 0.5)
                if isinstance(backoff, int | float) and backoff > 0:
                    sleep(backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"stage did not complete: {name}")

    def _record_stage(
        self,
        manifest: RunManifest,
        name: str,
        product: Any,
    ) -> None:
        stage = next(item for item in manifest.stages if item.name == name)
        stage.input_products = list(product.inputs)
        stage.output_products = [product.product_id]
        manifest.artifacts.extend(product.artifacts)
        self._save_manifest(manifest)

    def _read_artifact(self, uri: str) -> pl.DataFrame:
        product_id, relative = uri.split("/", 1)
        path = self.workspace.root / "lake"
        match = next(path.glob(f"*/**/{product_id}/{relative}"), None)
        if match is None:
            raise FileNotFoundError(uri)
        return pl.read_parquet(match)

    def _transition(self, manifest: RunManifest, status: str) -> RunManifest:
        manifest.status = status  # type: ignore[assignment]
        if status == "running":
            manifest.started_at = datetime.now(UTC)
        if status in {"succeeded", "failed", "partial", "cancelled"}:
            manifest.ended_at = datetime.now(UTC)
        self._save_manifest(manifest)
        return manifest

    def _event(
        self,
        sink: JsonlEventSink,
        manifest: RunManifest,
        event_type: str,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        sink.append(
            EventEnvelope(
                run_id=manifest.run_id,
                event_type=event_type,
                stage=stage,
                attempt=int(payload.get("attempt", 1)),
                graph_version=manifest.graph_version,
                payload=payload,
            )
        )

    def _complete_publish(
        self, manifest: RunManifest, sink: JsonlEventSink, products: list[str]
    ) -> None:
        stage = next(item for item in manifest.stages if item.name == "publish")
        stage.status = "running"
        stage.attempt = 1
        stage.started_at = datetime.now(UTC)
        stage.input_products = list(products)
        self._save_manifest(manifest)
        self._event(sink, manifest, "stage.started", "publish", {"attempt": 1})
        stage.status = "succeeded"
        stage.ended_at = datetime.now(UTC)
        self._save_manifest(manifest)
        self._event(sink, manifest, "stage.succeeded", "publish", {"attempt": 1})

    def _find_successful_run(self, run_key: str) -> RunManifest | None:
        for path in sorted(self.workspace.runs_dir.glob("*/run.json"), reverse=True):
            try:
                manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if manifest.run_key == run_key and manifest.status == "succeeded":
                return manifest
        return None

    def _allocate_run_id(self, run_key: str) -> str:
        candidate = make_run_id(run_key)
        if not self._manifest_path(candidate).exists():
            return candidate
        timestamp = candidate.rsplit("_", 1)[0]
        counter = 1
        while True:
            suffix = digest([run_key, counter])[7:19]
            candidate = f"{timestamp}_{suffix}"
            if not self._manifest_path(candidate).exists():
                return candidate
            counter += 1

    def _manifest_path(self, run_id: str) -> Path:
        return self.workspace.runs_dir / run_id / "run.json"

    def _cancellation_path(self, run_id: str) -> Path:
        return self.workspace.runs_dir / run_id / "CANCEL_REQUESTED"

    def _save_manifest(self, manifest: RunManifest) -> None:
        path = self._manifest_path(manifest.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary.replace(path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _is_retryable(error: Exception) -> bool:
    return isinstance(error, (BlockingIOError, ConnectionError, TimeoutError))
