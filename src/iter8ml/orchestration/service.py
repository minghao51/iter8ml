"""Local run lifecycle around Bronze -> Silver -> Gold -> Platinum."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from iter8ml.config import ExperimentConfig
from iter8ml.dataflows.bronze import materialize_bronze
from iter8ml.dataflows.gold import materialize_gold
from iter8ml.dataflows.platinum_train import materialize_platinum
from iter8ml.dataflows.silver import materialize_silver
from iter8ml.domain.events import EventEnvelope, JsonlEventSink
from iter8ml.domain.hashing import digest
from iter8ml.domain.ids import run_id as make_run_id
from iter8ml.domain.manifests import RunManifest, RunPlan, StageRecord
from iter8ml.runtime.plan import compile_run_plan
from iter8ml.storage.catalog import LocalCatalogStore
from iter8ml.storage.local import LocalArtifactStore
from iter8ml.verification.leakage import validate_split_frame
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
        run_id = make_run_id(resolved_plan.run_key)
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
        event_sink = JsonlEventSink(self.workspace.events_dir, run_id)
        products: list[str] = []
        self._save_manifest(manifest)
        try:
            manifest = self._transition(manifest, "running")
            self._event(event_sink, run_id, "run.started", "run", {"plan": resolved_plan.plan_name})

            bronze = self._stage(
                manifest,
                event_sink,
                "bronze",
                lambda: materialize_bronze(frame, resolved_plan.source, self.store),
            )
            products.append(bronze.product_id)
            self._record_stage(manifest, "bronze", bronze.product_id, bronze.artifacts)
            self.catalog.register_product(bronze)
            silver = self._stage(
                manifest,
                event_sink,
                "silver",
                lambda: materialize_silver(frame, bronze, self.store, target_col=config.target_col),
            )
            products.append(silver.product_id)
            self._record_stage(manifest, "silver", silver.product_id, silver.artifacts)
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
                ),
            )
            split_frame = self._read_artifact(split.artifact.uri)
            split_result = validate_split_frame(split_frame)
            if not split_result["ok"]:
                raise ValueError(f"Gold leakage gate failed: {split_result['errors']}")
            products.append(gold.product_id)
            self._record_stage(manifest, "gold", gold.product_id, gold.artifacts)
            self.catalog.register_product(gold)

            results: dict[str, Any] = {}
            if execute_training:
                from iter8ml.engine.trainer import Trainer

                results = Trainer(config=config, workspace=self.workspace).run(frame)
            platinum = self._stage(
                manifest,
                event_sink,
                "platinum",
                lambda: materialize_platinum(
                    run_id, gold, results, self.store, experiment_name=config.name
                ),
            )
            products.append(platinum.product_id)
            self._record_stage(manifest, "platinum", platinum.product_id, platinum.artifacts)
            self.catalog.register_product(platinum)
            manifest = self._transition(manifest, "succeeded")
            self._event(event_sink, run_id, "run.succeeded", "publish", {"products": products})
        except BaseException as exc:
            manifest.status = "failed"
            manifest.error = {"type": type(exc).__name__, "message": str(exc)}
            manifest.ended_at = datetime.now(UTC)
            self._save_manifest(manifest)
            self._event(event_sink, run_id, "run.failed", "publish", manifest.error)
            raise
        finally:
            archive = event_sink.finalize(manifest.status)
            manifest.event_archive = str(archive.relative_to(self.workspace.root))
            self._save_manifest(manifest)
            self.catalog.register_run(manifest)
        return ExecutionResult(
            run_id, manifest.status, self._manifest_path(run_id), tuple(products)
        )

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
        stage = next(item for item in manifest.stages if item.name == name)
        configured_attempts = manifest.plan.retry.get("max_attempts", 1)
        attempts = configured_attempts if isinstance(configured_attempts, int) else 1
        for attempt in range(1, max(attempts, 1) + 1):
            stage.status = "running"
            stage.attempt = attempt
            stage.started_at = datetime.now(UTC)
            self._save_manifest(manifest)
            self._event(sink, manifest.run_id, "stage.started", name, {"attempt": attempt})
            try:
                result = operation()
                stage.status = "succeeded"
                stage.ended_at = datetime.now(UTC)
                self._save_manifest(manifest)
                self._event(sink, manifest.run_id, "stage.succeeded", name, {"attempt": attempt})
                return result
            except BaseException as exc:
                stage.status = "failed"
                stage.ended_at = datetime.now(UTC)
                stage.error = {"type": type(exc).__name__, "message": str(exc)}
                self._save_manifest(manifest)
                self._event(
                    sink,
                    manifest.run_id,
                    "stage.failed",
                    name,
                    {"attempt": attempt, **stage.error},
                )
                if attempt >= max(attempts, 1):
                    raise
        raise RuntimeError(f"stage did not complete: {name}")

    def _record_stage(
        self,
        manifest: RunManifest,
        name: str,
        product_id: str,
        artifacts: list[Any],
    ) -> None:
        stage = next(item for item in manifest.stages if item.name == name)
        stage.output_products = [product_id]
        manifest.artifacts.extend(artifacts)
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
        run_id: str,
        event_type: str,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        sink.append(
            EventEnvelope(run_id=run_id, event_type=event_type, stage=stage, payload=payload)
        )

    def _manifest_path(self, run_id: str) -> Path:
        return self.workspace.runs_dir / run_id / "run.json"

    def _save_manifest(self, manifest: RunManifest) -> None:
        path = self._manifest_path(manifest.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
