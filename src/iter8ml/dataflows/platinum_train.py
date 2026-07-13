"""Platinum run evidence wrapper for legacy Trainer results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from iter8ml.domain.hashing import digest
from iter8ml.domain.ids import product_id
from iter8ml.domain.manifests import LineageEdge, ProductManifest
from iter8ml.storage.local import LocalArtifactStore


def materialize_platinum(
    run_id: str,
    gold: ProductManifest,
    results: dict[str, Any],
    store: LocalArtifactStore,
    *,
    experiment_name: str = "experiment",
) -> ProductManifest:
    pid = product_id("platinum", experiment_name, gold.product_id, run_id)
    if store.exists(pid):
        return store.read_manifest(pid)
    writer = store.begin(pid, product_type="platinum", name=experiment_name)
    try:
        metrics_ref = writer.write_json(results, relative_path="metrics.json", kind="metrics")
        manifest = ProductManifest(
            product_id=pid,
            product_type="platinum",
            name=experiment_name,
            created_at=datetime.now(UTC),
            inputs=[gold.product_id],
            specification_digest=digest({"run_id": run_id, "experiment": experiment_name}),
            code_digest=digest("iter8ml.dataflows.platinum_train.v1"),
            graph_version=gold.graph_version,
            artifacts=[metrics_ref],
            quality_summary={"has_results": bool(results)},
            lineage=[LineageEdge(upstream=gold.product_id, downstream=pid)],
            metadata={"run_id": run_id},
        )
        return writer.commit(manifest)
    except BaseException:
        writer.abort()
        raise
