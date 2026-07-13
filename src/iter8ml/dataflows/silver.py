"""Silver canonical data and quality contract."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from iter8ml.domain.hashing import dataframe_digest, digest
from iter8ml.domain.ids import product_id
from iter8ml.domain.manifests import LineageEdge, ProductManifest
from iter8ml.storage.local import LocalArtifactStore


def materialize_silver(
    frame: pl.DataFrame,
    bronze: ProductManifest,
    store: LocalArtifactStore,
    *,
    target_col: str | None = None,
    contract: dict[str, object] | None = None,
) -> ProductManifest:
    """Validate the canonical frame without applying learned transformations."""
    if target_col and target_col not in frame.columns:
        raise ValueError(f"target_col '{target_col}' is not present in the Silver input")
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Silver schema contains duplicate column names")
    canonical = frame.clone()
    contract_digest = digest(contract or {"target_col": target_col})
    pid = product_id("silver", bronze.name, bronze.product_id, contract_digest)
    if store.exists(pid):
        return store.read_manifest(pid)
    writer = store.begin(pid, product_type="silver", name=bronze.name)
    try:
        data_ref = writer.write_parquet(
            canonical, relative_path="data/data.parquet", kind="dataset"
        )
        quality_ref = writer.write_json(
            {
                "schema_version": 1,
                "row_count": canonical.height,
                "column_count": canonical.width,
                "null_counts": {name: canonical[name].null_count() for name in canonical.columns},
                "target_present": target_col is None or target_col in canonical.columns,
            },
            relative_path="quality.json",
            kind="quality",
        )
        manifest = ProductManifest(
            product_id=pid,
            product_type="silver",
            name=bronze.name,
            created_at=datetime.now(UTC),
            inputs=[bronze.product_id],
            specification_digest=contract_digest,
            code_digest=digest("iter8ml.dataflows.silver.v1"),
            graph_version=bronze.graph_version,
            artifacts=[data_ref, quality_ref],
            schema_digest=dataframe_digest(canonical),
            quality_summary={"valid": True, "row_count": canonical.height},
            lineage=[LineageEdge(upstream=bronze.product_id, downstream=pid)],
        )
        return writer.commit(manifest)
    except BaseException:
        writer.abort()
        raise
