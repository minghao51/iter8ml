"""Bronze source snapshots."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from iter8ml.domain.hashing import dataframe_digest, digest
from iter8ml.domain.ids import product_id
from iter8ml.domain.manifests import ProductManifest, SourceSpec
from iter8ml.storage.local import LocalArtifactStore


def materialize_bronze(
    frame: pl.DataFrame,
    source: SourceSpec,
    store: LocalArtifactStore,
    *,
    specification: dict[str, object] | None = None,
) -> ProductManifest:
    """Persist an immutable source snapshot and its observed source contract."""
    source_fingerprint = source.fingerprint or dataframe_digest(frame)
    spec_digest = digest(specification or {})
    pid = product_id("bronze", source.name, source_fingerprint, spec_digest)
    if store.exists(pid):
        return store.read_manifest(pid)
    writer = store.begin(pid, product_type="bronze", name=source.name)
    try:
        data_ref = writer.write_parquet(frame, relative_path="data/data.parquet", kind="dataset")
        manifest = ProductManifest(
            product_id=pid,
            product_type="bronze",
            name=source.name,
            created_at=datetime.now(UTC),
            inputs=[],
            specification_digest=spec_digest,
            code_digest=digest("iter8ml.dataflows.bronze.v1"),
            graph_version=digest("iter8ml.medallion.graph.v1"),
            artifacts=[data_ref],
            schema_digest=dataframe_digest(frame),
            quality_summary={
                "readable": True,
                "row_count": frame.height,
                "column_count": frame.width,
            },
            lineage=[],
            metadata={"source_type": source.source_type, "source_uri": source.uri},
        )
        return writer.commit(manifest)
    except BaseException:
        writer.abort()
        raise
