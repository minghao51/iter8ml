"""Silver canonical data and quality contract."""

from __future__ import annotations

from collections.abc import Mapping
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
    contract: Mapping[str, object] | None = None,
) -> ProductManifest:
    """Validate the canonical frame without applying learned transformations."""
    if target_col and target_col not in frame.columns:
        raise ValueError(f"target_col '{target_col}' is not present in the Silver input")
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Silver schema contains duplicate column names")
    canonical = frame.clone()
    resolved_contract = contract or {"target_col": target_col}
    contract_checks = _validate_contract(canonical, resolved_contract)
    contract_digest = digest(resolved_contract)
    pid = product_id("silver", bronze.name, bronze.product_id, contract_digest)
    if store.exists(pid):
        return store.read_verified_manifest(pid)
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
                "contract_checks": contract_checks,
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
            quality_summary={
                "valid": True,
                "row_count": canonical.height,
                "contract_checks": len(contract_checks),
            },
            lineage=[LineageEdge(upstream=bronze.product_id, downstream=pid)],
        )
        return writer.commit(manifest)
    except BaseException:
        writer.abort()
        raise


def _validate_contract(frame: pl.DataFrame, contract: Mapping[str, object]) -> list[str]:
    checks: list[str] = []
    required = contract.get("required_columns", {})
    if not isinstance(required, dict):
        raise ValueError("Silver contract required_columns must be a mapping")
    if not all(
        isinstance(column, str) and isinstance(dtype, str) for column, dtype in required.items()
    ):
        raise ValueError("Silver contract required_columns must map names to dtype strings")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Silver contract missing required columns: {missing}")
    for column, expected_dtype in required.items():
        observed_dtype = str(frame.schema[column]).lower()
        if observed_dtype != expected_dtype.lower():
            raise ValueError(
                f"Silver contract dtype mismatch for '{column}': "
                f"expected {expected_dtype}, observed {frame.schema[column]}"
            )
        checks.append(f"dtype:{column}")

    unique = contract.get("unique", [])
    if not isinstance(unique, list) or not all(isinstance(column, str) for column in unique):
        raise ValueError("Silver contract unique must be a list of column names")
    missing_unique = sorted(set(unique) - set(frame.columns))
    if missing_unique:
        raise ValueError(f"Silver contract unique columns are missing: {missing_unique}")
    if unique and frame.n_unique(subset=unique) != frame.height:
        raise ValueError(f"Silver contract uniqueness failed for columns: {unique}")
    if unique:
        checks.append("unique:" + ",".join(unique))

    null_thresholds = contract.get("null_thresholds", {})
    if not isinstance(null_thresholds, dict):
        raise ValueError("Silver contract null_thresholds must be a mapping")
    default_threshold = null_thresholds.get("default")
    for column in frame.columns:
        threshold = null_thresholds.get(column, default_threshold)
        if threshold is None:
            continue
        if not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
            raise ValueError(f"Silver null threshold for '{column}' must be between 0 and 1")
        null_ratio = frame[column].null_count() / max(frame.height, 1)
        if null_ratio > float(threshold):
            raise ValueError(
                f"Silver contract null threshold exceeded for '{column}': "
                f"{null_ratio:.6f} > {float(threshold):.6f}"
            )
        checks.append(f"nulls:{column}")
    return checks
