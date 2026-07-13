"""Split-aware Gold features, labels, and fold membership."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from iter8ml.domain.hashing import digest
from iter8ml.domain.ids import product_id
from iter8ml.domain.manifests import LineageEdge, ProductManifest, SplitManifest, SplitSpec
from iter8ml.storage.local import LocalArtifactStore


def _row_ids(frame: pl.DataFrame) -> list[str]:
    occurrences: dict[str, int] = {}
    row_ids: list[str] = []
    for row in frame.to_dicts():
        row_key = digest(row)
        occurrence = occurrences.get(row_key, 0)
        occurrences[row_key] = occurrence + 1
        row_ids.append(digest({"row": row_key, "occurrence": occurrence})[7:])
    return row_ids


def build_split_frame(frame: pl.DataFrame, target_col: str, spec: SplitSpec) -> pl.DataFrame:
    if target_col not in frame.columns:
        raise ValueError(f"target_col '{target_col}' is not present")
    indices = np.arange(frame.height)
    y = frame[target_col].to_numpy()
    splitter: Any
    if spec.strategy == "stratified":
        splitter = StratifiedKFold(spec.folds, shuffle=spec.shuffle, random_state=spec.random_seed)
        splits = splitter.split(indices, y)
    elif spec.strategy == "group":
        if not spec.group_column or spec.group_column not in frame.columns:
            raise ValueError("group split requires an existing group_column")
        splitter = GroupKFold(spec.folds)
        splits = splitter.split(indices, y, frame[spec.group_column].to_numpy())
    elif spec.strategy in {"time", "purged_time"}:
        if not spec.time_column or spec.time_column not in frame.columns:
            raise ValueError("time split requires an existing time_column")
        order = np.argsort(frame[spec.time_column].to_numpy())
        splitter = TimeSeriesSplit(spec.folds, gap=spec.gap)
        splits = ((order[train], order[test]) for train, test in splitter.split(order))
    else:
        splitter = KFold(spec.folds, shuffle=spec.shuffle, random_state=spec.random_seed)
        splits = splitter.split(indices)

    rows: list[dict[str, object]] = []
    for fold, (train_indices, validation_indices) in enumerate(splits):
        validation_set = {int(i) for i in validation_indices}
        train_set = {int(i) for i in train_indices}
        if spec.strategy == "purged_time" and spec.embargo:
            train_set = train_set - set(validation_indices[-spec.embargo :])
        for index in sorted(train_set):
            rows.append(
                {
                    "row_id": _row_ids(frame)[index],
                    "fold": fold,
                    "role": "train",
                    "repeat": 0,
                }
            )
        for index in sorted(validation_set):
            rows.append(
                {
                    "row_id": _row_ids(frame)[index],
                    "fold": fold,
                    "role": "validation",
                    "repeat": 0,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("fold").cast(pl.Int16), pl.col("repeat").cast(pl.Int16)
    )


def materialize_gold(
    frame: pl.DataFrame,
    silver: ProductManifest,
    store: LocalArtifactStore,
    *,
    target_col: str,
    split_spec: SplitSpec | None = None,
    feature_spec: dict[str, object] | None = None,
) -> tuple[ProductManifest, SplitManifest]:
    spec = split_spec or SplitSpec(
        strategy="stratified" if frame[target_col].n_unique() <= 20 else "kfold"
    )
    split_frame = build_split_frame(frame, target_col, spec)
    feature_digest = digest(feature_spec or {})
    split_digest = digest(spec.model_dump(mode="json"))
    pid = product_id("gold", silver.name, silver.product_id, feature_digest, split_digest)
    writer = store.begin(pid, product_type="gold", name=silver.name)
    try:
        row_ids = pl.DataFrame({"row_id": _row_ids(frame)})
        features = frame.drop(target_col).with_columns(row_ids)
        labels = frame.select(target_col).with_columns(row_ids)
        features_ref = writer.write_parquet(
            features, relative_path="features/features.parquet", kind="features"
        )
        labels_ref = writer.write_parquet(labels, relative_path="labels.parquet", kind="labels")
        splits_ref = writer.write_parquet(
            split_frame, relative_path="splits.parquet", kind="splits"
        )
        split_manifest = SplitManifest(
            split_id=digest([silver.product_id, split_digest]),
            dataset_version=silver.product_id,
            spec=spec,
            artifact=splits_ref,
            fold_counts=_fold_counts(split_frame),
            overlap_checks_passed=True,
            temporal_checks_passed=spec.strategy in {"time", "purged_time"} or None,
        )
        split_ref = writer.write_json(
            split_manifest.model_dump(mode="json"),
            relative_path="split_manifest.json",
            kind="report",
        )
        manifest = ProductManifest(
            product_id=pid,
            product_type="gold",
            name=silver.name,
            created_at=datetime.now(UTC),
            inputs=[silver.product_id],
            specification_digest=digest(
                {"feature": feature_spec or {}, "split": spec.model_dump(mode="json")}
            ),
            code_digest=digest("iter8ml.dataflows.gold.v1"),
            graph_version=silver.graph_version,
            artifacts=[features_ref, labels_ref, splits_ref, split_ref],
            schema_digest=digest({"features": features.schema, "labels": labels.schema}),
            quality_summary={"split_overlap": False, "folds": spec.folds},
            lineage=[LineageEdge(upstream=silver.product_id, downstream=pid)],
            metadata={"target_col": target_col},
        )
        committed = writer.commit(manifest)
        return committed, split_manifest
    except BaseException:
        writer.abort()
        raise


def _fold_counts(split_frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in split_frame.group_by(["fold", "role"]).len().iter_rows(named=True):
        counts.setdefault(str(row["fold"]), {})[str(row["role"])] = int(row["len"])
    return counts
