"""Split-aware Gold features, labels, and fold membership."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from iter8ml.domain.hashing import digest
from iter8ml.domain.ids import product_id
from iter8ml.domain.manifests import LineageEdge, ProductManifest, SplitManifest, SplitSpec
from iter8ml.storage.local import LocalArtifactStore
from iter8ml.verification.split_validation import validate_split


def _row_ids(frame: pl.DataFrame) -> list[str]:
    occurrences: dict[str, int] = {}
    row_ids: list[str] = []
    for row in frame.iter_rows(named=True):
        row_key = digest(row)
        occurrence = occurrences.get(row_key, 0)
        occurrences[row_key] = occurrence + 1
        row_ids.append(digest({"row": row_key, "occurrence": occurrence})[7:])
    return row_ids


def build_split_frame(frame: pl.DataFrame, target_col: str, spec: SplitSpec) -> pl.DataFrame:
    if target_col not in frame.columns:
        raise ValueError(f"target_col '{target_col}' is not present")
    row_ids = _row_ids(frame)
    return _build_split_frame(frame, target_col, spec, row_ids)


def _build_split_frame(
    frame: pl.DataFrame, target_col: str, spec: SplitSpec, row_ids: list[str]
) -> pl.DataFrame:
    indices = np.array(sorted(range(frame.height), key=row_ids.__getitem__))
    y = frame[target_col].to_numpy()
    splitter: Any
    if spec.strategy == "stratified":
        splitter = StratifiedKFold(
            spec.folds,
            shuffle=spec.shuffle,
            random_state=spec.random_seed if spec.shuffle else None,
        )
        splits = (
            (indices[train], indices[validation])
            for train, validation in splitter.split(indices, y[indices])
        )
    elif spec.strategy == "group":
        if not spec.group_column or spec.group_column not in frame.columns:
            raise ValueError("group split requires an existing group_column")
        splitter = GroupKFold(spec.folds)
        groups = frame[spec.group_column].to_numpy()
        splits = (
            (indices[train], indices[validation])
            for train, validation in splitter.split(indices, y[indices], groups[indices])
        )
    elif spec.strategy in {"time", "purged_time"}:
        if not spec.time_column or spec.time_column not in frame.columns:
            raise ValueError("time split requires an existing time_column")
        time_values = frame[spec.time_column].to_list()
        if any(value is None for value in time_values):
            raise ValueError("time split does not allow null values in time_column")
        order = np.array(
            sorted(range(frame.height), key=lambda index: (time_values[index], row_ids[index]))
        )
        splitter = TimeSeriesSplit(spec.folds, gap=spec.gap)
        splits = ((order[train], order[test]) for train, test in splitter.split(order))
    else:
        splitter = KFold(
            spec.folds,
            shuffle=spec.shuffle,
            random_state=spec.random_seed if spec.shuffle else None,
        )
        splits = (
            (indices[train], indices[validation]) for train, validation in splitter.split(indices)
        )

    rows: list[dict[str, object]] = []
    for fold, (train_indices, validation_indices) in enumerate(splits):
        validation_set = {int(i) for i in validation_indices}
        ordered_train = [int(i) for i in train_indices]
        if spec.strategy == "purged_time" and spec.embargo:
            ordered_train = ordered_train[: -spec.embargo]
        train_set = set(ordered_train)
        for index in sorted(train_set):
            rows.append(
                {
                    "row_id": row_ids[index],
                    "fold": fold,
                    "role": "train",
                    "repeat": 0,
                }
            )
        for index in sorted(validation_set):
            rows.append(
                {
                    "row_id": row_ids[index],
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
    feature_spec: Mapping[str, object] | None = None,
) -> tuple[ProductManifest, SplitManifest]:
    if target_col not in frame.columns:
        raise ValueError(f"target_col '{target_col}' is not present")
    spec = split_spec or SplitSpec(
        strategy="stratified" if frame[target_col].n_unique() <= 20 else "kfold"
    )
    feature_digest = digest(feature_spec or {})
    split_digest = digest(spec.model_dump(mode="json"))
    pid = product_id("gold", silver.name, silver.product_id, feature_digest, split_digest)
    if store.exists(pid):
        committed = store.read_verified_manifest(pid)
        split_ref = next(
            ref for ref in committed.artifacts if ref.uri.endswith("split_manifest.json")
        )
        with store.open_artifact(split_ref) as handle:
            split_manifest = SplitManifest.model_validate_json(handle.read())
        return committed, split_manifest
    row_id_values = _row_ids(frame)
    split_frame = _build_split_frame(frame, target_col, spec, row_id_values)
    split_result = validate_split(split_frame)
    if not split_result["ok"]:
        raise ValueError(f"Gold split validation failed: {split_result['errors']}")
    temporal_checks_passed = _validate_temporal_order(frame, split_frame, spec, row_id_values)
    writer = store.begin(pid, product_type="gold", name=silver.name)
    try:
        row_ids = pl.DataFrame({"row_id": row_id_values})
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
            temporal_checks_passed=temporal_checks_passed,
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


def _validate_temporal_order(
    frame: pl.DataFrame, split_frame: pl.DataFrame, spec: SplitSpec, row_ids: list[str]
) -> bool | None:
    if spec.strategy not in {"time", "purged_time"}:
        return None
    assert spec.time_column is not None
    row_times = dict(zip(row_ids, frame[spec.time_column].to_list(), strict=True))
    for fold in split_frame["fold"].unique().to_list():
        fold_frame = split_frame.filter(pl.col("fold") == fold)
        train_times = [
            row_times[row_id]
            for row_id in fold_frame.filter(pl.col("role") == "train")["row_id"].to_list()
        ]
        validation_times = [
            row_times[row_id]
            for row_id in fold_frame.filter(pl.col("role") == "validation")["row_id"].to_list()
        ]
        if max(train_times) >= min(validation_times):
            raise ValueError(f"Gold temporal split ordering failed for fold {fold}")
    return True
