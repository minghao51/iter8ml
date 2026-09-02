"""Consistency tests: the engine training path consumes the medallion split.

Topology A requires the engine's CV folds to be the medallion's ``SplitManifest``
rather than a recomputed (weaker) split. These tests verify the row-id alignment
and the full medallion -> Trainer -> training-DAG wiring.
"""

from __future__ import annotations

import polars as pl
import pytest

from iter8ml.config import ExperimentConfig
from iter8ml.constants import TaskType
from iter8ml.dataflows.gold import build_split_frame
from iter8ml.domain.hashing import row_ids as frame_row_ids
from iter8ml.domain.manifests import SplitSpec
from iter8ml.engine.pipelines.nodes.train import _fold_indices_from_split
from iter8ml.orchestration import MedallionExecutionService
from iter8ml.workspace import Workspace


def _assert_folds_match_gold(split_frame: pl.DataFrame, row_ids: list[str]) -> None:
    folds_sorted = sorted(set(split_frame["fold"].to_list()))
    fold_indices = _fold_indices_from_split(split_frame, row_ids)
    assert len(fold_indices) == len(folds_sorted)
    for k, fold in enumerate(folds_sorted):
        gold_train = set(
            split_frame.filter((pl.col("fold") == fold) & (pl.col("role") == "train"))[
                "row_id"
            ].to_list()
        )
        gold_val = set(
            split_frame.filter((pl.col("fold") == fold) & (pl.col("role") == "validation"))[
                "row_id"
            ].to_list()
        )
        eng_train = {row_ids[i] for i in fold_indices[k][0].tolist()}
        eng_val = {row_ids[i] for i in fold_indices[k][1].tolist()}
        assert gold_train == eng_train
        assert gold_val == eng_val


@pytest.mark.parametrize(
    "spec",
    [
        SplitSpec(strategy="stratified", folds=3, shuffle=True, random_seed=42),
        SplitSpec(strategy="kfold", folds=4, shuffle=False, random_seed=0),
        SplitSpec(
            strategy="group",
            folds=3,
            group_column="grp",
        ),
        SplitSpec(
            strategy="purged_time",
            folds=3,
            time_column="ts",
            embargo=2,
        ),
    ],
)
def test_fold_indices_match_gold_split(spec):
    if spec.strategy == "group":
        frame = pl.DataFrame(
            {
                "grp": ["a", "b", "c"] * 4 + ["a", "b", "c"] * 4,
                "f": list(range(24)),
                "target": [0, 1] * 12,
            }
        )
    elif spec.strategy == "purged_time":
        frame = pl.DataFrame(
            {
                "ts": list(range(24)),
                "f": list(range(24)),
                "target": [0, 1] * 12,
            }
        )
    else:
        frame = pl.DataFrame({"f": list(range(24)), "target": [0, 1] * 12})

    split_frame = build_split_frame(frame, "target", spec)
    row_ids = frame_row_ids(frame)
    _assert_folds_match_gold(split_frame, row_ids)


def test_medallion_run_consumes_gold_split_for_training(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    from iter8ml.engine.pipelines.nodes import train as train_nodes

    orig = train_nodes._fold_indices_from_split

    def spy(split_frame, row_ids):
        captured["split_frame"] = split_frame
        captured["row_ids"] = row_ids
        return orig(split_frame, row_ids)

    monkeypatch.setattr(train_nodes, "_fold_indices_from_split", spy)

    config = ExperimentConfig(
        name="split_run",
        task=TaskType.CLASSIFICATION,
        target_col="target",
        data_path="",
        models=["lightgbm"],
        cv_folds=3,
    )
    frame = pl.DataFrame({"f1": list(range(30)), "f2": list(range(30)), "target": [0, 1] * 15})

    service = MedallionExecutionService(Workspace(root=tmp_path))
    result = service.run(config, frame, execute_training=True)

    assert result.status == "succeeded"
    assert captured.get("split_frame") is not None
    _assert_folds_match_gold(
        captured["split_frame"],
        captured["row_ids"],  # type: ignore[arg-type]
    )
