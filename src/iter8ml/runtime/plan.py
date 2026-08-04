"""Compile the legacy ExperimentConfig into a complete medallion plan."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from iter8ml.config import CVStrategy, ExperimentConfig, StepName
from iter8ml.domain.manifests import RunPlan, SourceSpec, SplitSpec


def compile_run_plan(
    config: ExperimentConfig,
    *,
    source_name: str | None = None,
    materialization: str = "reproducible",
) -> RunPlan:
    source_path = Path(config.data_path) if config.data_path else None
    source_type: Literal["csv", "parquet", "sqlite", "memory"] = "memory"
    if source_path:
        source_type = cast(
            Literal["csv", "parquet", "sqlite", "memory"],
            {
                ".csv": "csv",
                ".parquet": "parquet",
                ".db": "sqlite",
                ".sqlite": "sqlite",
            }.get(source_path.suffix.lower(), "memory"),
        )
    if config.cv_strategy == CVStrategy.STRATIFIED:
        strategy: Literal["kfold", "stratified", "group", "time", "purged_time"] = "stratified"
    elif config.cv_strategy == CVStrategy.TIMESERIES:
        strategy = "time"
    else:
        strategy = "kfold"
    split = SplitSpec(
        strategy=strategy,
        folds=config.cv_folds,
        shuffle=strategy == "stratified",
        random_seed=config.random_seed,
    )
    source = SourceSpec(
        name=source_name or _safe_name(config.name),
        source_type=source_type,
        uri=config.data_path or "memory://frame",
    )
    return RunPlan(
        plan_name=config.name,
        materialization=materialization,  # type: ignore[arg-type]
        source=source,
        contract={"schema_version": 1, "target_col": config.target_col},
        target={"column": config.target_col, "task": config.task.value},
        split=split,
        features={
            "strategy": config.pipeline.step_params(StepName.FEATURE_ENGINEERING).get(
                "strategy", "none"
            )
        },
        models=config.models,
        evaluation={"metrics": config.metrics, "cv_folds": config.cv_folds},
        resources={
            "max_workers": config.max_workers,
            "strict_thread_safety": config.strict_thread_safety,
        },
        retry={"max_attempts": 1, "backoff_seconds": 0.5},
        failure={"allow_partial": True, "fail_fast_quality": True},
        promotion={"enabled": False},
        documentation={"export": True},
    )


def _safe_name(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isascii() and char.isalnum() else "_" for char in value
    ).strip("_")
    if not normalized or not normalized[0].isalpha():
        normalized = f"experiment_{normalized}".rstrip("_")
    return normalized[:64]
