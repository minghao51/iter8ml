from __future__ import annotations

from typing import Any


def models_to_run(
    data_prep_result: object,
    task: str,
    vram_gb: float,
    config_models: Any,
    completed_models: list[str] | None = None,
    include_baselines: bool = True,
) -> list[str]:
    completed = set(completed_models or [])

    if isinstance(config_models, list):
        return [m for m in config_models if m not in completed]
    if config_models != "auto":
        return [config_models] if config_models not in completed else []

    from tabular_blueprint.models.selector import ModelSelector

    selector = ModelSelector()
    selected = selector.select(
        n_rows=data_prep_result.n_rows,
        task=task,
        vram_gb=vram_gb,
        include_baselines=include_baselines,
    )
    return [m for m in selected if m not in completed]
