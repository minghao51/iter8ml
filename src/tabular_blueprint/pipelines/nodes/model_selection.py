from __future__ import annotations

from typing import Any


def models_to_run(
    data_prep_result: object,
    task: str,
    vram_gb: float,
    config_models: Any,
    include_baselines: bool = True,
) -> list[str]:
    if isinstance(config_models, list):
        return config_models
    if config_models != "auto":
        return [config_models]

    from tabular_blueprint.models.selector import ModelSelector

    selector = ModelSelector()
    return selector.select(
        n_rows=data_prep_result.n_rows,
        task=task,
        vram_gb=vram_gb,
        include_baselines=include_baselines,
    )
