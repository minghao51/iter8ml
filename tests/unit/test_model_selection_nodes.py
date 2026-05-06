from tabular_blueprint.pipelines.nodes.model_selection import models_to_run


class _Prep:
    n_rows = 1_000


def test_models_to_run_filters_completed_for_explicit_list():
    selected = models_to_run(
        data_prep_result=_Prep(),
        task="classification",
        vram_gb=0.0,
        config_models=["catboost", "lightgbm"],
        completed_models=["catboost"],
    )
    assert selected == ["lightgbm"]


def test_models_to_run_filters_completed_for_auto():
    selected = models_to_run(
        data_prep_result=_Prep(),
        task="classification",
        vram_gb=0.0,
        config_models="auto",
        completed_models=["naive_baseline"],
    )
    assert "naive_baseline" not in selected
