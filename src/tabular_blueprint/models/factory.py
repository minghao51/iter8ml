"""Shared model class factory."""

import importlib

_MODEL_REGISTRY = {
    "catboost": ("tabular_blueprint.models.conventional.catboost_model", "CatBoostModel"),
    "lightgbm": ("tabular_blueprint.models.conventional.lightgbm_model", "LightGBMModel"),
    "xgboost": ("tabular_blueprint.models.conventional.xgboost_model", "XGBoostModel"),
    "tabpfn": ("tabular_blueprint.models.tabular_foundation.tabpfn_model", "TabPFNModel"),
    "ft_transformer": ("tabular_blueprint.models.deep.ft_transformer", "FTTransformerModel"),
    "tabnet": ("tabular_blueprint.models.deep.tabnet_model", "TabNetModel"),
    "naive_baseline": ("tabular_blueprint.models.baselines", "NaiveBaseline"),
    "linear_baseline": ("tabular_blueprint.models.baselines", "LinearBaseline"),
}
_MODEL_CLASS_CACHE: dict[str, type] = {}


def available_model_names() -> list[str]:
    """Return the supported model names."""
    return sorted(_MODEL_REGISTRY)


def validate_model_name(model_name: str) -> str:
    """Validate a model name and return it unchanged."""
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(available_model_names())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    return model_name


def get_model_class(model_name: str) -> type:
    """Resolve a model class by name with lazy imports."""
    validate_model_name(model_name)

    if model_name in _MODEL_CLASS_CACHE:
        return _MODEL_CLASS_CACHE[model_name]

    module_path, class_name = _MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _MODEL_CLASS_CACHE[model_name] = cls
    return cls  # type: ignore[no-any-return]
