"""Shared model class factory with plugin discovery."""

import importlib

from iter8ml.config import HardwareProfile

_BUILT_IN_REGISTRY: dict[str, tuple[str, str]] = {
    "catboost": ("iter8ml.engine.models.catboost_model", "CatBoostModel"),
    "lightgbm": ("iter8ml.engine.models.lightgbm_model", "LightGBMModel"),
    "xgboost": ("iter8ml.engine.models.xgboost_model", "XGBoostModel"),
    "tabpfn": ("iter8ml.engine.models.tabpfn_model", "TabPFNModel"),
    "ft_transformer": ("iter8ml.engine.models.ft_transformer", "FTTransformerModel"),
    "tabnet": ("iter8ml.engine.models.tabnet_model", "TabNetModel"),
    "naive_baseline": ("iter8ml.engine.models.baselines", "NaiveBaseline"),
    "linear_baseline": ("iter8ml.engine.models.baselines", "LinearBaseline"),
}

_MODEL_CLASS_CACHE: dict[str, type] = {}


def _discover_models() -> dict[str, tuple[str, str]]:
    """Merge built-in registry with externally registered plugins."""
    registry = dict(_BUILT_IN_REGISTRY)
    from importlib.metadata import entry_points

    try:
        for ep in entry_points(group="iter8ml.models"):
            if ep.name not in registry:
                module, attr = ep.value.rsplit(":", 1)
                registry[ep.name] = (module, attr)
    except (TypeError, ImportError):
        pass

    return registry


_MODEL_REGISTRY = _discover_models()


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
    # Cap OpenMP threads before any GBDT module import can load libgomp
    # (ADR-0004/0006): the factory is the single seam every model import
    # passes through — Trainer, DAG nodes, HPO, MCP — so the cap is safe by
    # default here. Idempotent/re-entrant; Trainer.__init__ re-applies it.
    HardwareProfile.configure_omp_threads()

    validate_model_name(model_name)

    if model_name in _MODEL_CLASS_CACHE:
        return _MODEL_CLASS_CACHE[model_name]

    module_path, class_name = _MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _MODEL_CLASS_CACHE[model_name] = cls
    return cls  # type: ignore[no-any-return]
