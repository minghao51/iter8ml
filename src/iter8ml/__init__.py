"""iter8ml — a high-velocity iteration framework for tabular ML."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

try:
    __version__ = _version("iter8ml")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Eager, lightweight core (cheap imports, no heavy ML/training graph).
from iter8ml.config import ExperimentConfig, HardwareProfile
from iter8ml.constants import CVStrategy, EmbeddingMethod, TaskType, TrackerType
from iter8ml.data.loader import load_data
from iter8ml.exceptions import (
    ArtifactError,
    DataLoadError,
    HamiltonUnavailableError,
    Iter8MLError,
    ModelFitError,
    RegistryError,
    TabularBlueprintError,
)
from iter8ml.workspace import Workspace

# DAG-internal config primitives (PipelineSpec, PipelineStep, StepName) are
# intentionally NOT exposed at the top level — import them from iter8ml.config.

_LAZY: dict[str, str] = {
    "Evaluator": "iter8ml.engine.evaluator",
    "Trainer": "iter8ml.engine.trainer",
    "ModelSelector": "iter8ml.engine.models.selector",
    "available_model_names": "iter8ml.engine.models.factory",
    "get_model_class": "iter8ml.engine.models.factory",
    "Tracker": "iter8ml.engine.tracker",
    "JSONLTracker": "iter8ml.engine.tracker",
    "MedallionExecutionService": "iter8ml.orchestration",
    "compile_run_plan": "iter8ml.runtime",
    "ExportService": "iter8ml.services.export",
    "RegistryService": "iter8ml.services.registry",
    "PromotionResult": "iter8ml.services.registry",
    "ReportService": "iter8ml.services.reporting",
    "ExperimentSession": "iter8ml.session",
}

__all__ = [
    "ArtifactError",
    "CVStrategy",
    "DataLoadError",
    "EmbeddingMethod",
    "Evaluator",
    "ExperimentConfig",
    "ExperimentSession",
    "ExportService",
    "HamiltonUnavailableError",
    "HardwareProfile",
    "Iter8MLError",
    "JSONLTracker",
    "MedallionExecutionService",
    "ModelFitError",
    "ModelSelector",
    "PromotionResult",
    "RegistryError",
    "RegistryService",
    "ReportService",
    "TabularBlueprintError",
    "TaskType",
    "Tracker",
    "TrackerType",
    "Trainer",
    "Workspace",
    "available_model_names",
    "compile_run_plan",
    "get_model_class",
    "load_data",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from iter8ml.engine.evaluator import Evaluator
    from iter8ml.engine.models.factory import available_model_names, get_model_class
    from iter8ml.engine.models.selector import ModelSelector
    from iter8ml.engine.tracker import JSONLTracker, Tracker
    from iter8ml.engine.trainer import Trainer
    from iter8ml.orchestration import MedallionExecutionService
    from iter8ml.runtime import compile_run_plan
    from iter8ml.services.export import ExportService
    from iter8ml.services.registry import PromotionResult, RegistryService
    from iter8ml.services.reporting import ReportService
    from iter8ml.session import ExperimentSession
