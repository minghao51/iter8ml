"""iter8ml — a high-velocity iteration framework for tabular ML."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("iter8ml")
except PackageNotFoundError:
    __version__ = "0.0.0"

from iter8ml.config import (
    ExperimentConfig,
    HardwareProfile,
    PipelineSpec,
    PipelineStep,
    StepName,
)
from iter8ml.constants import CVStrategy, EmbeddingMethod, TaskType, TrackerType
from iter8ml.data.loader import load_data
from iter8ml.engine.evaluator import Evaluator
from iter8ml.engine.models.factory import available_model_names, get_model_class
from iter8ml.engine.models.selector import ModelSelector
from iter8ml.engine.tracker import JSONLTracker, Tracker
from iter8ml.engine.trainer import Trainer
from iter8ml.exceptions import (
    ArtifactError,
    DataLoadError,
    HamiltonUnavailableError,
    Iter8MLError,
    ModelFitError,
    RegistryError,
    TabularBlueprintError,
)
from iter8ml.orchestration import MedallionExecutionService
from iter8ml.runtime import compile_run_plan
from iter8ml.services.export import ExportService
from iter8ml.services.registry import PromotionResult, RegistryService
from iter8ml.services.reporting import ReportService
from iter8ml.session import ExperimentSession
from iter8ml.workspace import Workspace

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
    "PipelineSpec",
    "PipelineStep",
    "PromotionResult",
    "RegistryError",
    "RegistryService",
    "ReportService",
    "StepName",
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
