"""MCP Server: exposes atomic tools for LLM agents."""

import json
from pathlib import Path
from typing import Any

from iter8ml.config import ExperimentConfig
from iter8ml.constants import from_task_type
from iter8ml.data.loader import load_data
from iter8ml.engine.models.factory import get_model_class
from iter8ml.engine.state_observer import StateObserver
from iter8ml.engine.trainer import Trainer
from iter8ml.services.registry import RegistryService
from iter8ml.utils.io import load_events

_TOOLS: list[Any] = []


def _tool(func: Any) -> Any:
    """Deferred tool registration — wired to FastMCP on first access."""
    _TOOLS.append(func)
    return func


def _init_mcp() -> Any:
    """Lazy initializer for the FastMCP server."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("iter8ml")
    for func in _TOOLS:
        mcp.tool()(func)
    return mcp


def __getattr__(name: str) -> Any:
    if name == "mcp":
        return _init_mcp()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@_tool
def get_experiment_state() -> str:
    """Returns current_state.md content with leaderboard and resource status."""
    observer = StateObserver()
    return observer.generate()


@_tool
def get_column_stats(data_path: str) -> str:
    """Returns Polars describe() output for a dataset."""
    df = load_data(data_path)
    desc = df.describe()
    headers = desc.columns
    separator = ["---"] * len(headers)
    rows = [headers, separator]

    for record in desc.iter_rows():
        rows.append([str(value) for value in record])

    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


@_tool
def run_baseline(data_path: str, target_col: str, task: str = "classification") -> str:
    """Triggers a TabPFN/CatBoost quick baseline run."""
    df = load_data(data_path)

    config = ExperimentConfig(
        name="baseline",
        task=from_task_type(task),
        target_col=target_col,
        data_path=data_path,
        models=["tabpfn", "catboost"],
    )

    trainer = Trainer(config)
    results = trainer.run(df)
    return json.dumps(results, indent=2)


@_tool
def run_hpo(
    data_path: str,
    target_col: str,
    model: str = "catboost",
    task: str = "classification",
    trials: int = 50,
) -> str:
    """Triggers Optuna study for a named model."""
    from iter8ml.engine.hpo import optimize_model, setup_hpo_components

    X, y, evaluator, search_space = setup_hpo_components(data_path, target_col, task, model)

    from iter8ml.engine.tracker import JSONLTracker

    tracker = JSONLTracker(log_path="workspace/experiments.jsonl")

    model_cls = get_model_class(model)
    result = optimize_model(
        model_cls,
        X,
        y,
        evaluator,
        model,
        n_trials=trials,
        search_space=search_space,
        task=task,
        log_path="workspace/experiments.jsonl",
        tracker=tracker,
    )

    return json.dumps(result, indent=2)


@_tool
def get_event_log(n: int = 10) -> str:
    """Returns last N JSONL events."""
    log_path = Path("workspace/experiments.jsonl")
    if not log_path.exists():
        return "No events found."

    events = load_events(log_path)
    return json.dumps(events[-n:], indent=2)


@_tool
def registry_show() -> str:
    """Returns current registry.json content."""
    registry = RegistryService("workspace/registry.json")
    data = registry.get_all()
    if not data:
        return "Registry is empty."
    return json.dumps(data, indent=2)


@_tool
def registry_promote(run_id: str, key: str) -> str:
    """Promotes a run_id to champion in the registry."""
    log_path = Path("workspace/experiments.jsonl")
    if not log_path.exists():
        return "No events found to locate run."

    registry = RegistryService("workspace/registry.json")
    result = registry.promote_run(run_id=run_id, key=key, log_path=log_path)
    return result.model_dump_json(indent=2)


@_tool
def detect_drift(reference_path: str, new_path: str) -> str:
    """Detects distribution drift between reference and new datasets."""
    from iter8ml.analysis.drift import DriftDetector

    ref_df = load_data(reference_path)
    new_df = load_data(new_path)

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    return json.dumps(report.model_dump(), indent=2)


@_tool
def export_champion(key: str, target_col: str = "") -> str:
    """Export the champion model for a registry key as a portable package."""
    from iter8ml.services.export import ExportService

    service = ExportService()
    try:
        export_path = service.export(key, target_col=target_col or None)
        return json.dumps(
            {
                "status": "ok",
                "export_path": str(export_path),
                "files": ["model.artifact", "predictor.py", "metadata.json", "pipelines/"],
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
