"""MCP Server: exposes atomic tools for LLM agents."""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from core.data.loaders import load_data
from core.engine.state_observer import StateObserver
from core.models.factory import get_model_class
from core.services.registry_service import RegistryService
from core.utils.jsonl import load_events

mcp = FastMCP("tabular-blueprint")


@mcp.tool()
def get_experiment_state() -> str:
    """Returns current_state.md content with leaderboard and resource status."""
    observer = StateObserver()
    return observer.generate()


@mcp.tool()
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


@mcp.tool()
def run_baseline(data_path: str, target_col: str, task: str = "classification") -> str:
    """Triggers a TabPFN/CatBoost quick baseline run."""
    from configs.experiment import ExperimentConfig
    from core.engine.trainer import Trainer

    df = load_data(data_path)

    config = ExperimentConfig(
        name="baseline",
        task=task,
        target_col=target_col,
        data_path=data_path,
        models=["tabpfn", "catboost"],
    )

    trainer = Trainer(config)
    results = trainer.run(df)
    return json.dumps(results, indent=2)


@mcp.tool()
def run_hpo(
    data_path: str,
    target_col: str,
    model: str = "catboost",
    task: str = "classification",
    trials: int = 50,
) -> str:
    """Triggers Optuna study for a named model."""
    from core.engine.hpo import optimize_model, setup_hpo_components

    X, y, evaluator, search_space = setup_hpo_components(data_path, target_col, task, model)

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
    )

    return json.dumps(result, indent=2)


@mcp.tool()
def get_event_log(n: int = 10) -> str:
    """Returns last N JSONL events."""
    log_path = Path("workspace/experiments.jsonl")
    if not log_path.exists():
        return "No events found."

    events = load_events(log_path)
    return json.dumps(events[-n:], indent=2)


@mcp.tool()
def registry_show() -> str:
    """Returns current registry.json content."""
    registry = RegistryService("workspace/registry.json")
    data = registry.get_all()
    if not data:
        return "Registry is empty."
    return json.dumps(data, indent=2)


@mcp.tool()
def registry_promote(run_id: str, key: str) -> str:
    """Promotes a run_id to champion in the registry."""
    log_path = Path("workspace/experiments.jsonl")
    if not log_path.exists():
        return "No events found to locate run."

    registry = RegistryService("workspace/registry.json")
    result = registry.promote_run(run_id=run_id, key=key, log_path=log_path)
    return result.message


@mcp.tool()
def detect_drift(reference_path: str, new_path: str) -> str:
    """Detects distribution drift between reference and new datasets."""
    from core.monitoring.drift import DriftDetector

    ref_df = load_data(reference_path)
    new_df = load_data(new_path)

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    return json.dumps(report.model_dump(), indent=2)
