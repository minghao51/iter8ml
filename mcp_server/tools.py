"""MCP Server: exposes atomic tools for LLM agents."""

import json
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from core.data.loaders import load_data
from core.engine.state_observer import StateObserver
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
    import polars as pl

    df = load_data(data_path)
    desc = df.describe()
    return desc.to_pandas().to_markdown()


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
    from configs.model_configs import ModelConfigs
    from core.data.adapter import DataAdapter
    from core.engine.evaluator import Evaluator
    from core.engine.hpo import optimize_model
    from core.engine.trainer import _get_model_class

    df = load_data(data_path)

    adapter = DataAdapter(target_format="numpy")
    X, y = adapter.transform(df, target_col)

    evaluator = Evaluator(task=task)

    model_configs = ModelConfigs()
    search_space = getattr(model_configs, model).hpo_search_space()

    model_cls = _get_model_class(model)
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

    events = load_events(log_path)

    run_event = next(
        (
            e
            for e in events
            if e.get("run_id") == run_id and e.get("event") == "model_completed"
        ),
        None,
    )

    if not run_event:
        return f"Run {run_id} not found."

    cv_scores = run_event.get("cv_scores", {})
    score = cv_scores.get("roc_auc", cv_scores.get("r2", 0))

    registry = RegistryService("workspace/registry.json")
    updated = registry.update_if_better(
        key=key,
        model_name=run_event.get("model"),
        run_id=run_id,
        score=score,
        artifact_path=run_event.get("artifact_path"),
    )

    if updated:
        return f"Promoted {run_id} to champion for {key}."
    return f"Existing champion for {key} has better score."


@mcp.tool()
def detect_drift(reference_path: str, new_path: str) -> str:
    """Detects distribution drift between reference and new datasets."""
    from core.monitoring.drift import DriftDetector

    ref_df = load_data(reference_path)
    new_df = load_data(new_path)

    detector = DriftDetector(ref_df)
    report = detector.detect(new_df)

    return json.dumps(report.model_dump(), indent=2)
