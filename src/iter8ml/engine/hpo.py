"""Optuna study factory for hyperparameter optimization."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from iter8ml.constants import from_task_type
from iter8ml.engine.evaluator import Evaluator

if TYPE_CHECKING:
    from iter8ml.engine.tracker import Tracker


def _build_pruner(pruner: str) -> optuna.pruners.BasePruner:
    if pruner == "median":
        return optuna.pruners.MedianPruner()
    if pruner == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.NopPruner()


def create_study(
    model_name: str,
    direction: str = "maximize",
    n_trials: int = 50,
    pruner: str = "median",
) -> optuna.Study:
    """Create an Optuna study for a given model."""
    return optuna.create_study(direction=direction, pruner=_build_pruner(pruner))


def _validate_bounds(param_name: str, low: int | float, high: int | float) -> None:
    """Validate that search space bounds are numeric and ordered."""
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f"Invalid search space for '{param_name}': bounds must be numeric")
    if low >= high:
        raise ValueError(
            f"Invalid search space for '{param_name}': lower bound {low} >= upper bound {high}"
        )


def setup_hpo_components(
    data_path: str,
    target_col: str,
    task: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray, Evaluator, dict]:
    """
    Shared setup for HPO across CLI and MCP.

    Args:
        data_path: Path to data file
        target_col: Target column name
        task: "classification" or "regression"
        model: Model name for HPO

    Returns:
        (X, y, evaluator, search_space)
    """
    from iter8ml.config import ExperimentConfig
    from iter8ml.data.adapter import DataAdapter
    from iter8ml.data.loader import load_data
    from iter8ml.engine.evaluator import Evaluator
    from iter8ml.engine.models.factory import validate_model_name
    from iter8ml.engine.models.model_configs import ModelConfigs

    df = load_data(data_path)
    adapter = DataAdapter()
    X, y = adapter.transform(df, target_col)

    hpo_config = ExperimentConfig(
        name="hpo",
        task=from_task_type(task),
        target_col=target_col,
        data_path=data_path,
    )
    evaluator = Evaluator(hpo_config)

    validate_model_name(model)
    model_configs = ModelConfigs()
    search_space = getattr(model_configs, model).hpo_search_space()

    return X, y, evaluator, search_space


def optimize_model(
    model_cls: Any,
    X: np.ndarray,
    y: np.ndarray,
    evaluator: Any,
    model_name: str,
    n_trials: int = 50,
    search_space: dict | None = None,
    task: str = "classification",
    log_path: str | None = None,
    tracker: "Tracker | None" = None,
) -> dict[str, Any]:
    """Run HPO for a model and return best params + scores.

    Args:
        model_cls: Model class to optimize.
        X: Feature matrix.
        y: Target vector.
        evaluator: Evaluator instance for cross-validation.
        model_name: Name of the model (used for warmstart and logging).
        n_trials: Number of HPO trials.
        search_space: Parameter search space dict.
        task: Task type.
        log_path: Optional path to experiments JSONL. When provided, historical
            trials from matching model_completed events are injected into the
            study before optimization (pre-warmed HPO).
        tracker: Optional Tracker instance. When provided, HPO events are
            logged through the tracker instead of direct file writes.

    Returns:
        Dict with best_params, best_value, n_trials, and optionally
        warmstart_trials (number injected) and param_importances.
    """
    warnings: list[dict[str, str]] = []

    if log_path is not None:
        from iter8ml.engine.hpo_warmstart import create_warmstarted_study

        study, injection = create_warmstarted_study(
            model_name=model_name,
            direction="maximize",
            log_path=log_path,
            n_trials=n_trials,
        )
    else:
        study = create_study(model_name, n_trials=n_trials)
        injection = None

    def _log_hpo_trial(
        *,
        params: dict[str, Any],
        cv_scores: dict[str, float],
    ) -> None:
        event = {
            "event": "hpo_trial_completed",
            "run_id": f"hpo_{model_name}",
            "timestamp": datetime.now(UTC).isoformat(),
            "model": model_name,
            "task": task,
            "params": params,
            "cv_scores": cv_scores,
        }
        if tracker is not None:
            tracker.log_event(event)
            return
        if log_path is None:
            return
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _log_warning_event(
        *,
        source: str,
        warning_type: str,
        message: str,
    ) -> None:
        warning = {"source": source, "warning_type": warning_type, "message": message}
        warnings.append(warning)
        event = {
            "event": "hpo_warning",
            "run_id": f"hpo_{model_name}",
            "timestamp": datetime.now(UTC).isoformat(),
            "model": model_name,
            "warning_source": source,
            "warning_type": warning_type,
            "message": message,
        }
        if tracker is not None:
            tracker.log_event(event)
            return
        if log_path is None:
            return
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def objective(trial: optuna.Trial) -> float:
        params = {}
        if search_space:
            for param_name, param_range in search_space.items():
                if not isinstance(param_range, (tuple, list)):
                    raise ValueError(
                        f"Invalid search space for '{param_name}': "
                        f"expected tuple/list, got {type(param_range).__name__}"
                    )
                if len(param_range) not in (2, 3):
                    raise ValueError(
                        f"Invalid search space for '{param_name}': "
                        f"expected 2 or 3 elements, got {len(param_range)}"
                    )
                if len(param_range) == 2:
                    low, high = param_range
                    _validate_bounds(param_name, low, high)
                    if isinstance(low, float) or isinstance(high, float):
                        params[param_name] = trial.suggest_float(param_name, low, high)
                    else:
                        params[param_name] = trial.suggest_int(param_name, low, high)
                elif len(param_range) == 3:
                    low, high, kind = param_range
                    _validate_bounds(param_name, low, high)
                    if kind not in ("linear", "log"):
                        raise ValueError(
                            f"Invalid search space for '{param_name}': "
                            f"kind must be 'linear' or 'log', got '{kind}'"
                        )
                    if kind == "log":
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)

        try:
            scores = evaluator.evaluate(model_cls, X, y, task=task, **params)
            if not scores:
                raise optuna.TrialPruned("No scores returned from evaluator")
            _log_hpo_trial(params=params, cv_scores=scores)
            primary = next(iter(scores.values()))
            return primary
        except optuna.TrialPruned:
            raise
        except Exception as e:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            raise optuna.TrialPruned(
                f"Evaluation failed for trial with params: {params_str}. Error: {e}"
            ) from e

    study.optimize(objective, n_trials=n_trials)

    result: dict[str, Any] = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }

    if injection is not None:
        result["warmstart_trials"] = injection.n_trials_injected
        result["warmstart_summary"] = {
            "n_runs_scanned": injection.n_runs_scanned,
            "n_trials_injected": injection.n_trials_injected,
            "n_skipped_missing_scores": injection.n_skipped_missing_scores,
            "n_skipped_missing_params": injection.n_skipped_missing_params,
            "n_skipped_invalid_trials": injection.n_skipped_invalid_trials,
        }

    try:
        from iter8ml.engine.hpo_importance import compute_param_importance

        importance_report = compute_param_importance(study)
        result["param_importances"] = [
            {"param": p.param_name, "importance": p.importance}
            for p in importance_report.importances
        ]
    except Exception as e:
        _log_warning_event(
            source="hpo_importance",
            warning_type=type(e).__name__,
            message=f"Failed to compute parameter importances: {e}",
        )

    if warnings:
        result["warnings"] = warnings

    return result
