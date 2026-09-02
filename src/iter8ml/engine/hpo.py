"""Optuna study factory for hyperparameter optimization."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from iter8ml.constants import TaskType
from iter8ml.engine.evaluator import Evaluator
from iter8ml.services.reporting import metric_higher_is_better

if TYPE_CHECKING:
    from iter8ml.engine.tracker import Tracker

_hpo_file_lock = threading.Lock()


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


def _write_hpo_event(
    event: dict[str, Any],
    log_path: str | None,
    tracker: "Tracker | None",
) -> None:
    """Write an HPO event to the tracker or log file."""
    if tracker is not None:
        tracker.log_event(event)
        return
    if log_path is None:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _hpo_file_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _log_hpo_trial(
    *,
    params: dict[str, Any],
    cv_scores: dict[str, float],
    model_name: str,
    task: str,
    log_path: str | None,
    tracker: "Tracker | None",
) -> None:
    """Log a completed HPO trial as an event."""
    event = {
        "event": "hpo_trial_completed",
        "run_id": f"hpo_{model_name}",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "task": task,
        "params": params,
        "cv_scores": cv_scores,
    }
    _write_hpo_event(event, log_path, tracker)


def _log_warning_event(
    *,
    source: str,
    warning_type: str,
    message: str,
    model_name: str,
    warnings: list[dict[str, str]],
    log_path: str | None,
    tracker: "Tracker | None",
) -> None:
    """Append a warning and write it as an HPO event."""
    warnings.append({"source": source, "warning_type": warning_type, "message": message})
    event = {
        "event": "hpo_warning",
        "run_id": f"hpo_{model_name}",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "warning_source": source,
        "warning_type": warning_type,
        "message": message,
    }
    _write_hpo_event(event, log_path, tracker)


def _parse_trial_params(trial: optuna.Trial, search_space: dict | None) -> dict[str, Any]:
    """Sample parameters from the search space using an Optuna trial."""
    params: dict[str, Any] = {}
    if not search_space:
        return params
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
    return params


def setup_hpo_components(
    data_path: str,
    target_col: str,
    task: str,
    model: str,
    cv_folds: int | None = None,
    metrics: list[str] | None = None,
    random_seed: int | None = None,
    ignore_cols: list[str] | None = None,
    positive_class: str | float | bool | None = None,
) -> tuple[np.ndarray, np.ndarray, Evaluator, dict]:
    """
    Shared setup for HPO across CLI and MCP.

    Args:
        data_path: Path to data file
        target_col: Target column name
        task: "classification" or "regression"
        model: Model name for HPO
        cv_folds: Optional fold count override (from an ExperimentConfig).
        metrics: Optional metric list override; primary metric first.
        random_seed: Optional seed override, wired to CV splitters/model ctors
            so HPO folds are reproducible and comparable with ``iter8 run``.
        ignore_cols: Optional column drops, matching the training config so
            HPO sees the same feature set as ``iter8 run``.
        positive_class: Optional binary-target orientation (see prep's
            ``target_oriented_df``) so HPO scores the same orientation as
            training instead of appearance-order codes.

    Returns:
        (X, y, evaluator, search_space)
    """
    from iter8ml.config import ExperimentConfig
    from iter8ml.data.adapter import DataAdapter
    from iter8ml.data.loader import load_data
    from iter8ml.engine.evaluator import Evaluator
    from iter8ml.engine.models.factory import validate_model_name
    from iter8ml.engine.models.model_configs import ModelConfigs

    overrides: dict[str, Any] = {}
    if cv_folds is not None:
        overrides["cv_folds"] = cv_folds
    if metrics is not None:
        overrides["metrics"] = metrics
    if random_seed is not None:
        overrides["random_seed"] = random_seed
    if ignore_cols is not None:
        overrides["ignore_cols"] = ignore_cols
    if positive_class is not None:
        overrides["positive_class"] = positive_class
    hpo_config = ExperimentConfig(
        name="hpo",
        task=TaskType(task),
        target_col=target_col,
        data_path=data_path,
        **overrides,
    )

    # Validate the model before spending prep compute on a doomed run.
    validate_model_name(model)
    model_configs = ModelConfigs()
    model_config = getattr(model_configs, model, None)
    if model_config is None:
        raise ValueError(
            f"Model '{model}' is not HPO-able: it has no configurable search space. "
            "HPO requires a configurable model; baseline models are not tunable."
        )

    # Route the raw frame through the same prep chain as training
    # (ignore_cols → null fill → dates → categorical encoding → target
    # validation) so DataAdapter receives numeric codes, not raw strings —
    # string categoricals crash LightGBM/XGBoost constructors.
    from iter8ml.engine.pipelines.executor import (
        PipelineExecutor,
        PipelineMode,
        _resolve_hamilton_config,
    )

    executor = PipelineExecutor(mode=PipelineMode.HPO, config=_resolve_hamilton_config(hpo_config))
    df = load_data(data_path)
    df = executor.run_prep(hpo_config, df)

    adapter = DataAdapter()
    X, y = adapter.transform(df, target_col)

    evaluator = Evaluator(hpo_config)

    search_space = model_config.hpo_search_space()

    return X, y, evaluator, search_space


def _compute_hpo_result(
    study: optuna.Study,
    injection: Any | None,
    warnings: list[dict[str, str]],
    model_name: str,
    log_path: str | None,
    tracker: "Tracker | None",
    direction: str,
    primary_metric: str | None,
) -> dict[str, Any]:
    """Assemble the result dict from a completed HPO study."""
    result: dict[str, Any] = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "direction": direction,
        "primary_metric": primary_metric,
    }

    if injection is not None:
        result["warmstart_trials"] = injection.n_trials_injected
        result["warmstart_summary"] = {
            "n_runs_scanned": injection.n_runs_scanned,
            "n_trials_injected": injection.n_trials_injected,
            "n_skipped_missing_scores": injection.n_skipped_missing_scores,
            "n_skipped_missing_params": injection.n_skipped_missing_params,
            "n_skipped_invalid_trials": injection.n_skipped_invalid_trials,
            "n_skipped_metric_mismatch": injection.n_skipped_metric_mismatch,
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
            model_name=model_name,
            warnings=warnings,
            log_path=log_path,
            tracker=tracker,
        )

    if warnings:
        result["warnings"] = warnings

    return result


def _resolve_primary_metric(metrics: list[str] | None, evaluator: Any) -> str | None:
    """Resolve the HPO primary metric from the explicit list or the evaluator.

    An explicit ``metrics`` list wins (first entry). Otherwise fall back to the
    evaluator's configured metric list — the same keys its scores dict returns.
    Returns None when neither is available; the objective then resolves the
    primary from the first returned score at trial time.
    """
    if metrics:
        return metrics[0]
    evaluator_metrics = getattr(evaluator, "metrics", None)
    if isinstance(evaluator_metrics, list | tuple) and evaluator_metrics:
        return str(evaluator_metrics[0])
    return None


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
    metrics: list[str] | None = None,
    fixed_params: dict[str, Any] | None = None,
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
        metrics: Metrics the evaluation computes, primary first. The study
            direction is derived from the primary metric via the central
            direction registry (``services.reporting``): lower-is-better
            metrics (e.g. rmse) minimize, others maximize. When omitted, the
            evaluator's configured metric list is used, falling back to the
            first returned score at trial time.
        fixed_params: Constants merged into every trial's params (trial
            suggestions win on collision) — e.g. ``model_overrides`` from an
            ExperimentConfig that should hold fixed while HPO tunes the rest.

    Returns:
        Dict with best_params, best_value, n_trials, direction,
        primary_metric, and optionally warmstart_trials (number injected)
        and param_importances.
    """
    primary_metric = _resolve_primary_metric(metrics, evaluator)
    direction = "maximize" if metric_higher_is_better(primary_metric) else "minimize"

    warnings: list[dict[str, str]] = []

    if log_path is not None:
        from iter8ml.engine.hpo_warmstart import create_warmstarted_study

        study, injection = create_warmstarted_study(
            model_name=model_name,
            direction=direction,
            log_path=log_path,
            n_trials=n_trials,
            primary_metric=primary_metric,
        )
        if injection.n_trials_injected == 0 and injection.n_skipped_metric_mismatch > 0:
            _log_warning_event(
                source="hpo_warmstart",
                warning_type="MetricMismatch",
                message=(
                    "Warmstart skipped: no historical events for "
                    f"'{model_name}' were scored on the current primary metric "
                    f"({primary_metric!r}); started a fresh study instead."
                ),
                model_name=model_name,
                warnings=warnings,
                log_path=log_path,
                tracker=tracker,
            )
    else:
        study = create_study(model_name, direction=direction, n_trials=n_trials)
        injection = None

    pruned_errors: list[str] = []

    def objective(trial: optuna.Trial) -> float:
        params = {**(fixed_params or {}), **_parse_trial_params(trial, search_space)}
        try:
            scores = evaluator.evaluate(model_cls, X, y, task=task, **params)
            if not scores:
                pruned_errors.append("evaluator returned no scores")
                raise optuna.TrialPruned("No scores returned from evaluator")
            _log_hpo_trial(
                params=params,
                cv_scores=scores,
                model_name=model_name,
                task=task,
                log_path=log_path,
                tracker=tracker,
            )
            trial_primary = primary_metric if primary_metric in scores else next(iter(scores))
            return scores[trial_primary]
        except optuna.TrialPruned:
            raise
        except Exception as e:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            pruned_errors.append(f"params ({params_str}): {e}")
            raise optuna.TrialPruned(
                f"Evaluation failed for trial with params: {params_str}. Error: {e}"
            ) from e

    study.optimize(objective, n_trials=n_trials)

    # A systematically broken model/config prunes every trial; without this
    # guard the study "completes" and _compute_hpo_result crashes on an empty
    # best_value — or worse, crowns a winner over a fluke survivor set.
    completed = [
        t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE
    ]
    # Minimum survivor set for a trustworthy "best": 10% of the study, at
    # least 3 — but never more than the study itself asked for.
    min_completed = min(n_trials, max(3, n_trials // 10))
    if len(completed) < min_completed:
        detail = pruned_errors[0] if pruned_errors else "no trial reported an exception"
        raise ValueError(
            f"HPO for '{model_name}': only {len(completed)} of {n_trials} trials completed "
            f"(minimum {min_completed}). First failure — {detail}"
        )

    return _compute_hpo_result(
        study,
        injection,
        warnings,
        model_name,
        log_path,
        tracker,
        direction,
        primary_metric,
    )
