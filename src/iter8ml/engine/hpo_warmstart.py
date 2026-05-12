"""Pre-warmed HPO: inject historical trials into new Optuna studies."""

from collections.abc import Generator
from typing import Any

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from pydantic import BaseModel

from iter8ml.utils.io import load_events


class WarmstartInjection(BaseModel):
    """Metadata about historical HPO trials injected into a new study."""

    n_trials_injected: int
    n_runs_scanned: int
    n_skipped_missing_scores: int = 0
    n_skipped_missing_params: int = 0
    n_skipped_invalid_trials: int = 0
    model_name: str
    source_log: str


def _parse_model_completed_events(
    events: list[dict[str, Any]],
    model_name: str,
) -> Generator[dict[str, Any], None, None]:
    """Yield training/HPO completion events for the given model name."""
    for event in events:
        if event.get("event") not in {"model_completed", "hpo_trial_completed"}:
            continue
        if event.get("model") == model_name:
            yield event


def _infer_distribution(name: str, value: Any) -> Any:
    """Infer an Optuna distribution from a parameter value and name.

    Uses naming conventions to choose the appropriate distribution type.
    """
    if isinstance(value, bool):
        return CategoricalDistribution([True, False])

    if isinstance(value, int):
        if any(kw in name for kw in ("n_estimators", "iterations", "num_boost_round")):
            return IntDistribution(50, 5000)
        if any(kw in name for kw in ("depth", "max_depth")):
            return IntDistribution(2, 15)
        if "num_leaves" in name:
            return IntDistribution(8, 256)
        if "batch_size" in name:
            return IntDistribution(32, 1024)
        if any(kw in name for kw in ("n_epochs", "epochs")):
            return IntDistribution(10, 200)
        return IntDistribution(max(1, int(value * 0.5)), max(2, int(value * 2)))

    if isinstance(value, float):
        if any(kw in name for kw in ("lr", "learning_rate")):
            return FloatDistribution(1e-5, 0.5, log=True)
        if "dropout" in name:
            return FloatDistribution(0.0, 0.5)
        if any(kw in name for kw in ("subsample", "colsample", "frac")):
            return FloatDistribution(0.4, 1.0)
        return FloatDistribution(max(0.0, value * 0.5), value * 2.0 + 1e-6)

    return FloatDistribution(-10.0, 10.0)


def _build_trial_data(
    params: dict[str, Any],
    value: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build params/distributions dicts for create_trial.

    Returns (params, distributions) suitable for optuna.trial.create_trial().
    """
    distributions = {name: _infer_distribution(name, val) for name, val in params.items()}
    return params, distributions


def _valid_params(params: Any) -> bool:
    return isinstance(params, dict) and len(params) > 0


def create_warmstarted_study(
    model_name: str,
    direction: str = "maximize",
    log_path: str = "workspace/experiments.jsonl",
    n_trials: int = 50,
    pruner: str = "median",
) -> tuple[optuna.Study, WarmstartInjection]:
    """Create an Optuna study pre-warmed with trials from historical experiment logs.

    Args:
        model_name: Model name to inject trials from (e.g. "catboost", "lightgbm")
        direction: Optimization direction ("maximize" or "minimize")
        log_path: Path to experiments JSONL log
        n_trials: Total trials for the study (used for pruner sizing)
        pruner: Pruner type ("median", "hyperband", "nop")

    Returns:
        Tuple of (Optuna study, WarmstartInjection metadata)
    """
    from iter8ml.engine.hpo import _build_pruner

    pruner_obj = _build_pruner(pruner)

    study = optuna.create_study(
        direction=direction,
        pruner=pruner_obj,
        study_name=f"warmstart_{model_name}",
    )

    events = load_events(log_path)
    matching_events = list(_parse_model_completed_events(events, model_name))

    injected = 0
    skipped_missing_scores = 0
    skipped_missing_params = 0
    skipped_invalid_trials = 0
    for event in matching_events:
        cv_scores = event.get("cv_scores", {})
        if not cv_scores:
            skipped_missing_scores += 1
            continue

        primary_score = next(iter(cv_scores.values()), None)
        if primary_score is None:
            skipped_missing_scores += 1
            continue

        params = event.get("params", {})
        if not _valid_params(params):
            skipped_missing_params += 1
            continue

        try:
            _, distributions = _build_trial_data(params, primary_score)
            trial = optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=primary_score,
                params=params,
                distributions=distributions,
            )
            study.add_trial(trial)
            injected += 1
        except (ValueError, TypeError, RuntimeError):
            skipped_invalid_trials += 1
            continue

    injection = WarmstartInjection(
        n_trials_injected=injected,
        n_runs_scanned=len(matching_events),
        n_skipped_missing_scores=skipped_missing_scores,
        n_skipped_missing_params=skipped_missing_params,
        n_skipped_invalid_trials=skipped_invalid_trials,
        model_name=model_name,
        source_log=str(log_path),
    )

    return study, injection


def inject_trials_from_previous_runs(
    study: optuna.Study,
    model_name: str,
    log_path: str,
) -> int:
    """Inject historical trials into an existing study in-place.

    Returns the number of trials successfully injected.
    """
    events = load_events(log_path)
    matching_events = list(_parse_model_completed_events(events, model_name))

    injected = 0
    for event in matching_events:
        cv_scores = event.get("cv_scores", {})
        if not cv_scores:
            continue
        primary_score = next(iter(cv_scores.values()), None)
        if primary_score is None:
            continue
        params = event.get("params", {})
        if not _valid_params(params):
            continue

        try:
            _, distributions = _build_trial_data(params, primary_score)
            trial = optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=primary_score,
                params=params,
                distributions=distributions,
            )
            study.add_trial(trial)
            injected += 1
        except (ValueError, TypeError, RuntimeError):
            continue

    return injected
