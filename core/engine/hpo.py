"""Optuna study factory for hyperparameter optimization."""

import optuna


def create_study(
    model_name: str,
    direction: str = "maximize",
    n_trials: int = 50,
    pruner: str = "median",
) -> optuna.Study:
    """Create an Optuna study for a given model."""
    if pruner == "median":
        pruner_obj = optuna.pruners.MedianPruner()
    elif pruner == "hyperband":
        pruner_obj = optuna.pruners.HyperbandPruner()
    else:
        pruner_obj = optuna.pruners.NopPruner()

    return optuna.create_study(direction=direction, pruner=pruner_obj)


def _validate_bounds(param_name: str, low: int | float, high: int | float) -> None:
    """Validate that search space bounds are numeric and ordered."""
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f"Invalid search space for '{param_name}': bounds must be numeric")
    if low >= high:
        raise ValueError(
            f"Invalid search space for '{param_name}': lower bound {low} >= upper bound {high}"
        )


def optimize_model(
    model_cls,
    X,
    y,
    evaluator,
    model_name: str,
    n_trials: int = 50,
    search_space: dict | None = None,
    task: str = "classification",
) -> dict:
    """Run HPO for a model and return best params + scores."""
    study = create_study(model_name, n_trials=n_trials)

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
            primary = list(scores.values())[0]
            return primary
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Add context about which parameters failed
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            raise optuna.TrialPruned(
                f"Evaluation failed for trial with params: {params_str}. Error: {e}"
            ) from e

    study.optimize(objective, n_trials=n_trials)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
