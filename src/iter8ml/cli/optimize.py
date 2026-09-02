"""Optimization command — hyperparameter optimization."""

import typer

from iter8ml.engine.models.factory import get_model_class

from .main import app


@app.command()
def hpo(
    data_path: str | None = typer.Option(None, "--data", "-d"),
    target_col: str | None = typer.Option(None, "--target", "-t"),
    model: str | None = typer.Option(None, "--model", "-m"),
    task: str | None = typer.Option(None, "--task"),
    trials: int = typer.Option(50, "--trials", "-n"),
    config_path: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="ExperimentConfig file (YAML/TOML/JSON). Reuses task, target, data, "
        "cv_folds, metrics, random_seed and model_overrides; explicit CLI flags override.",
    ),
    log_path: str = typer.Option(
        "workspace/experiments.jsonl",
        "--log",
        help="Path to experiments JSONL for warmstart",
    ),
) -> None:
    """Run hyperparameter optimization for a model."""
    from iter8ml.engine.hpo import optimize_model, setup_hpo_components
    from iter8ml.engine.tracker import JSONLTracker

    cv_folds: int | None = None
    metrics: list[str] | None = None
    random_seed: int | None = None
    fixed_params: dict | None = None
    cfg_ignore_cols: list[str] | None = None
    cfg_positive_class: str | float | bool | None = None
    if config_path is not None:
        from iter8ml.config import ExperimentConfig

        try:
            cfg = ExperimentConfig.from_file(config_path)
        except Exception as e:
            typer.echo(f"Error: invalid config '{config_path}': {e}")
            raise typer.Exit(1) from e
        if data_path is None:
            data_path = cfg.data_path
        if target_col is None:
            target_col = cfg.target_col
        if task is None:
            task = cfg.task.value
        if model is None:
            # "auto" or empty: no concrete model in config — flag/default decides.
            model = None if isinstance(cfg.models, str) or not cfg.models else cfg.models[0]
        cv_folds = cfg.cv_folds
        metrics = list(cfg.metrics)
        # HPO must optimize what the leaderboard ranks by.
        if cfg.primary_metric and cfg.primary_metric in metrics:
            metrics = [cfg.primary_metric] + [m for m in metrics if m != cfg.primary_metric]
        random_seed = cfg.random_seed
        cfg_ignore_cols = list(cfg.ignore_cols) if cfg.ignore_cols else None
        cfg_positive_class = cfg.positive_class
        # model_overrides are nested per-model; flatten for the resolved model.
        if fixed_params is None and cfg.model_overrides and model is not None:
            fixed_params = dict(cfg.model_overrides.get(model, {})) or None

    if data_path is None or target_col is None:
        typer.echo("Error: --data and --target are required (or pass --config).")
        raise typer.Exit(1)
    task = task or "classification"
    model = model or "catboost"

    typer.echo(
        f"HPO config: data={data_path} target={target_col} task={task} model={model} "
        f"folds={cv_folds if cv_folds is not None else 'default'} "
        f"metrics={metrics if metrics is not None else 'default'} "
        f"seed={random_seed if random_seed is not None else 'default'}"
    )

    try:
        X, y, evaluator, search_space = setup_hpo_components(
            data_path,
            target_col,
            task,
            model,
            cv_folds=cv_folds,
            metrics=metrics,
            random_seed=random_seed,
            ignore_cols=cfg_ignore_cols,
            positive_class=cfg_positive_class,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

    model_cls = get_model_class(model)
    tracker = JSONLTracker(log_path=log_path)

    typer.echo(f"Running HPO for {model} ({trials} trials)...")
    try:
        result = optimize_model(
            model_cls,
            X,
            y,
            evaluator,
            model,
            n_trials=trials,
            search_space=search_space,
            task=task,
            log_path=log_path,
            tracker=tracker,
            metrics=evaluator.metrics,
            fixed_params=fixed_params,
        )
    except ValueError as e:
        # e.g. systematically failing model: too few completed trials.
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

    typer.echo(f"\nBest params: {result['best_params']}")
    typer.echo(f"Best value: {result['best_value']:.4f}")
    typer.echo(f"Trials completed: {result['n_trials']}")

    if result.get("warmstart_trials", 0) > 0:
        typer.echo(f"Warmstart: injected {result['warmstart_trials']} historical trials")

    warmstart_summary = result.get("warmstart_summary")
    if isinstance(warmstart_summary, dict):
        typer.echo(
            "Warmstart summary: "
            f"scanned={warmstart_summary.get('n_runs_scanned', 0)}, "
            f"injected={warmstart_summary.get('n_trials_injected', 0)}, "
            f"skipped_missing_scores={warmstart_summary.get('n_skipped_missing_scores', 0)}, "
            f"skipped_missing_params={warmstart_summary.get('n_skipped_missing_params', 0)}, "
            f"skipped_invalid_trials={warmstart_summary.get('n_skipped_invalid_trials', 0)}, "
            f"skipped_metric_mismatch={warmstart_summary.get('n_skipped_metric_mismatch', 0)}"
        )

    if result.get("param_importances"):
        typer.echo("\nHyperparameter Importance (PedAnova):")
        for item in result["param_importances"]:
            typer.echo(f"  {item['param']}: {item['importance']:.4f}")
