"""Optimization command — hyperparameter optimization."""

import typer

from iter8ml.engine.models.factory import get_model_class

from .main import app


@app.command()
def hpo(
    data_path: str = typer.Option(..., "--data", "-d"),
    target_col: str = typer.Option(..., "--target", "-t"),
    model: str = typer.Option("catboost", "--model", "-m"),
    task: str = typer.Option("classification", "--task"),
    trials: int = typer.Option(50, "--trials", "-n"),
    log_path: str = typer.Option(
        "workspace/experiments.jsonl",
        "--log",
        help="Path to experiments JSONL for warmstart",
    ),
) -> None:
    """Run hyperparameter optimization for a model."""
    from iter8ml.engine.hpo import optimize_model, setup_hpo_components
    from iter8ml.engine.tracker import JSONLTracker

    try:
        X, y, evaluator, search_space = setup_hpo_components(data_path, target_col, task, model)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

    model_cls = get_model_class(model)
    tracker = JSONLTracker(log_path=log_path)

    typer.echo(f"Running HPO for {model} ({trials} trials)...")
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
    )

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
            f"skipped_invalid_trials={warmstart_summary.get('n_skipped_invalid_trials', 0)}"
        )

    if result.get("param_importances"):
        typer.echo("\nHyperparameter Importance (PedAnova):")
        for item in result["param_importances"]:
            typer.echo(f"  {item['param']}: {item['importance']:.4f}")
