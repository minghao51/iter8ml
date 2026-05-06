"""CLI entry point for tabular-blueprint."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tabular_blueprint.config import ExperimentConfig, HardwareProfile
from tabular_blueprint.constants import from_task_type
from tabular_blueprint.data.loaders import load_data
from tabular_blueprint.engine.trainer import Trainer
from tabular_blueprint.models.factory import get_model_class
from tabular_blueprint.services.report_service import ReportService
from tabular_blueprint.utils.jsonl import load_events

app = typer.Typer(name="tabblueprint", help="A high-velocity iteration framework for tabular ML")


@app.command()
def init(
    data: str | None = None,
    force_reset_registry: bool = typer.Option(
        False,
        "--force-reset-registry",
        help="Reset registry.json even if it already exists (destructive).",
    ),
) -> None:
    """Initialize workspace and optionally load data."""
    workspace = Path("workspace")
    workspace.mkdir(exist_ok=True)
    (workspace / "artifacts").mkdir(exist_ok=True)

    (workspace / "experiments.jsonl").touch(exist_ok=True)
    registry_path = workspace / "registry.json"
    if force_reset_registry or not registry_path.exists():
        registry_path.write_text("{}")

    typer.echo("Workspace initialized.")
    if data:
        typer.echo(f"Data path set to: {data}")


@app.command()
def run(
    config: str = typer.Option(
        None, "--config", "-c", help="Config file (.yaml/.yml/.toml/.json/.py)"
    ),
    data_path: str = typer.Option(None, "--data", "-d", help="Path to data file"),
    target_col: str = typer.Option(None, "--target", "-t", help="Target column name"),
    task: str = typer.Option("classification", "--task", help="classification or regression"),
    models: list[str] | None = typer.Option(None, "--models", "-m", help="Model names to run"),
    quick: bool = typer.Option(
        False, "--quick", help="Fast mode: 2 folds, 20% data, skip SHAP/AFE"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume previous run, skip completed models"
    ),
    allow_unsafe_config: bool = typer.Option(
        False,
        "--allow-unsafe-config",
        help="Allow loading .py config files (unsafe, executes code).",
    ),
) -> None:
    """Run an experiment."""
    experiment_config = None
    if config:
        config_path = Path(config)
        if not config_path.exists() or not config_path.is_file():
            typer.echo(f"Error: config file not found: {config}")
            raise typer.Exit(1)

        try:
            experiment_config = ExperimentConfig.from_file(
                config_path, allow_unsafe_python=allow_unsafe_config
            )
        except (ValueError, FileNotFoundError) as e:
            typer.echo(f"Error loading config: {e}")
            raise typer.Exit(1) from e

    if experiment_config is None:
        experiment_config = ExperimentConfig(
            name="experiment",
            task=from_task_type(task),
            target_col=target_col or "target",
            data_path=data_path or "",
        )

    if models is not None and models:
        experiment_config.models = models

    if not data_path:
        data_path = experiment_config.data_path

    if not data_path:
        typer.echo("Error: --data or config.data_path required")
        raise typer.Exit(1)

    try:
        df = load_data(data_path)
    except ValueError as e:
        typer.echo(f"Error loading data: {e}")
        raise typer.Exit(code=1) from e

    if quick:
        experiment_config.cv_folds = 2
        experiment_config.afe_enabled = False
        experiment_config.shap_enabled = False
        experiment_config.calibration = "none"
        experiment_config.data_sample = 0.2
        typer.echo("[quick mode] 2 folds, 20% data, SHAP/AFE/calibration disabled")

    typer.echo(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    typer.echo(f"Task: {experiment_config.task}, Target: {experiment_config.target_col}")

    trainer = Trainer(
        experiment_config,
        resume_run_id=None if not resume else _find_last_run_id(experiment_config),
    )
    results = trainer.run(df)

    typer.echo("\nResults:")
    for model, scores in results.items():
        typer.echo(f"  {model}: {scores}")


@app.command()
def leaderboard(
    top: int = typer.Option(10, "--top", "-n", help="Number of top results to show"),
    metric: str = typer.Option(None, "--metric", help="Sort by this metric"),
) -> None:
    """Show experiment leaderboard."""
    report = ReportService().format_leaderboard_console(metric=metric, limit=top)
    typer.echo(report)


@app.command()
def registry(action: str = typer.Argument("show", help="show or promote")) -> None:
    """Show or manage model registry."""
    registry_path = Path("workspace/registry.json")
    if not registry_path.exists():
        typer.echo("Registry is empty.")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    if action == "show":
        typer.echo("\n# Model Registry\n")
        for key, entry in registry.items():
            typer.echo(f"**{key}**")
            typer.echo(f"  Model: {entry.get('model')}")
            typer.echo(f"  Run ID: {entry.get('run_id')}")
            typer.echo(f"  Score: {entry.get('score')}")
            typer.echo(f"  Registered: {entry.get('registered_at')}")
            typer.echo("")
    else:
        typer.echo(f"Unknown action: {action}")


@app.command()
def hardware() -> None:
    """Show detected hardware profile."""
    profile = HardwareProfile.detect()
    typer.echo("\n# Hardware Profile")
    typer.echo(f"GPU: {profile.gpu_name or 'None'}")
    typer.echo(f"VRAM: {profile.vram_gb} GB")
    typer.echo(f"RAM: {profile.system_ram_gb} GB")
    typer.echo(f"CPU Cores: {profile.cpu_cores}")


@app.command()
def drift(
    reference: str = typer.Option(..., "--reference", "-r"),
    new: str = typer.Option(..., "--new", "-n"),
    method: str = typer.Option("ks", "--method", "-m", help="ks, psi, domain, or both"),
) -> None:
    """Detect distribution drift between two datasets."""
    ref_df = load_data(reference)
    new_df = load_data(new)

    if method in ("psi", "domain", "both"):
        from tabular_blueprint.pipelines.executor import PipelineExecutor

        executor = PipelineExecutor()
        drift_method_map = {"psi": "psi", "domain": "domain_classifier", "both": "both"}
        hamilton_method = drift_method_map.get(method, method)

        if executor.available:
            report = executor.run_drift(ref_df, new_df, drift_method=hamilton_method)
            if report is not None:
                typer.echo("\n# Drift Detection Report")
                typer.echo(f"Drift detected: {report.drift_detected}")
                if report.psi_report is not None:
                    pr = report.psi_report
                    typer.echo("\n## PSI Report")
                    typer.echo(f"Features tested: {pr.n_features_tested}")
                    typer.echo(f"Moderate drift: {pr.n_moderate}")
                    typer.echo(f"Severe drift: {pr.n_severe}\n")
                    for f_psi in pr.feature_psi:
                        level = f_psi.drift_level.upper()
                        typer.echo(f"{level:>8} | {f_psi.feature} | PSI={f_psi.psi_value:.6f}")
                if report.domain_report is not None:
                    dr_report = report.domain_report
                    typer.echo("\n## Domain Classifier Report")
                    typer.echo(f"Drift detected: {dr_report.drift_detected}")
                    typer.echo(
                        f"AUC score: {dr_report.auc_score:.6f} (threshold: {dr_report.threshold})"
                    )
                    typer.echo(
                        f"Reference rows: {dr_report.n_reference}, Live rows: {dr_report.n_live}"
                    )
                return

    if method in ("ks", "both"):
        from tabular_blueprint.monitoring.drift import DriftDetector

        detector = DriftDetector(ref_df)
        report = detector.detect(new_df)

        typer.echo("\n# KS/Chi2 Drift Detection Report")
        typer.echo(f"Drift detected: {report.drift_detected}")
        typer.echo(f"Columns tested: {report.n_columns_tested}")
        typer.echo(f"Columns drifted: {report.n_drifted}\n")

        for col_result in report.column_results:
            status = "DRIFT" if col_result.drift_detected else "OK"
            typer.echo(
                f"{status} | {col_result.column} | "
                f"p={col_result.p_value:.6f} | {col_result.test_used}"
            )

    if method in ("psi", "both"):
        from tabular_blueprint.monitoring.psi_drift import PSIDriftDetector

        psi_detector = PSIDriftDetector(ref_df)
        psi_report = psi_detector.detect(new_df)

        typer.echo("\n# PSI Drift Detection Report")
        typer.echo(f"Drift detected: {psi_report.drift_detected}")
        typer.echo(f"Features tested: {psi_report.n_features_tested}")
        typer.echo(f"Moderate drift: {psi_report.n_moderate}")
        typer.echo(f"Severe drift: {psi_report.n_severe}\n")

        for f_psi in psi_report.feature_psi:
            level = f_psi.drift_level.upper()
            typer.echo(f"{level:>8} | {f_psi.feature} | PSI={f_psi.psi_value:.6f}")

    if method == "domain":
        from tabular_blueprint.monitoring.domain_classifier import DomainClassifierDriftDetector

        domain_detector = DomainClassifierDriftDetector(ref_df)
        domain_report = domain_detector.detect(new_df)

        typer.echo("\n# Domain Classifier Drift Report")
        typer.echo(f"Drift detected: {domain_report.drift_detected}")
        auc_msg = f"AUC score: {domain_report.auc_score:.6f} (threshold: {domain_report.threshold})"
        typer.echo(auc_msg)
        rows_msg = f"Reference rows: {domain_report.n_reference}, Live rows: {domain_report.n_live}"
        typer.echo(rows_msg)


@app.command()
def state() -> None:
    """Generate and display current experiment state."""
    from tabular_blueprint.engine.state_observer import StateObserver

    observer = StateObserver()
    content = observer.generate()
    typer.echo(content)


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
    from tabular_blueprint.engine.hpo import optimize_model, setup_hpo_components

    try:
        X, y, evaluator, search_space = setup_hpo_components(data_path, target_col, task, model)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e

    model_cls = get_model_class(model)

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

    warnings = result.get("warnings", [])
    if warnings:
        typer.echo(f"HPO warnings: {len(warnings)}")
        for warning in warnings:
            typer.echo(
                f"  [{warning.get('source', 'unknown')}] "
                f"{warning.get('warning_type', 'Warning')}: {warning.get('message', '')}"
            )

    if result.get("param_importances"):
        typer.echo("\nHyperparameter Importance (PedAnova):")
        for item in result["param_importances"]:
            typer.echo(f"  {item['param']}: {item['importance']:.4f}")


@app.command()
def diff(
    id1: str = typer.Argument(..., help="First run ID"),
    id2: str = typer.Argument(..., help="Second run ID"),
    log_path: str = typer.Option("workspace/experiments.jsonl", "--log", help="Path to JSONL log"),
) -> None:
    """Side-by-side comparison of two experiment runs."""
    console = Console()
    events = load_events(log_path)

    run1_events = [e for e in events if e.get("run_id") == id1]
    run2_events = [e for e in events if e.get("run_id") == id2]

    if not run1_events:
        typer.echo(f"Run ID not found: {id1}")
        raise typer.Exit(1)
    if not run2_events:
        typer.echo(f"Run ID not found: {id2}")
        raise typer.Exit(1)

    def _extract_run_summary(events: list[dict]) -> dict:
        summary: dict = {"run_id": events[0].get("run_id", "?")}
        for event in events:
            if event.get("event") == "experiment_started":
                config = event.get("config", {})
                summary["task"] = config.get("task", "?")
                summary["models"] = config.get("models", "?")
                summary["cv_folds"] = config.get("cv_folds", "?")
                summary["metrics"] = config.get("metrics", "?")
                summary["n_rows"] = event.get("n_rows", "?")
                summary["n_features"] = event.get("n_features", "?")
            elif event.get("event") == "model_completed":
                model = event.get("model", "?")
                scores = event.get("cv_scores", {})
                duration = event.get("duration_seconds", "?")
                summary.setdefault("models_detail", []).append(
                    {"name": model, "scores": scores, "duration": duration}
                )
            elif event.get("event") == "leakage_audit":
                summary["leakage_flagged"] = event.get("n_flagged", 0)
        return summary

    s1 = _extract_run_summary(run1_events)
    s2 = _extract_run_summary(run2_events)

    table = Table(title=f"Experiment Diff: {id1} vs {id2}")
    table.add_column("Field", style="bold")
    table.add_column(id1, style="cyan")
    table.add_column(id2, style="green")
    table.add_column("Delta", style="yellow")

    for field in [
        "task",
        "n_rows",
        "n_features",
        "cv_folds",
        "metrics",
        "models",
        "leakage_flagged",
    ]:
        v1 = str(s1.get(field, "?"))
        v2 = str(s2.get(field, "?"))
        delta = "" if v1 == v2 else "CHANGED"
        style = "bold red" if delta else None
        table.add_row(field, v1, v2, delta, style=style)

    models1 = {m["name"]: m for m in s1.get("models_detail", [])}
    models2 = {m["name"]: m for m in s2.get("models_detail", [])}
    all_models = sorted(set(models1.keys()) | set(models2.keys()))

    for model_name in all_models:
        m1 = models1.get(model_name, {})
        m2 = models2.get(model_name, {})
        scores1 = m1.get("scores", {})
        scores2 = m2.get("scores", {})
        all_metrics = sorted(set(scores1.keys()) | set(scores2.keys()))

        for metric in all_metrics:
            sv1 = scores1.get(metric)
            sv2 = scores2.get(metric)
            v1_str = f"{sv1:.4f}" if isinstance(sv1, (int, float)) else "?"
            v2_str = f"{sv2:.4f}" if isinstance(sv2, (int, float)) else "?"
            if isinstance(sv1, (int, float)) and isinstance(sv2, (int, float)):
                delta_val = sv2 - sv1
                delta_str = f"{delta_val:+.4f}"
            else:
                delta_str = "?"
            table.add_row(f"{model_name}/{metric}", v1_str, v2_str, delta_str)

    console.print(table)


@app.command()
def export(
    key: str = typer.Argument(..., help="Registry key (e.g. experiment:classification)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column name"),
) -> None:
    """Export champion model as a portable prediction package."""
    from tabular_blueprint.services.export_service import ExportService

    service = ExportService()
    try:
        export_path = service.export(key, output_dir=output, target_col=target)
        typer.echo(f"Exported to: {export_path}")
        typer.echo("  Model artifact: model.artifact")
        typer.echo("  Preprocessing:  pipelines/preprocessing.py")
        typer.echo("  Predictor:      predictor.py")
        typer.echo("\nUsage:")
        typer.echo(f"  cd {export_path}")
        typer.echo("  python predictor.py data.csv")
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1) from e


def _find_last_run_id(config: ExperimentConfig) -> str | None:
    """Find the most recent run_id from the experiment log."""
    log_path = config.workspace_dir / "experiments.jsonl"
    events = load_events(log_path)
    run_ids = [
        e.get("run_id")
        for e in events
        if e.get("event") == "experiment_started" and e.get("run_id")
    ]
    return run_ids[-1] if run_ids else None


if __name__ == "__main__":
    app()
