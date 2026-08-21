"""Analysis commands — drift, leaderboard, state, diff."""

from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from iter8ml.data.loader import load_data
from iter8ml.services.reporting import ReportService
from iter8ml.utils.io import load_events
from iter8ml.workspace import Workspace

from .main import app


def _print_psi_report(pr: Any) -> None:
    typer.echo("## PSI Report")
    typer.echo(f"Features tested: {pr.n_features_tested}")
    typer.echo(f"Moderate drift: {pr.n_moderate}")
    typer.echo(f"Severe drift: {pr.n_severe}\n")
    for f_psi in pr.feature_psi:
        level = f_psi.drift_level.upper()
        typer.echo(f"{level:>8} | {f_psi.feature} | PSI={f_psi.psi_value:.6f}")


def _print_domain_report(dr: Any) -> None:
    typer.echo("## Domain Classifier Report")
    typer.echo(f"Drift detected: {dr.drift_detected}")
    typer.echo(f"AUC score: {dr.auc_score:.6f} (threshold: {dr.threshold})")
    typer.echo(f"Reference rows: {dr.n_reference}, Live rows: {dr.n_live}")


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
        from iter8ml.engine.pipelines.executor import PipelineExecutor

        executor = PipelineExecutor()
        drift_method_map = {"psi": "psi", "domain": "domain_classifier", "both": "both"}
        hamilton_method = drift_method_map.get(method, method)

        if executor.available:
            report = executor.run_drift(ref_df, new_df, drift_method=hamilton_method)
            if report is not None:
                typer.echo("\n# Drift Detection Report")
                typer.echo(f"Drift detected: {report.drift_detected}")
                if report.psi_report is not None:
                    _print_psi_report(report.psi_report)
                if report.domain_report is not None:
                    _print_domain_report(report.domain_report)
                return

    if method in ("ks", "both"):
        from iter8ml.analysis.drift import DriftDetector

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
        from iter8ml.analysis.psi import PSIDriftDetector

        psi_detector = PSIDriftDetector(ref_df)
        psi_report = psi_detector.detect(new_df)
        typer.echo("\n# PSI Drift Detection Report")
        typer.echo(f"Drift detected: {psi_report.drift_detected}")
        _print_psi_report(psi_report)

    if method == "domain":
        from iter8ml.analysis.domain_classifier import DomainClassifierDriftDetector

        domain_detector = DomainClassifierDriftDetector(ref_df)
        domain_report = domain_detector.detect(new_df)
        typer.echo("\n# Domain Classifier Drift Report")
        _print_domain_report(domain_report)


@app.command()
def leaderboard(
    top: int = typer.Option(10, "--top", "-n", help="Number of top results to show"),
    metric: str = typer.Option(None, "--metric", help="Sort by this metric"),
) -> None:
    """Show experiment leaderboard."""
    ws = Workspace()
    report = ReportService(workspace=ws).format_leaderboard_console(metric=metric, limit=top)
    typer.echo(report)


@app.command()
def state(
    llm: bool = typer.Option(
        False, "--llm/--no-llm", help="Include LLM-generated commentary (requires LLM config)."
    ),
) -> None:
    """Generate and display current experiment state."""
    from iter8ml.engine.state_observer import StateObserver

    ws = Workspace()
    observer = StateObserver(workspace=ws, llm_enabled=llm)
    content = observer.generate()
    typer.echo(content)


@app.command()
def diff(
    id1: str = typer.Argument(..., help="First run ID"),
    id2: str = typer.Argument(..., help="Second run ID"),
    log_path: str | None = typer.Option(None, "--log", help="Path to JSONL log"),
) -> None:
    """Side-by-side comparison of two experiment runs."""
    console = Console()
    if log_path is None:
        log_path = str(Workspace().experiments_path)
    events = load_events(log_path)

    run1_events = [e for e in events if e.get("run_id") == id1]
    run2_events = [e for e in events if e.get("run_id") == id2]

    if not run1_events:
        typer.echo(f"Run ID not found: {id1}")
        raise typer.Exit(1)
    if not run2_events:
        typer.echo(f"Run ID not found: {id2}")
        raise typer.Exit(1)

    def _extract_run_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"run_id": events[0].get("run_id", "?")}
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
