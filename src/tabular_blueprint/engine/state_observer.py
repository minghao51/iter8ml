"""StateObserver: generates LLM-readable experiment state summaries."""

from pathlib import Path
from typing import Any

from tabular_blueprint.services.report_service import ExperimentReport, ReportService


class StateObserver:
    """Generates workspace/current_state.md after every trainer run."""

    def __init__(
        self,
        log_path: str = "workspace/experiments.jsonl",
        registry_path: str = "workspace/registry.json",
        output_path: str = "workspace/current_state.md",
        leaderboard_path: str | None = None,
        llm_enabled: bool = False,
        llm_model: str = "claude-sonnet-4-20250514",
        llm_api_key_env: str = "",
        llm_api_base: str | None = None,
    ):
        self.log_path = Path(log_path)
        self.registry_path = Path(registry_path)
        self.output_path = Path(output_path)
        self.leaderboard_path = (
            Path(leaderboard_path)
            if leaderboard_path is not None
            else self.output_path.with_name("leaderboard.md")
        )
        self._llm_enabled = llm_enabled
        self._llm_model = llm_model
        self._llm_api_key_env = llm_api_key_env
        self._llm_api_base = llm_api_base

    def generate(self) -> str:
        """Read experiment state and render current_state.md and leaderboard.md."""
        self._report_svc = ReportService(
            log_path=self.log_path,
            registry_path=self.registry_path,
        )
        report = self._report_svc.build_report()

        if not report.latest_run:
            content = "# No experiments run yet\n"
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(content)
            return content

        state_content = self._render_state(report)
        leaderboard_content = self._report_svc.format_leaderboard_markdown()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(state_content)
        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        self.leaderboard_path.write_text(leaderboard_content)
        return state_content

    def _render_state(self, report: ExperimentReport) -> str:
        latest = report.latest_run
        if latest is None:
            raise ValueError("Cannot render state: report has no latest_run")

        lines = [
            "# Current Experiment State\n",
            f"**Task:** {latest.task.title()}",
            f"**Dataset:** {latest.dataset}",
            f"**Rows / Features:** {latest.n_rows} / {latest.n_features}",
            f"**Latest Run ID:** {latest.run_id}\n",
            "## Leaderboard\n",
            "| Rank | Model | Run ID | Primary Metric | Score | Duration |",
            "|---|---|---|---|---|---|",
        ]

        for index, entry in enumerate(report.leaderboard, start=1):
            lines.append(
                f"| {index} | {entry.model} | {entry.run_id} | {entry.primary_metric} "
                f"| {entry.primary_score:.4f} | {entry.duration_seconds}s |"
            )

        lines.extend(
            [
                "\n## Resource Status\n",
                f"Device: {latest.hardware.get('device', 'cpu')}",
                f"VRAM Used: {latest.hardware.get('vram_used_gb', 0)} GB",
            ]
        )

        all_events = self._load_all_events()
        leakage = [e for e in all_events if e.get("event") == "leakage_audit"]
        if leakage:
            latest_leakage = leakage[-1]
            lines.extend(
                [
                    "\n## Leakage Audit\n",
                    f"Features tested: {latest_leakage.get('n_flagged', 'N/A')} flagged",
                    f"Baseline score: {latest_leakage.get('baseline_score', 'N/A')}",
                ]
            )

        target_transforms = [e for e in all_events if e.get("event") == "target_transform"]
        if target_transforms:
            latest_tt = target_transforms[-1]
            if latest_tt.get("applied"):
                orig_skew = latest_tt.get("original_skewness", "N/A")
                trans_skew = latest_tt.get("transformed_skewness", "N/A")
                orig_str = (
                    f"{orig_skew:.4f}" if isinstance(orig_skew, (int, float)) else str(orig_skew)
                )
                trans_str = (
                    f"{trans_skew:.4f}" if isinstance(trans_skew, (int, float)) else str(trans_skew)
                )
                lines.extend(
                    [
                        "\n## Target Transform\n",
                        f"Method: {latest_tt.get('method', 'N/A')}",
                        f"Original skewness: {orig_str}",
                        f"Transformed skewness: {trans_str}",
                    ]
                )

        afe_events = [e for e in all_events if e.get("event") == "afe_completed"]
        if afe_events:
            latest_afe = afe_events[-1]
            lines.extend(
                [
                    "\n## Automated Feature Engineering\n",
                    f"Candidates tested: {latest_afe.get('n_candidates_tested', 0)}",
                    f"Candidates kept: {latest_afe.get('n_candidates_kept', 0)}",
                    f"New features: {', '.join(latest_afe.get('new_feature_names', [])) or 'None'}",
                ]
            )

        shap_events = [e for e in all_events if e.get("event") == "shap_explainability"]
        if shap_events:
            latest_shap = shap_events[-1]
            top_feats = latest_shap.get("top_features", [])
            lines.extend(
                [
                    "\n## SHAP Explainability\n",
                    f"Model: {latest_shap.get('model', 'N/A')}",
                    f"Features analyzed: {latest_shap.get('n_features', 'N/A')}",
                    "Top features:",
                ]
            )
            for feat in top_feats[:10]:
                lines.append(f"  - {feat.get('name', '?')}: {feat.get('importance', 0):.4f}")

            plot_paths = latest_shap.get("plot_paths", [])
            if plot_paths:
                lines.append("\n  Plots:")
                for pp in plot_paths:
                    lines.append(f"  - {Path(pp).name}")

        if self._llm_enabled and (shap_events or report.latest_run):
            lines.extend(self._render_llm_commentary(all_events, report))

        drift_events = [e for e in all_events if e.get("event") == "drift_check"]
        if drift_events:
            lines.append("\n## Drift Detection\n")
            for de in drift_events:
                method = de.get("method", "unknown")
                detected = de.get("drift_detected", False)
                status = "**DRIFT DETECTED**" if detected else "No drift"
                if method == "psi":
                    lines.append(
                        f"- PSI ({method}): {status} | "
                        f"moderate={de.get('n_moderate', 0)}, severe={de.get('n_severe', 0)}"
                    )
                elif method == "domain_classifier":
                    lines.append(
                        f"- Domain Classifier: {status} | "
                        f"AUC={de.get('auc_score', 0):.4f} (threshold={de.get('threshold', 0.7)})"
                    )

        if report.registry:
            lines.append("\n## Registered Champions\n")
            for key, entry in report.registry.items():
                lines.append(f"- **{key}**: {entry.get('model')} (score: {entry.get('score')})")

        lines.extend(
            [
                "\n## Data Pipeline\n",
                "```mermaid",
                self._render_pipeline_dag(),
                "```",
            ]
        )

        return "\n".join(lines) + "\n"

    def _render_pipeline_dag(self) -> str:
        try:
            from tabular_blueprint.pipelines import visualize_pipeline

            result = visualize_pipeline(output_format="mermaid")
            if isinstance(result, str):
                return result
            return str(result)
        except ImportError:
            return (
                "graph TD\n"
                "    A[Raw Data] --> B[Fill Nulls]\n"
                "    B --> C[Decompose Dates]\n"
                "    C --> D[Encode Categoricals]\n"
                "    D --> E[Processed Data]"
            )

    def _get_agent(self) -> Any:
        from tabular_blueprint.llm import LLMAgentConfig, TabularAgent

        return TabularAgent(
            LLMAgentConfig(
                enabled=True,
                model=self._llm_model,
                api_key_env=self._llm_api_key_env,
                api_base=self._llm_api_base,
            )
        )

    def _render_llm_commentary(self, all_events: list[dict], report: ExperimentReport) -> list[str]:
        agent = self._get_agent()
        lines: list[str] = ["\n## LLM Commentary\n"]

        shap_events = [e for e in all_events if e.get("event") == "shap_explainability"]
        if shap_events:
            latest_shap = shap_events[-1]
            commentary = agent.explain_shap(
                top_features=latest_shap.get("top_features", []),
                model_name=latest_shap.get("model", "unknown"),
                task=latest_shap.get("task", "classification"),
            )
            if commentary.content:
                lines.append(f"**SHAP Explanation:** {commentary.content}\n")

        if report.latest_run and report.latest_run.cv_scores:
            commentary = agent.explain_performance(
                cv_scores=report.latest_run.cv_scores,
                model_name=report.latest_run.model,
                task=report.latest_run.task,
            )
            if commentary.content:
                lines.append(f"**Performance Commentary:** {commentary.content}\n")

        return lines

    def _load_all_events(self) -> list[dict]:
        from tabular_blueprint.utils.jsonl import load_events

        return load_events(self.log_path)
