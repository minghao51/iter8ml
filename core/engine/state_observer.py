"""StateObserver: generates LLM-readable experiment state summaries."""

from pathlib import Path

from core.services.report_service import ExperimentReport, ReportService


class StateObserver:
    """Generates workspace/current_state.md after every trainer run."""

    def __init__(
        self,
        log_path: str = "workspace/experiments.jsonl",
        registry_path: str = "workspace/registry.json",
        output_path: str = "workspace/current_state.md",
        leaderboard_path: str | None = None,
    ):
        self.log_path = Path(log_path)
        self.registry_path = Path(registry_path)
        self.output_path = Path(output_path)
        self.leaderboard_path = (
            Path(leaderboard_path)
            if leaderboard_path is not None
            else self.output_path.with_name("leaderboard.md")
        )

    def generate(self) -> str:
        """Read experiment state and render current_state.md and leaderboard.md."""
        report = ReportService(
            log_path=self.log_path,
            registry_path=self.registry_path,
        ).build_report()

        if not report.latest_run:
            content = "# No experiments run yet\n"
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(content)
            return content

        state_content = self._render_state(report)
        leaderboard_content = self._render_leaderboard()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(state_content)
        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        self.leaderboard_path.write_text(leaderboard_content)
        return state_content

    def _render_state(self, report: ExperimentReport) -> str:
        latest = report.latest_run
        assert latest is not None

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

        if report.registry:
            lines.append("\n## Registered Champions\n")
            for key, entry in report.registry.items():
                lines.append(f"- **{key}**: {entry.get('model')} (score: {entry.get('score')})")

        return "\n".join(lines) + "\n"

    def _render_leaderboard(self) -> str:
        return ReportService(
            log_path=self.log_path,
            registry_path=self.registry_path,
        ).format_leaderboard_markdown()
