"""StateRenderer: pure markdown rendering of experiment state sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iter8ml.services.reporting import ExperimentReport


class StateRenderer:
    """Static methods that render experiment state as markdown lines."""

    @staticmethod
    def render_header(latest: Any) -> list[str]:
        return [
            "# Current Experiment State\n",
            f"**Task:** {latest.task.title()}",
            f"**Dataset:** {latest.dataset}",
            f"**Rows / Features:** {latest.n_rows} / {latest.n_features}",
            f"**Latest Run ID:** {latest.run_id}\n",
        ]

    @staticmethod
    def render_leaderboard_section(report: ExperimentReport) -> list[str]:
        lines = [
            "## Leaderboard\n",
            "| Rank | Model | Run ID | Primary Metric | Score | Duration |",
            "|---|---|---|---|---|---|",
        ]
        for index, entry in enumerate(report.leaderboard, start=1):
            lines.append(
                f"| {index} | {entry.model} | {entry.run_id} | {entry.primary_metric} "
                f"| {entry.primary_score:.4f} | {entry.duration_seconds}s |"
            )
        return lines

    @staticmethod
    def render_resource_section(latest: Any) -> list[str]:
        return [
            "\n## Resource Status\n",
            f"Device: {latest.hardware.get('device', 'cpu')}",
            f"VRAM Used: {latest.hardware.get('vram_used_gb', 0)} GB",
        ]

    @staticmethod
    def render_leakage_section(all_events: list[dict[str, Any]]) -> list[str]:
        leakage = [e for e in all_events if e.get("event") == "leakage_audit"]
        if not leakage:
            return []
        latest_leakage = leakage[-1]
        return [
            "\n## Leakage Audit\n",
            f"Features tested: {latest_leakage.get('n_flagged', 'N/A')} flagged",
            f"Baseline score: {latest_leakage.get('baseline_score', 'N/A')}",
        ]

    @staticmethod
    def render_target_transform_section(all_events: list[dict[str, Any]]) -> list[str]:
        target_transforms = [e for e in all_events if e.get("event") == "target_transform"]
        if not target_transforms:
            return []
        latest_tt = target_transforms[-1]
        if not latest_tt.get("applied"):
            return []
        orig_skew = latest_tt.get("original_skewness", "N/A")
        trans_skew = latest_tt.get("transformed_skewness", "N/A")
        orig_str = f"{orig_skew:.4f}" if isinstance(orig_skew, (int, float)) else str(orig_skew)
        trans_str = f"{trans_skew:.4f}" if isinstance(trans_skew, (int, float)) else str(trans_skew)
        return [
            "\n## Target Transform\n",
            f"Method: {latest_tt.get('method', 'N/A')}",
            f"Original skewness: {orig_str}",
            f"Transformed skewness: {trans_str}",
        ]

    @staticmethod
    def render_afe_section(all_events: list[dict[str, Any]]) -> list[str]:
        afe_events = [e for e in all_events if e.get("event") == "afe_completed"]
        if not afe_events:
            return []
        latest_afe = afe_events[-1]
        return [
            "\n## Automated Feature Engineering\n",
            f"Candidates tested: {latest_afe.get('n_candidates_tested', 0)}",
            f"Candidates kept: {latest_afe.get('n_candidates_kept', 0)}",
            f"New features: {', '.join(latest_afe.get('new_feature_names', [])) or 'None'}",
        ]

    @staticmethod
    def render_shap_section(all_events: list[dict[str, Any]]) -> list[str]:
        shap_events = [e for e in all_events if e.get("event") == "shap_explainability"]
        if not shap_events:
            return []
        latest_shap = shap_events[-1]
        top_feats = latest_shap.get("top_features", [])
        lines = [
            "\n## SHAP Explainability\n",
            f"Model: {latest_shap.get('model', 'N/A')}",
            f"Features analyzed: {latest_shap.get('n_features', 'N/A')}",
            "Top features:",
        ]
        for feat in top_feats[:10]:
            lines.append(f"  - {feat.get('name', '?')}: {feat.get('importance', 0):.4f}")

        plot_paths = latest_shap.get("plot_paths", [])
        if plot_paths:
            lines.append("\n  Plots:")
            for pp in plot_paths:
                lines.append(f"  - {Path(pp).name}")
        return lines

    @staticmethod
    def render_drift_section(all_events: list[dict[str, Any]]) -> list[str]:
        drift_events = [e for e in all_events if e.get("event") == "drift_check"]
        if not drift_events:
            return []
        lines = ["\n## Drift Detection\n"]
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
        return lines

    @staticmethod
    def render_registry_section(report: ExperimentReport) -> list[str]:
        if not report.registry:
            return []
        lines = ["\n## Registered Champions\n"]
        for key, entry in report.registry.items():
            lines.append(f"- **{key}**: {entry.get('model')} (score: {entry.get('score')})")
        return lines

    @staticmethod
    def render_pipeline_section() -> list[str]:
        return [
            "\n## Data Pipeline\n",
            "```mermaid",
            StateRenderer.render_pipeline_dag(),
            "```",
        ]

    @staticmethod
    def render_pipeline_dag() -> str:
        try:
            from iter8ml.engine.pipelines import visualize_pipeline

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

    @staticmethod
    def render_state(
        report: ExperimentReport,
        all_events: list[dict[str, Any]],
        llm_lines: list[str],
    ) -> str:
        latest = report.latest_run
        if latest is None:
            raise ValueError("Cannot render state: report has no latest_run")

        lines = StateRenderer.render_header(latest)
        lines.extend(StateRenderer.render_leaderboard_section(report))
        lines.extend(StateRenderer.render_resource_section(latest))
        lines.extend(StateRenderer.render_leakage_section(all_events))
        lines.extend(StateRenderer.render_target_transform_section(all_events))
        lines.extend(StateRenderer.render_afe_section(all_events))
        lines.extend(StateRenderer.render_shap_section(all_events))
        lines.extend(llm_lines)
        lines.extend(StateRenderer.render_drift_section(all_events))
        lines.extend(StateRenderer.render_registry_section(report))
        lines.extend(StateRenderer.render_pipeline_section())

        return "\n".join(lines) + "\n"
