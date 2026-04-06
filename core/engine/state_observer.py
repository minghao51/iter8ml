"""StateObserver: generates LLM-readable experiment state summaries."""

import json
from pathlib import Path

from core.utils.jsonl import load_events


class StateObserver:
    """Generates workspace/current_state.md after every trainer run."""

    def __init__(
        self,
        log_path: str = "workspace/experiments.jsonl",
        registry_path: str = "workspace/registry.json",
        output_path: str = "workspace/current_state.md",
    ):
        self.log_path = Path(log_path)
        self.registry_path = Path(registry_path)
        self.output_path = Path(output_path)

    def generate(self) -> str:
        """Read JSONL events and registry, produce current_state.md."""
        events = self._load_events()
        registry = self._load_registry()
        completed = [e for e in events if e.get("event") == "model_completed"]

        if not completed:
            return "# No experiments run yet\n"

        latest = completed[-1]
        ranked = sorted(
            completed,
            key=lambda x: x.get("cv_scores", {}).get(
                "roc_auc", x.get("cv_scores", {}).get("r2", 0)
            ),
            reverse=True,
        )
        lines = [
            "## Current Experiment State\n",
            f"**Task:** {latest.get('task', '?').title()}",
            f"**Dataset:** {latest.get('dataset', 'unknown')}",
            f"({latest.get('n_rows', '?')} rows, {latest.get('n_features', '?')} features)\n",
            "### Leaderboard (sorted by primary metric)\n",
            "| Model | ROC-AUC | F1 | Duration |",
            "|---|---|---|---|",
        ]

        for e in ranked:
            scores = e.get("cv_scores", {})
            roc = scores.get("roc_auc", scores.get("r2", 0))
            f1 = scores.get("f1_macro", "-")
            dur = e.get("duration_seconds", "?")
            lines.append(f"| {e.get('model', '?')} | {roc:.4f} | {f1} | {dur}s |")

        hw = latest.get("hardware", {})
        lines.extend(
            [
                "\n### Resource Status\n",
                f"Device: {hw.get('device', 'cpu')}",
                f"VRAM Used: {hw.get('vram_used_gb', 0)} GB",
            ]
        )

        if self.registry_path.exists():
            lines.append("\n### Registered Champions\n")
            for key, entry in registry.items():
                lines.append(f"- **{key}**: {entry.get('model')} (score: {entry.get('score')})")

        content = "\n".join(lines) + "\n"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(content)
        return content

    def _load_events(self) -> list[dict]:
        return load_events(self.log_path)

    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}
