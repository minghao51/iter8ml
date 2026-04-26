"""SHAP explainability as a standalone service."""

from typing import Any

import numpy as np

from tabular_blueprint.config import ExperimentConfig
from tabular_blueprint.engine.tracker import Tracker


class ExplainabilityService:
    def __init__(self, config: ExperimentConfig, tracker: Tracker):
        self.config = config
        self.tracker = tracker

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        run_id: str,
        feature_names: list[str],
    ) -> None:
        try:
            from tabular_blueprint.monitoring.explainability import Explainer

            explainer = Explainer(
                model,
                feature_names=feature_names,
                output_dir=str(self.config.workspace_dir / "artifacts"),
            )
            shap_result = explainer.explain(X, run_id, generate_plots=True)

            self.tracker.log_event(
                {
                    "event": "shap_explainability",
                    "run_id": run_id,
                    "model": getattr(model, "model_name", "unknown"),
                    "n_features": shap_result.n_features,
                    "top_features": [
                        {"name": f.feature_name, "importance": f.importance}
                        for f in shap_result.top_features[:10]
                    ],
                    "plot_paths": shap_result.plot_paths,
                }
            )
        except Exception as e:
            self.tracker.log_event(
                {
                    "event": "shap_failed",
                    "run_id": run_id,
                    "error": str(e),
                }
            )
