"""SHAP-based explainability: global feature importance and local explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


class FeatureImportance(BaseModel):
    """A single feature's importance score from SHAP."""

    feature_name: str
    importance: float


class SHAPExplanationResult(BaseModel):
    """Complete SHAP explanation result with feature rankings and plot paths."""

    model_name: str
    n_features: int
    top_features: list[FeatureImportance]
    plot_paths: list[str]


class Explainer:
    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        output_dir: str = "workspace/artifacts",
    ):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)

    def explain(
        self,
        X: np.ndarray,
        run_id: str,
        max_display: int = 20,
        generate_plots: bool = True,
    ) -> SHAPExplanationResult:
        explainer = self._create_explainer(X)
        shap_values = explainer(X)

        values = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)

        if values.ndim == 3:
            mean_abs = np.mean(np.abs(values), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(values), axis=0)

        n_features = len(mean_abs)
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        sorted_indices = np.argsort(mean_abs)[::-1]
        top_features = [
            FeatureImportance(
                feature_name=self.feature_names[int(i)],
                importance=round(float(mean_abs[int(i)]), 6),
            )
            for i in sorted_indices[:max_display]
        ]

        plot_paths: list[str] = []
        if generate_plots:
            plot_paths = self._generate_plots(shap_values, X, run_id, max_display)

        return SHAPExplanationResult(
            model_name=getattr(self.model, "model_name", "unknown"),
            n_features=n_features,
            top_features=top_features,
            plot_paths=plot_paths,
        )

    def _create_explainer(self, X: np.ndarray) -> Any:
        import shap

        model_type = type(self.model).__name__.lower()
        model_name = getattr(self.model, "model_name", "").lower()

        if any(
            t in model_type or t in model_name
            for t in ["lgbm", "lightgbm", "xgboost", "xgb", "catboost", "gbdt"]
        ):
            model_ref = getattr(self.model, "_model", None) or getattr(
                self.model, "model", self.model
            )
            return shap.TreeExplainer(model_ref)

        background = shap.sample(X, min(100, len(X)))
        return shap.KernelExplainer(
            (
                self.model.predict_proba
                if hasattr(self.model, "predict_proba") and self.model.predict_proba is not None
                else self.model.predict
            ),
            background,
        )

    def _generate_plots(
        self,
        shap_values: Any,
        X: np.ndarray,
        run_id: str,
        max_display: int,
    ) -> list[str]:
        import shap

        plot_dir = self.output_dir / f"shap_{run_id}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            beeswarm_path = str(plot_dir / "beeswarm.png")
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(beeswarm_path)

            for i in range(min(5, X.shape[1])):
                dep_path = str(plot_dir / f"dependence_{i}.png")
                shap.plots.scatter(shap_values[:, i], show=False)
                plt.tight_layout()
                plt.savefig(dep_path, dpi=150, bbox_inches="tight")
                plt.close()
                paths.append(dep_path)
        except (ValueError, RuntimeError) as e:
            import logging

            logging.getLogger(__name__).warning("SHAP dependence plot failed: %s", e)

        return paths
