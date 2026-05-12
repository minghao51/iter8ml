"""LLM TabularAgent: natural language explanations for model results."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from iter8ml.config import DEFAULT_LLM_MODEL


class LLMCommentary(BaseModel):
    """A single section of LLM-generated natural language commentary."""

    section: str
    content: str


class LLMAgentConfig(BaseModel):
    """Configuration for the LLM-powered TabularAgent."""

    enabled: bool = False
    model: str = Field(
        default_factory=lambda: os.getenv("TABBLUEPRINT_LLM_MODEL", DEFAULT_LLM_MODEL)
    )
    api_key_env: str = ""
    api_base: str | None = None


class TabularAgent:
    def __init__(self, config: LLMAgentConfig | None = None):
        self.config = config or LLMAgentConfig()

    def explain_shap(
        self,
        top_features: list[dict[str, Any]],
        model_name: str,
        task: str,
    ) -> LLMCommentary:
        if not self.config.enabled:
            return LLMCommentary(section="shap_explanation", content="")

        prompt = _build_shap_prompt(top_features, model_name, task)
        response = self._call_llm(prompt)
        return LLMCommentary(section="shap_explanation", content=response)

    def explain_performance(
        self,
        cv_scores: dict[str, float],
        model_name: str,
        task: str,
        baseline_scores: dict[str, float] | None = None,
    ) -> LLMCommentary:
        if not self.config.enabled:
            return LLMCommentary(section="performance_commentary", content="")

        prompt = _build_performance_prompt(cv_scores, model_name, task, baseline_scores)
        response = self._call_llm(prompt)
        return LLMCommentary(section="performance_commentary", content=response)

    def summarize_features(
        self,
        feature_importance: list[dict[str, Any]],
        n_top: int = 5,
    ) -> LLMCommentary:
        if not self.config.enabled:
            return LLMCommentary(section="feature_summary", content="")

        prompt = _build_feature_summary_prompt(feature_importance, n_top)
        response = self._call_llm(prompt)
        return LLMCommentary(section="feature_summary", content=response)

    def _call_llm(self, prompt: str) -> str:
        try:
            import litellm

            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
            }
            if self.config.api_key_env:
                import os

                api_key = os.environ.get(self.config.api_key_env, "")
                if not api_key:
                    raise ValueError(
                        f"API key not found in environment variable '{self.config.api_key_env}'. "
                        "Set the variable or disable LLM features."
                    )
                kwargs["api_key"] = api_key
            if self.config.api_base:
                kwargs["api_base"] = self.config.api_base

            response = litellm.completion(**kwargs)
            return response.choices[0].message.content or ""
        except ImportError:
            return "[LLM disabled: litellm package not installed]"
        except Exception as e:
            return f"[LLM error: {e}]"


def _build_shap_prompt(
    top_features: list[dict[str, Any]],
    model_name: str,
    task: str,
) -> str:
    feature_lines = "\n".join(
        f"- {f['feature_name']}: importance={f['importance']:.4f}" for f in top_features[:10]
    )
    return (
        f"You are a machine learning explainability assistant.\n"
        f"Model: {model_name}\n"
        f"Task: {task}\n"
        f"Top features by SHAP importance:\n{feature_lines}\n\n"
        f"Provide a concise (2-3 sentence) natural language explanation of what these "
        f"feature importances tell us about the model's decision process."
    )


def _build_performance_prompt(
    cv_scores: dict[str, float],
    model_name: str,
    task: str,
    baseline_scores: dict[str, float] | None,
) -> str:
    scores_str = ", ".join(f"{k}={v:.4f}" for k, v in cv_scores.items())
    baseline_str = ""
    if baseline_scores:
        baseline_str = "\nBaseline scores: " + ", ".join(
            f"{k}={v:.4f}" for k, v in baseline_scores.items()
        )

    return (
        f"You are a machine learning performance analyst.\n"
        f"Model: {model_name}, Task: {task}\n"
        f"CV scores: {scores_str}{baseline_str}\n\n"
        f"Provide a brief (2-3 sentence) commentary on this model's performance, "
        f"including how it compares to baselines if available."
    )


def _build_feature_summary_prompt(
    feature_importance: list[dict[str, Any]],
    n_top: int,
) -> str:
    feature_lines = "\n".join(
        f"- {f['feature_name']}: importance={f['importance']:.4f}"
        for f in feature_importance[:n_top]
    )
    return (
        f"Summarize the top {n_top} most important features for this model:\n"
        f"{feature_lines}\n\n"
        f"Provide a concise 1-2 sentence summary of what drives this model's predictions."
    )
