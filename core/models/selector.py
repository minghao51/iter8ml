"""ModelSelector: hardware-aware and data-size-aware model routing."""

from typing import Literal


class ModelSelector:
    """
    Given a dataset profile and hardware profile, returns an ordered list
    of models to run, from fastest/cheapest to most expensive.

    Routing logic per spec:
      n_rows < 10k        -> [TabPFN, CatBoost, LightGBM]
      10k <= n_rows < 500k -> [CatBoost, LightGBM, XGBoost]
      n_rows >= 500k       -> [LightGBM, XGBoost]
      vram_gb > 12         -> append FT-Transformer (n_rows > 50k)
    """

    TABPFN_ROW_LIMIT = 10_000
    FT_TRANSFORMER_ROW_MIN = 50_000

    def select(
        self,
        n_rows: int,
        task: Literal["classification", "regression"],
        vram_gb: float = 0.0,
    ) -> list[str]:
        models = []

        if n_rows < self.TABPFN_ROW_LIMIT:
            models.extend(["tabpfn", "catboost", "lightgbm"])
        elif n_rows < 500_000:
            models.extend(["catboost", "lightgbm", "xgboost"])
        else:
            models.extend(["lightgbm", "xgboost"])

        if vram_gb > 12 and n_rows >= self.FT_TRANSFORMER_ROW_MIN:
            models.append("ft_transformer")

        return models
