"""ModelSelector: hardware-aware and data-size-aware model routing."""

from typing import Literal


class ModelSelector:
    """
    Given a dataset profile and hardware profile, returns an ordered list
    of models to run, from fastest/cheapest to most expensive.

    Routing logic per spec:
      n_rows < 50k + GPU   -> [TabPFN, CatBoost, LightGBM]
      n_rows < 50k no GPU  -> [CatBoost, LightGBM, XGBoost]
      50k <= n_rows < 500k -> [CatBoost, LightGBM, XGBoost]
      n_rows >= 500k       -> [LightGBM, XGBoost]
      vram_gb > 12         -> append FT-Transformer (n_rows > 50k)
      vram_gb > 8          -> append TabNet/NODE as DL alternatives
    """

    TABPFN_ROW_LIMIT = 50_000
    FT_TRANSFORMER_ROW_MIN = 50_000

    def _has_gpu(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()  # type: ignore[no-any-return]
        except ImportError:
            return False

    def select(
        self,
        n_rows: int,
        task: Literal["classification", "regression"],
        vram_gb: float = 0.0,
        include_baselines: bool = True,
    ) -> list[str]:
        models = []
        has_gpu = self._has_gpu()

        if include_baselines:
            models.extend(["naive_baseline", "linear_baseline"])

        # TabPFN is now always included if GPU is present, regardless of row count.
        # The Trainer will handle the warning if n_rows > TABPFN_ROW_LIMIT.
        if has_gpu:
            models.append("tabpfn")

        if n_rows < 500_000:
            models.extend(["catboost", "lightgbm", "xgboost"])
        else:
            models.extend(["lightgbm", "xgboost"])

        if vram_gb > 12 and n_rows >= self.FT_TRANSFORMER_ROW_MIN:
            models.append("ft_transformer")

        if vram_gb > 8:
            models.append("tabnet")

        seen: set[str] = set()
        result: list[str] = []
        for x in models:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result
