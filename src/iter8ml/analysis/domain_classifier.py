"""Domain classifier drift detection: multivariate drift via classifier AUC."""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class DomainDriftReport(BaseModel):
    """Multivariate drift detection result from domain classifier AUC."""

    drift_detected: bool
    auc_score: float
    threshold: float
    n_reference: int
    n_live: int


DOMAIN_AUC_THRESHOLD = 0.7


class DomainClassifierDriftDetector:
    def __init__(
        self,
        reference_df: pl.DataFrame,
        threshold: float = DOMAIN_AUC_THRESHOLD,
        random_seed: int = 42,
    ):
        self.reference_df = reference_df
        self.threshold = threshold
        self.random_seed = random_seed

    def detect(self, live_df: pl.DataFrame) -> DomainDriftReport:
        common_cols = sorted(
            c
            for c in set(self.reference_df.columns) & set(live_df.columns)
            if self.reference_df[c].dtype.is_numeric()
        )

        if not common_cols:
            return DomainDriftReport(
                drift_detected=False,
                auc_score=0.5,
                threshold=self.threshold,
                n_reference=len(self.reference_df),
                n_live=len(live_df),
            )

        ref_np = self.reference_df.select(common_cols).drop_nulls().to_numpy()
        live_np = live_df.select(common_cols).drop_nulls().to_numpy()

        X = np.vstack([ref_np, live_np])
        y = np.concatenate([np.zeros(len(ref_np)), np.ones(len(live_np))])

        n_min = min(len(ref_np), len(live_np))
        n_folds = min(5, max(2, n_min))

        try:
            scores = cross_val_score(
                LogisticRegression(random_state=self.random_seed, max_iter=1000),
                X,
                y,
                cv=n_folds,
                scoring="roc_auc",
            )
            auc_score = float(np.mean(scores))
        except ValueError:
            auc_score = 0.5

        return DomainDriftReport(
            drift_detected=auc_score > self.threshold,
            auc_score=round(auc_score, 6),
            threshold=self.threshold,
            n_reference=len(ref_np),
            n_live=len(live_np),
        )
