"""Tier 1 for transactions: LightGBM on raw + engineered features."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class LgbmScorer:
    def __init__(self, feature_cols: Sequence[str], **params):
        self.cols = list(feature_cols)
        # no class re-weighting: keeps probabilities close to calibrated, and the
        # cost-optimal threshold is chosen on validation data instead
        self.params = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 64,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                       "verbose": -1, **params}
        self.model = None

    def fit(self, df: pd.DataFrame) -> "LgbmScorer":
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(**self.params).fit(df[self.cols], df["isFraud"])
        return self

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(rows[self.cols])[:, 1]

    def explain(self, row: pd.DataFrame, k: int = 5) -> list[tuple[str, float]]:
        """Top-k SHAP contributions (log-odds) for one row, via LightGBM's built-in TreeSHAP."""
        contrib = self.model.predict(row[self.cols], pred_contrib=True)[0][:-1]  # last = bias
        top = np.argsort(-np.abs(contrib))[:k]
        return [(self.cols[i], round(float(contrib[i]), 3)) for i in top]
