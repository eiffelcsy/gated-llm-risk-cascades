"""Tier 1 for transactions: LightGBM. Report p99 single-row latency, not just batch."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class LgbmScorer:
    def __init__(self, feature_cols: Sequence[str], **params):
        self.cols = list(feature_cols)
        self.params = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 64, **params}
        self.model = None

    def fit(self, df_train, df_val=None) -> "LgbmScorer":
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(**self.params)
        eval_set = [(df_val[self.cols], df_val["isFraud"])] if df_val is not None else None
        self.model.fit(df_train[self.cols], df_train["isFraud"], eval_set=eval_set)
        return self

    def score(self, rows) -> np.ndarray:
        return self.model.predict_proba(rows[self.cols])[:, 1]
