"""Override gate: should tier 2's disagreement with tier 1 be trusted?

Trained only on cases where tier 2 disagreed with tier 1 on a validation
split. Target = 1 if tier 2 was right. Features = tier-1 score, its distance
from the threshold, tier 2's verdict, and the track's evidence features.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class OverrideGate:
    def __init__(self, threshold: float = 0.5, min_trust: float = 0.5):
        self.threshold = threshold
        self.min_trust = min_trust
        self.model = LogisticRegression(max_iter=1000)
        self._constant: float | None = None

    def _features(self, s: np.ndarray, v: np.ndarray, ev: np.ndarray) -> np.ndarray:
        ev = np.asarray(ev, dtype=float).reshape(len(s), -1)
        return np.column_stack([s, np.abs(s - self.threshold), v, ev])

    def fit(self, s: np.ndarray, v: np.ndarray, ev: np.ndarray, y: np.ndarray) -> "OverrideGate":
        """Fit on disagreements only; y is the true label."""
        s, v, y = map(np.asarray, (s, v, y))
        ev = np.asarray(ev, dtype=float).reshape(len(s), -1)
        disagree = v != (s >= self.threshold).astype(int)
        if disagree.sum() == 0:
            self._constant = 0.0
            return self
        target = (v[disagree] == y[disagree]).astype(int)
        if target.min() == target.max():
            # tier 2 was always right (or always wrong) on disagreements
            self._constant = float(target[0])
            return self
        self.model.fit(self._features(s[disagree], v[disagree], ev[disagree]), target)
        return self

    def trust(self, s: np.ndarray, v: np.ndarray, ev: np.ndarray) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(s), self._constant)
        return self.model.predict_proba(self._features(s, v, ev))[:, 1]

    def accept(self, s: np.ndarray, v: np.ndarray, ev: np.ndarray) -> np.ndarray:
        return self.trust(s, v, ev) >= self.min_trust
