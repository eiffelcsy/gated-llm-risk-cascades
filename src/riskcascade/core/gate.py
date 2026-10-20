"""Override gate: should tier 2's disagreement with tier 1 be trusted?

Fit only on cases where tier 2 disagreed with tier 1. Target = 1 if tier 2
was right, so trust() estimates P(tier 2 is right | it disagrees).

Acceptance is cost-aware (the Bayes rule for a calibrated trust):
flip to "block" when trust * c_fn > (1 - trust) * c_fp,
flip to "allow" when trust * c_fp > (1 - trust) * c_fn.

Two knobs exist for the ablation study:
  model     logreg | gbdt | mlp
  features  score (tier-1 only) | verdict (+ tier-2 verdict) | all (+ evidence)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    # small and regularised: disagreement sets are only hundreds of rows
    "gbdt": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100, min_samples_leaf=20),
    "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=1e-2, max_iter=2000, random_state=0)),
}
FEATURE_SETS = ("score", "verdict", "all")


class OverrideGate:
    def __init__(self, threshold: float = 0.5, c_fn: float = 1.0, c_fp: float = 1.0,
                 model: str = "logreg", features: str = "all"):
        if model not in MODELS or features not in FEATURE_SETS:
            raise ValueError(f"model in {list(MODELS)}, features in {FEATURE_SETS}")
        self.threshold, self.c_fn, self.c_fp = threshold, c_fn, c_fp
        self.features, self.model = features, MODELS[model]()
        self._constant: float | None = None  # used when there is nothing to learn from

    def _x(self, s, v, ev) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        cols = [s, np.abs(s - self.threshold)]
        if self.features != "score":
            cols.append(np.asarray(v, dtype=float))
        x = np.column_stack(cols)
        if self.features == "all":
            x = np.column_stack([x, np.asarray(ev, dtype=float).reshape(len(s), -1)])
        return x

    def fit(self, s, v, ev, y) -> "OverrideGate":
        """Inputs are disagreement cases only; y is the true label."""
        target = (np.asarray(v) == np.asarray(y)).astype(int)
        if len(np.unique(target)) < 2:
            # none -> never override; one-sided -> use the observed rate
            self._constant = float(target.mean()) if len(target) else 0.0
            return self
        self._constant = None
        self.model.fit(self._x(s, v, ev), target)
        return self

    def trust(self, s, v, ev) -> np.ndarray:
        """P(tier 2 is right | it disagrees)."""
        if self._constant is not None:
            return np.full(len(s), self._constant)
        return self.model.predict_proba(self._x(s, v, ev))[:, 1]

    def accept(self, s, v, ev) -> np.ndarray:
        """Bool mask over disagreement cases: take tier 2's verdict."""
        if len(s) == 0:
            return np.zeros(0, dtype=bool)
        min_trust = np.where(np.asarray(v) == 1, self.c_fp, self.c_fn) / (self.c_fn + self.c_fp)
        return self.trust(s, v, ev) > min_trust
