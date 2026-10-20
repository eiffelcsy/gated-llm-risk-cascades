"""Tier-1 -> uncertainty band -> tier-2 -> override gate.

Both tracks plug in a Tier1 and a Tier2; everything else is shared.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .gate import OverrideGate
from .metrics import best_threshold

POLICIES = ("tier1", "always", "gated")


class Tier1(Protocol):
    """Cheap scorer run on every item. Returns P(risky) in [0, 1]."""

    def score(self, items: Any) -> np.ndarray: ...


class Tier2(Protocol):
    """Expensive judge run only on the uncertain band.

    Returns (verdicts, evidence features of shape (n, k)). A verdict is 0/1,
    or -1 to abstain (tier 1's decision stands). Evidence feeds the override gate.
    """

    def judge(self, items: Any, tier1_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def take(items: Any, idx: np.ndarray) -> Any:
    """Subset a list or a DataFrame by integer positions."""
    if hasattr(items, "iloc"):
        return items.iloc[idx]
    return [items[i] for i in idx]


@dataclass
class Trace:
    """Everything one pass of the cascade observed. Policies and gates are decided from it,
    so tier 2 runs once per item however many gates are compared."""

    scores: np.ndarray        # tier-1 P(risky)
    base: np.ndarray          # tier-1 decision
    routed: np.ndarray        # bool mask: inside the band
    verdicts: np.ndarray      # tier-2 verdict, -1 where not routed or abstained
    evidence: np.ndarray      # tier-2 evidence, zeros where not routed
    tier1_ms: float           # batch ms per item
    tier2_ms: float           # ms per routed item

    @property
    def routed_share(self) -> float:
        return float(self.routed.mean())

    def disagreements(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Routed items where tier 2 gave a verdict different from tier 1's."""
        d = self.routed & (self.verdicts >= 0) & (self.verdicts != self.base)
        return d if mask is None else d & mask


class Cascade:
    """Routes tier-1 scores in [low, high) to tier 2.

    Policies:
      tier1   ignore tier 2 (baseline)
      always  tier 2's verdict always wins inside the band (ungated cascade)
      gated   tier 2 wins only when the override gate trusts it
    """

    def __init__(self, tier1: Tier1, tier2: Tier2, threshold: float = 0.5,
                 low: float = 0.2, high: float = 0.8, gate: OverrideGate | None = None):
        if not 0.0 <= low <= threshold <= high <= 1.0 or low == high:
            raise ValueError("need 0 <= low <= threshold <= high <= 1 and low < high")
        self.tier1, self.tier2 = tier1, tier2
        self.threshold, self.low, self.high = threshold, low, high
        self.gate = gate

    def trace(self, items: Any) -> Trace:
        t0 = time.perf_counter()
        s = np.asarray(self.tier1.score(items), dtype=float)
        t1_ms = (time.perf_counter() - t0) * 1000 / max(len(s), 1)

        routed = (s >= self.low) & (s < self.high)
        idx = np.flatnonzero(routed)
        verdicts, evidence, t2_ms = np.full(len(s), -1), np.zeros((len(s), 1)), 0.0
        if len(idx):
            t0 = time.perf_counter()
            v, ev = self.tier2.judge(take(items, idx), s[idx])
            t2_ms = (time.perf_counter() - t0) * 1000 / len(idx)
            ev = np.asarray(ev, dtype=float).reshape(len(idx), -1)
            verdicts[idx], evidence = np.asarray(v, dtype=int), np.zeros((len(s), ev.shape[1]))
            evidence[idx] = ev
        return Trace(s, (s >= self.threshold).astype(int), routed, verdicts, evidence, t1_ms, t2_ms)

    def decide(self, tr: Trace, policy: str = "gated", gate: OverrideGate | None = None,
               mask: np.ndarray | None = None) -> np.ndarray:
        """Final 0/1 decisions without re-running either tier. `mask` limits which items may be overridden."""
        final = tr.base.copy()
        if policy == "tier1":
            return final
        cand = tr.disagreements(mask)
        if policy == "gated":
            gate = gate or self.gate
            if gate is None:
                raise ValueError("policy='gated' needs a fitted OverrideGate")
            if cand.any():
                cand[cand] = gate.accept(tr.scores[cand], tr.verdicts[cand], tr.evidence[cand])
        elif policy != "always":
            raise ValueError(f"unknown policy {policy!r}")
        final[cand] = tr.verdicts[cand]
        return final

    def run(self, items: Any, policy: str = "gated") -> np.ndarray:
        return self.decide(self.trace(items), policy)

    def fit_gate(self, tr: Trace, y, c_fn: float, c_fp: float, mask: np.ndarray | None = None,
                 **gate_kwargs) -> OverrideGate:
        """Fit an override gate on tier 2's disagreements in `tr` (validation data, or a recent window)."""
        d = tr.disagreements(mask)
        return OverrideGate(self.threshold, c_fn, c_fp, **gate_kwargs).fit(
            tr.scores[d], tr.verdicts[d], tr.evidence[d], np.asarray(y)[d])


def operating_point(scores: np.ndarray, y: np.ndarray, c_fn: float, c_fp: float,
                    budget: float) -> tuple[float, float, float]:
    """Pick (threshold, low, high) on validation data.

    threshold: minimises cost-weighted loss of tier 1 alone.
    band: the `budget` share of items whose scores sit closest to the threshold.
    """
    s = np.asarray(scores, dtype=float)
    thr = best_threshold(y, s, c_fn, c_fp)
    # band is defined in rank, not score, so skewed scores still route ~budget of traffic
    n = int(round(budget * len(s) / 2))
    below = np.sort(s[s < thr])[::-1][:n]   # n closest below the threshold
    above = np.sort(s[s >= thr])            # band takes the first n of these
    low = below[-1] if len(below) else thr
    high = above[n] if n < len(above) else 1.0
    high = max(high, np.nextafter(thr, 1.0))  # keep low < high when the band is empty
    return float(thr), float(low), float(min(high, 1.0))
