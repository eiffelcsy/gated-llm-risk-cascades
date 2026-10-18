"""Tier-1 -> uncertainty band -> tier-2 -> override gate.

Both tracks plug in a Tier1 and a Tier2; everything else is shared.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from .gate import OverrideGate


class Tier1(Protocol):
    """Cheap scorer run on every item. Returns P(risky) in [0, 1]."""

    def score(self, items: Sequence[Any]) -> np.ndarray: ...


class Tier2(Protocol):
    """Expensive judge run only on the uncertain band.

    Returns (verdict in {0,1}, evidence features). Evidence features feed the
    override gate: e.g. evidence count, tool hits, self-consistency.
    """

    def judge(self, items: Sequence[Any], tier1_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class CascadeResult:
    tier1_scores: np.ndarray
    routed: np.ndarray            # bool mask: sent to tier 2
    tier2_verdicts: np.ndarray    # -1 where not routed
    final: np.ndarray             # final 0/1 decision
    overridden: np.ndarray        # bool mask: tier 2 changed tier 1's decision
    tier1_ms_per_item: float
    extras: dict = field(default_factory=dict)

    @property
    def routed_share(self) -> float:
        return float(self.routed.mean())


class Cascade:
    """Routes the uncertain band [low, high) of tier-1 scores to tier 2.

    Policies, selected with `policy`:
      - "tier1":     ignore tier 2 entirely (baseline)
      - "always":    tier 2's verdict always wins inside the band
      - "gated":     tier 2 wins only when the override gate trusts it
    """

    def __init__(self, tier1: Tier1, tier2: Tier2, low: float = 0.2, high: float = 0.8,
                 threshold: float = 0.5, gate: OverrideGate | None = None):
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("need 0 <= low < high <= 1")
        self.tier1, self.tier2 = tier1, tier2
        self.low, self.high, self.threshold = low, high, threshold
        self.gate = gate

    def run(self, items: Sequence[Any], policy: str = "gated") -> CascadeResult:
        t0 = time.perf_counter()
        s = np.asarray(self.tier1.score(items), dtype=float)
        t1_ms = (time.perf_counter() - t0) * 1000 / max(len(items), 1)

        base = (s >= self.threshold).astype(int)
        routed = (s >= self.low) & (s < self.high)
        verdicts = np.full(len(s), -1)
        final = base.copy()

        if policy != "tier1" and routed.any():
            idx = np.flatnonzero(routed)
            v, ev = self.tier2.judge([items[i] for i in idx], s[idx])
            v = np.asarray(v, dtype=int)
            verdicts[idx] = v
            disagree = v != base[idx]
            if policy == "always":
                accept = np.ones_like(disagree)
            elif policy == "gated":
                if self.gate is None:
                    raise ValueError("policy='gated' needs a fitted OverrideGate")
                accept = self.gate.accept(s[idx], v, ev)
            else:
                raise ValueError(f"unknown policy {policy!r}")
            take = disagree & accept
            final[idx[take]] = v[take]

        return CascadeResult(
            tier1_scores=s, routed=routed, tier2_verdicts=verdicts, final=final,
            overridden=final != base, tier1_ms_per_item=t1_ms,
        )
