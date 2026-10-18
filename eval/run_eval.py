"""One results table for both tracks: tier1 vs always vs gated, on unseen threats.

Usage:
    python eval/run_eval.py --demo            # synthetic data, runs now
    python eval/run_eval.py --config configs/transactions.yaml   # once the track is filled in
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from riskcascade.core import Cascade, OverrideGate, metrics


def evaluate(cascade: Cascade, items, y, costs: dict) -> list[dict]:
    rows = []
    base = None
    for policy in ("tier1", "always", "gated"):
        r = cascade.run(items, policy=policy)
        if base is None:
            base = r.final
        rows.append({
            "policy": policy,
            "cost_per_item": metrics.cost_weighted_loss(y, r.final, **costs),
            **metrics.decision_rates(y, r.final),
            "recall@1%fpr_tier1": metrics.recall_at_fpr(y, r.tier1_scores, 0.01),
            "routed_share": r.routed_share if policy != "tier1" else 0.0,
            "tier1_ms_per_item": r.tier1_ms_per_item,
            **metrics.override_quality(y, base, r.final),
        })
    return rows


def demo():
    """Synthetic stand-in: tier 1 is noisy; tier 2 is right ~70% of the time,
    but more often right when its evidence feature is high. The gate should learn that."""
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)
    s = np.clip(y * 0.35 + rng.normal(0.35, 0.2, n), 0, 1)
    evidence = rng.random(n)
    p_right = 0.4 + 0.5 * evidence
    tier2_correct = rng.random(n) < p_right
    verdict_all = np.where(tier2_correct, y, 1 - y)

    class T1:
        def score(self, items):
            return s[np.asarray(items)]

    class T2:
        def judge(self, items, scores):
            idx = np.asarray(items)
            return verdict_all[idx], evidence[idx, None]

    idx = np.arange(n)
    val, test = idx[: n // 2], idx[n // 2:]
    gate = OverrideGate().fit(s[val], verdict_all[val], evidence[val, None], y[val])
    cascade = Cascade(T1(), T2(), gate=gate)
    return evaluate(cascade, list(test), y[test], {"c_fn": 10.0, "c_fp": 1.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()
    if args.demo:
        rows = demo()
    else:
        raise SystemExit("track wiring not implemented yet; run with --demo")
    for r in rows:
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))


if __name__ == "__main__":
    main()
